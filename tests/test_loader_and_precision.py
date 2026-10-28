"""Loader and mixed-precision equivalence of the wrapper training path.

The loader test runs on CPU on a synthetic manifest. The precision test needs a CUDA device.
Run from baselines/HITEC:  python -m pytest tests/test_loader_and_precision.py -q
"""

import os
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from emulator_bench import train  # noqa: E402

MANIFEST = Path(__file__).parent / "fixtures" / "manifest.csv"
OUTPUT_DIMS, LEVEL_STARTS = [2, 3, 4, 5], [0, 2, 5, 9]
BATCH = 4  # 10 fixture rows: two full batches and a short last batch


def fixture_dataset():
    return train.ManifestDataset(MANIFEST, OUTPUT_DIMS, LEVEL_STARTS)


def epoch_batches(num_workers, epochs=2):
    # The same arguments as train.main (batch size 4 instead of 64 so the 10 fixture rows give a short last batch).
    loader = DataLoader(fixture_dataset(), batch_size=BATCH, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    torch.manual_seed(0)
    return [batch for _ in range(epochs) for batch in loader]


def test_loader_batches_identical_for_any_worker_count():
    """Shuffle order, batch contents and the short last batch do not depend on num_workers."""
    base, workers = epoch_batches(0), epoch_batches(4)
    assert len(base) == len(workers) == 6
    assert [len(batch[0]) for batch in base[:3]] == [4, 4, 2]
    assert base[0][3].tolist() != sorted(base[0][3].tolist()) or base[1][3].tolist() != sorted(
        base[1][3].tolist()
    )  # the order is shuffled
    for got, want in zip(workers, base):
        for a, b in zip(got, want):
            assert torch.equal(a, b)


K = 10  # the production gap may be 10 times the fp32 noise (the production path with initial weights perturbed by one fp16 rounding) before it counts as a defect


U = 2**-11  # one rounding of the weights at the production precision (fp16)


def perturb(module, rel=U):
    generator = torch.Generator().manual_seed(123)
    with torch.no_grad():
        for p in module.parameters():
            p.add_(rel * p.abs() * torch.randn(p.shape, generator=generator).to(p.device))


def assert_within_noise(name, ref, noise, clean, prod, final_gap, final_floor, final_scale):
    """Per-step loss gap of the production path against the reference, bounded by K times the noise floor."""
    ref, noise, clean, prod = (torch.as_tensor(x, dtype=torch.float64) for x in (ref, noise, clean, prod))
    floor, gap = (noise - clean).abs().max().item(), (prod - ref).abs().max().item()
    tiny, tiny0 = 1e-7 * ref.abs().max().item(), 1e-7 * ref[0].abs().item()
    gap0, floor0 = (prod[0] - ref[0]).abs().item(), (noise[0] - clean[0]).abs().item()
    print(
        f"{name}: max gap={gap:.3e}, max floor={floor:.3e}, ratio={gap / max(floor, tiny):.2f}, K={K}, "
        f"step0 gap={gap0:.3e}, floor={floor0:.3e}, ratio={gap0 / max(floor0, tiny0):.2f}; final gap={final_gap:.3e}, final floor={final_floor:.3e}"
    )
    assert gap0 <= K * max(floor0, tiny0)
    assert gap <= K * max(floor, tiny)
    assert final_gap <= K * max(final_floor, 1e-7 * final_scale)


@pytest.mark.gpu
def test_production_fp16_compiled_training_tracks_fp32_eager(monkeypatch):
    """Production path (compiled model, fp16 autocast, GradScaler, fused AdamW) against eager fp32 with plain AdamW (the released optimizer).

    Same seed, same initial weights, same batches, 20 steps on the fixture rows with a model of
    dimension 64 and dropout 0 (compiled dropout draws a different random stream than eager dropout).
    The reference is eager fp32; the noise control is the production path with initial weights perturbed
    by 2**-11 relative (the fp16 unit roundoff). Bounds: max per-step loss gap and final mean absolute prediction gap each within K times
    their noise floor; the step-0 loss gap within K times its floor.
    The released code also trains in fp16, so the gap to fp32 bounds the gap to the released run.
    """
    monkeypatch.setattr(train, "MODEL_DIMENSION", 64)
    monkeypatch.setitem(train.CONFIG, "lr", 1e-3)
    monkeypatch.setitem(train.CONFIG, "dr", 0.0)
    dataset = fixture_dataset()
    rows = [dataset[i] for i in range(len(dataset))]
    tokens = torch.stack([r[0][:64] for r in rows]).cuda()
    target = torch.stack([r[1] for r in rows]).cuda()
    mask = torch.stack([r[2] for r in rows]).cuda()

    def run(production, noise=False, scale=1.0):
        torch.manual_seed(0)
        module = train.HitecModule(OUTPUT_DIMS).cuda()
        module.log = lambda *args, **kwargs: None
        if noise:
            perturb(module)
        if production:
            module.model.compile(dynamic=False, mode="max-autotune-no-cudagraphs")
        with torch.no_grad():
            for p in module.parameters():
                p.mul_(scale)
        optimizer = module.configure_optimizers()
        if not production:  # the released code uses plain (unfused) AdamW with the same groups
            groups = [{k: g[k] for k in ("params", "lr", "weight_decay")} for g in optimizer.param_groups]
            optimizer = torch.optim.AdamW(groups, lr=train.CONFIG["lr"])
        scaler = torch.amp.GradScaler("cuda", enabled=production)
        losses = []
        for _ in range(20):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=production):
                loss = module.training_step((tokens, target, mask, None), 0)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
        with torch.no_grad():
            return torch.tensor(losses), torch.sigmoid(module.eval()(tokens).float())

    ref, noise, clean = run(False), run(True, True), run(True)
    # The negative control scales only the gap run's initial weights; the floor keeps the unscaled production run.
    prod = run(True, scale=1.05) if os.environ.get("EQUIV_NEGATIVE_CONTROL") == "1" else clean
    assert_within_noise(
        "hitec fp16+compile",
        ref[0], noise[0], clean[0], prod[0],
        (prod[1] - ref[1]).abs().mean().item(), (noise[1] - clean[1]).abs().mean().item(), ref[1].abs().mean().item(),
    )

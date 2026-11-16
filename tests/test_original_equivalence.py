"""Equivalence of the emulator_bench wrapper with the authors' released HIT-EC code.

The original model/model.py and training/infer_training.py are read with `git show` from the
commit that holds the authors' code. Tests use a tiny model (dimension 64) in fp64 on CPU.

Run from baselines/HITEC:  CUDA_VISIBLE_DEVICES= python -m pytest tests/test_original_equivalence.py -q
"""

import importlib.util
import math
import subprocess
import sys
from pathlib import Path

import pytest
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import StochasticWeightAveraging
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from emulator_bench import train  # noqa: E402

ORIGINAL_COMMIT = "15ca6cd"  # last commit before the wrapper changed model/model.py
DIM, OUTPUT_DIMS, BATCH = 64, [3, 5, 7, 9], 4
LR = 1e-3  # larger than the released 8.75e-5 so that weight changes dominate fp64 rounding
# fp64 round-off of different but algebraically equal summation orders is ~1e-14; these bounds leave margin.
FWD_ATOL = 1e-10
WEIGHT_ATOL = 1e-10  # after Adam steps, which amplify round-off a little; observed ~3e-13


def git_show(path):
    return subprocess.run(
        ["git", "show", "{}:{}".format(ORIGINAL_COMMIT, path)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


@pytest.fixture(scope="module")
def original(tmp_path_factory):
    """(Transformer, Model) of the released code; Model is infer_training.Model without its broken training_step."""
    path = tmp_path_factory.mktemp("original") / "original_model.py"
    path.write_text(git_show("model/model.py"))
    spec = importlib.util.spec_from_file_location("original_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    source = git_show("training/infer_training.py")
    source = source.split("\nconfig = {")[0].replace("from ..model.model import Transformer", "")
    namespace = {"Transformer": module.Transformer}
    exec(compile(source, "infer_training.py", "exec"), namespace)
    base = namespace["Model"]

    class Model(base):
        def training_step(self, batch, batch_idx):
            tokens, target, mask, _ = batch
            return original_loss(self.model(tokens, mode="infer"), target, mask)

    return module.Transformer, Model


def original_loss(final, label, mask):
    """The released loss: BCE-with-logits summed over observed entries, divided by their count."""
    observed = mask.bool()
    observed_values = torch.where(observed, label, torch.zeros_like(label))
    observed_pred = torch.where(observed, final, torch.zeros_like(final))
    return nn.BCEWithLogitsLoss(reduction="sum")(observed_pred, observed_values) / torch.sum(
        observed
    )


def make_data(seed=0):
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.randint(1, 23, (BATCH, DIM), generator=generator)
    tokens[:, 0] = 22
    tokens[0, 20:] = 0
    tokens[1, 45:] = 0
    total = sum(OUTPUT_DIMS)
    target = (torch.rand(BATCH, total, generator=generator) < 0.3).double()
    mask = torch.zeros(BATCH, total, dtype=torch.double)
    starts = [0, 3, 8, 15]
    for row, levels in enumerate([[0, 1, 2, 3], [0, 1], [0], [0, 1, 2]]):
        for level in levels:
            mask[row, starts[level] : starts[level] + OUTPUT_DIMS[level]] = 1.0
    return tokens, target, mask


@pytest.fixture
def small_config(monkeypatch):
    monkeypatch.setitem(train.CONFIG, "dr", 0.0)
    monkeypatch.setitem(train.CONFIG, "lr", LR)
    monkeypatch.setattr(train, "MODEL_DIMENSION", DIM)


def original_config(dropout=0.0):
    return {
        "ah": 2,
        "dr": dropout,
        "beta": 0.59,
        "lr": LR,
        "observed_loss": 0.96,
        "alpha": 0.93,
        "dimension": DIM,
        "output_dims": OUTPUT_DIMS,
    }


def paired_modules(original, sdpa=False):
    """Wrapper module and original Model with identical weights."""
    _, Model = original
    torch.manual_seed(0)
    theirs = Model(original_config(), dimension=DIM, output_dims=OUTPUT_DIMS).double()
    ours = train.HitecModule(OUTPUT_DIMS, sdpa).double()
    ours.load_state_dict(theirs.state_dict())
    return ours, theirs


def max_weight_diff(a, b):
    assert list(a.state_dict()) == list(b.state_dict())
    return max(
        (x - y).abs().max().item()
        for x, y in zip(a.state_dict().values(), b.state_dict().values())
    )


def manual_steps(module, optimizer, loss_fn, steps):
    tokens, target, mask = make_data()
    losses = []
    for _ in range(steps):
        optimizer.zero_grad()
        loss = loss_fn(module, tokens, target, mask)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


@pytest.mark.parametrize("sdpa", [False, True])
def test_eval_forward_matches_original(small_config, original, sdpa):
    ours, theirs = paired_modules(original, sdpa)
    tokens, _, _ = make_data()
    with torch.no_grad():
        got, want = ours.eval()(tokens), theirs.eval().model(tokens, mode="infer")
    assert (got - want).abs().max() < FWD_ATOL


def test_eval_forward_matches_original_with_released_dropout(monkeypatch, original):
    """Dropout 0.1 (the released and wrapper setting) is inactive in eval."""
    monkeypatch.setattr(train, "MODEL_DIMENSION", DIM)
    _, Model = original
    torch.manual_seed(0)
    theirs = Model(original_config(0.1), dimension=DIM, output_dims=OUTPUT_DIMS).double().eval()
    ours = train.HitecModule(OUTPUT_DIMS).double().eval()
    ours.load_state_dict(theirs.state_dict())
    assert {m.p for m in ours.modules() if isinstance(m, nn.Dropout)} == {0.1}
    tokens, _, _ = make_data()
    with torch.no_grad():
        assert (ours(tokens) - theirs.model(tokens, mode="infer")).abs().max() < FWD_ATOL


def loss_pair(original):
    ours, theirs = paired_modules(original)
    tokens, target, mask = make_data()
    ours.log = lambda *args, **kwargs: None
    got = ours.training_step((tokens, target, mask, torch.arange(BATCH)), 0)
    want = theirs.training_step((tokens, target, mask, None), 0)
    return got.item(), want.item(), mask


@pytest.mark.xfail(
    strict=True,
    reason="released loss also sums log(2) for each unobserved entry (logit 0, target 0); wrapper omits that constant",
)
def test_loss_value_equals_original(small_config, original):
    got, want, _ = loss_pair(original)
    assert abs(got - want) < FWD_ATOL


def test_loss_differs_from_original_only_by_constant_unobserved_term(small_config, original):
    got, want, mask = loss_pair(original)
    unobserved, observed = (mask == 0).sum().item(), mask.sum().item()
    assert abs(want - got - unobserved * math.log(2) / observed) < FWD_ATOL


def test_gradients_match_original(small_config, original):
    ours, theirs = paired_modules(original)
    tokens, target, mask = make_data()
    ours.log = lambda *args, **kwargs: None
    ours.training_step((tokens, target, mask, None), 0).backward()
    theirs.training_step((tokens, target, mask, None), 0).backward()
    for (name, a), (_, b) in zip(ours.named_parameters(), theirs.named_parameters()):
        ga = torch.zeros_like(a) if a.grad is None else a.grad
        gb = torch.zeros_like(b) if b.grad is None else b.grad
        assert (ga - gb).abs().max() < FWD_ATOL, name


def test_param_groups_match_original(small_config, original):
    ours, theirs = paired_modules(original)
    got = ours.configure_optimizers().param_groups
    want = theirs.configure_optimizers()["optimizer"].param_groups
    assert [(g["lr"], g["weight_decay"]) for g in got] == [
        (g["lr"], g["weight_decay"]) for g in want
    ]


def test_weights_and_predictions_after_steps_match_original(small_config, original):
    ours, theirs = paired_modules(original)
    ours.train()
    theirs.train()
    ours.log = lambda *args, **kwargs: None
    opt_ours = ours.configure_optimizers()
    opt_theirs = theirs.configure_optimizers()["optimizer"]
    manual_steps(ours, opt_ours, lambda m, t, y, k: m.training_step((t, y, k, None), 0), 3)
    manual_steps(theirs, opt_theirs, lambda m, t, y, k: m.training_step((t, y, k, None), 0), 3)
    assert max_weight_diff(ours, theirs) < WEIGHT_ATOL
    start = paired_modules(original)[0]
    assert max_weight_diff(ours, start) > 1e-5  # the steps changed the weights
    tokens, _, _ = make_data()
    with torch.no_grad():
        got = torch.sigmoid(ours.eval()(tokens))
        want = torch.sigmoid(theirs.eval().model(tokens, mode="infer"))
    assert (got - want).abs().max() < WEIGHT_ATOL


def test_compiled_forward_and_gradients_match_original(small_config, original):
    """The Inductor path on CPU against the eager original, in fp32 (Inductor's CPU attention backward has no fp64)."""
    ours, theirs = paired_modules(original)
    ours.float().model.compile(dynamic=False)
    theirs.float()
    tokens, target, mask = make_data()
    target, mask = target.float(), mask.float()
    ours.log = lambda *args, **kwargs: None
    logits = ours(tokens)
    ours.training_step((tokens, target, mask, None), 0).backward()
    theirs.training_step((tokens, target, mask, None), 0).backward()
    with torch.no_grad():
        assert (
            logits - theirs.model(tokens, mode="infer")
        ).abs().max() < 1e-5  # fp32 round-off ~1e-6
    for (name, a), (_, b) in zip(ours.named_parameters(), theirs.named_parameters()):
        ga = torch.zeros_like(a) if a.grad is None else a.grad
        gb = torch.zeros_like(b) if b.grad is None else b.grad
        assert (ga - gb).abs().max() < 1e-5, name  # gradients are O(1e-2); fp32 round-off ~1e-7


def run_trainer(module, seed=0, epochs=6, **trainer_kwargs):
    loader = DataLoader(TensorDataset(*make_data(), torch.arange(BATCH)), batch_size=BATCH)
    pl.seed_everything(seed)
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="cpu",
        devices=1,
        callbacks=[StochasticWeightAveraging(swa_lrs=3e-7, swa_epoch_start=3, annealing_epochs=2)],
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        **trainer_kwargs,
    )
    trainer.fit(module, loader)
    return module


def run_wrapper_main(monkeypatch, tmp_path, original, stop_at=None, epochs=6):
    """train.main on the tiny tensors; returns (exit code, seed root). Compile runs with Dynamo disabled."""
    ticks = iter(range(0, 10_000, 10))
    data = TensorDataset(*make_data(), torch.arange(BATCH))
    metadata = {
        "run_root": str(tmp_path),
        "manifests": {"train": ""},
        "output_dims": OUTPUT_DIMS,
        "level_starts": [0, 3, 8, 15],
    }
    argv = [
        "train",
        "--split-group",
        "g",
        "--seed",
        "0",
        "--num-workers",
        "0",
        "--accelerator",
        "cpu",
    ]
    argv += ["--epochs", str(epochs), "--swa-epoch-start", "3", "--swa-annealing-epochs", "2"]
    argv += [] if stop_at is None else ["--stop-at", str(stop_at)]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(
        train, "time", type("Clock", (), {"time": staticmethod(lambda: next(ticks))})
    )
    monkeypatch.setattr(train, "load_run_metadata", lambda *args: metadata)
    monkeypatch.setattr(train, "ManifestDataset", lambda *args: data)
    monkeypatch.setattr(
        train, "DataLoader", lambda dataset, **kwargs: DataLoader(dataset, batch_size=BATCH)
    )
    monkeypatch.setattr(train, "HitecModule", lambda dims, sdpa: HitecDouble(dims, sdpa))
    seed_everything = pl.seed_everything
    monkeypatch.setattr(train.pl, "seed_everything", lambda *args, **kwargs: seed_everything(0))
    try:
        train.main()
        code = 0
    except SystemExit as exit_:
        code = exit_.code
    return code, tmp_path / "seed_0"


class HitecDouble(train.HitecModule):
    """The wrapper module with the weights of the original Model (seed 0) in fp64."""

    def __init__(self, output_dims, sdpa=False):
        super().__init__(output_dims, sdpa)
        self.double()


def seeded_original(original):
    _, Model = original
    torch.manual_seed(0)
    return Model(original_config(), dimension=DIM, output_dims=OUTPUT_DIMS).double()


def seeded_wrapper_init(monkeypatch):
    """Make HitecDouble start from the same weights as seeded_original."""
    init = train.HitecModule.__init__

    def seeded(self, output_dims, sdpa=False):
        torch.manual_seed(0)
        init(self, output_dims, sdpa)

    monkeypatch.setattr(train.HitecModule, "__init__", seeded)


def load_weights(path):
    return torch.load(path, weights_only=False)["state_dict"]


def state_diff(a, b):
    assert list(a) == list(b)
    return max((a[key] - b[key]).abs().max().item() for key in a)


@pytest.fixture
def original_trained(original, small_config):
    """The released Model after 6 epochs with SWA on the Lightning trainer (final SWA weights)."""
    start = seeded_original(original).state_dict()
    trained = run_trainer(seeded_original(original)).state_dict()
    assert state_diff(trained, start) > 1e-5  # training moved the weights
    return trained


def test_eager_fallback_training_matches_original(
    monkeypatch, tmp_path, original, small_config, original_trained
):
    """train.main with torch.compile turned off (TORCHDYNAMO_DISABLE=1 behaviour) vs the released trainer."""
    monkeypatch.setattr(torch._dynamo.config, "disable", True)
    seeded_wrapper_init(monkeypatch)
    code, root = run_wrapper_main(monkeypatch, tmp_path, original)
    assert code == 0
    assert state_diff(load_weights(root / "lastmodel.pth"), original_trained) < WEIGHT_ATOL


def test_resumed_chain_training_matches_original(
    monkeypatch, tmp_path, original, small_config, original_trained
):
    """Stop after epoch 4 (inside SWA), resume from checkpoint_last.pt, finish; same weights as one released run."""
    monkeypatch.setattr(torch._dynamo.config, "disable", True)
    seeded_wrapper_init(monkeypatch)
    code, root = run_wrapper_main(monkeypatch, tmp_path, original, stop_at=45)
    assert code == train.UNFINISHED
    assert (root / "checkpoint_last.pt").exists() and not (root / "lastmodel.pth").exists()
    code, root = run_wrapper_main(monkeypatch, tmp_path, original)
    assert code == 0
    assert state_diff(load_weights(root / "lastmodel.pth"), original_trained) < WEIGHT_ATOL

import sys
from pathlib import Path

import json
import types

import pandas as pd
import pytorch_lightning as pl
import pytest
import torch
from torch.utils.data import TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emulator_bench import evaluate, run_benchmarks, train
from model.model import Transformer
from emulator_bench.dataset_adapter import (
    SplitGroup,
    discover_split_groups,
    load_hitec_records,
    prepare_split_group,
    select_split_groups,
)
from emulator_bench.utils import seed_run_root, split_run_root


def make_tree(root):
    for folder in (
        "random_splits_grouped_sequence",
        "uniprot_time_splits",
        "enzyme_sequence_splits/threshold_0.39",
        "enzyme_structure_splits/threshold_0.85",
        "enzyme_structure_splits/backup_2026-10-21_thresholds_0.7_0.8_0.9",
        "enzyme_structure_splits/backup_2026-10-21_thresholds_0.7_0.8_0.9/threshold_0.7",
    ):
        (root / folder).mkdir(parents=True)
        for split in ("train", "val", "test"):
            (root / folder / f"{split}.parquet").touch()


def test_discovery_includes_structure_and_skips_backup(tmp_path):
    make_tree(tmp_path)
    names = [group.name for group in discover_split_groups(tmp_path)]
    assert names == [
        "random_splits_grouped_sequence",
        "uniprot_time_splits",
        "enzyme_sequence_splits/threshold_0.39",
        "enzyme_structure_splits/threshold_0.85",
    ]


def test_backup_cannot_be_requested(tmp_path):
    make_tree(tmp_path)
    with pytest.raises(ValueError):
        select_split_groups(
            tmp_path, ["enzyme_structure_splits/backup_2026-10-21_thresholds_0.7_0.8_0.9"]
        )


def test_run_layout(tmp_path):
    flat = {"run_root": str(split_run_root(tmp_path, "uniprot_time_splits"))}
    assert (
        split_run_root(tmp_path, "uniprot_time_splits")
        == tmp_path / "enzyme_classification/uniprot_time_splits/all"
    )
    assert (
        seed_run_root(flat, 666)
        == tmp_path / "enzyme_classification/uniprot_time_splits/all/seed_666"
    )
    assert (
        split_run_root(tmp_path, "enzyme_structure_splits/threshold_0.85")
        == tmp_path / "enzyme_classification/enzyme_structure_splits/threshold_0.85"
    )


def test_batch_size_default_is_64(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train", "--split-group", "g", "--seed", "1"])
    assert train.parse_args().batch_size == 64
    monkeypatch.setattr(sys, "argv", ["run_benchmarks"])
    assert run_benchmarks.parse_args().batch_size == 64


def test_compile_keeps_state_dict_keys():
    module = train.HitecModule([2, 3, 4, 5])
    keys = list(module.state_dict())
    module.model.compile(dynamic=False)
    assert list(module.state_dict()) == keys


def reference_logits(model, tokens):
    """The released forward: full attention for every layer, output viewed as (length, batch, dim, h) and summed over h."""

    def block(layer, x):
        attn = layer.attention
        x_norm = attn.norm(x)
        query, key = (
            attn.transpose_for_scores(attn.query(x_norm)),
            attn.transpose_for_scores(attn.key(x_norm)),
        )
        value = attn.transpose_for_scores_value(attn.value(x_norm))
        probs = torch.softmax(query @ key.transpose(-2, -1) / attn.d_k**0.5, dim=-1)
        out = (
            (probs @ value)
            .view(-1, x.size(0), attn.dim, attn.h)
            .sum(-1)
            .view(x.size(0), -1, attn.dim)
        )
        x = x + out
        return layer.dropout(layer.output_sublayer(x, x))

    emb = model.embeddings(tokens)
    x, local = emb, []
    for index, (layer, head) in enumerate(
        zip(
            (model.enc_1, model.enc_2, model.enc_3, model.enc_4),
            (model.linear1, model.linear2, model.linear3, model.linear4),
        )
    ):
        x = block(layer, x if index == 0 else x + emb)
        local.append(head(x[:, 0]))
    return model.beta * model.linear5(x[:, 0]) + (1 - model.beta) * torch.cat(local, dim=1)


def test_model_matches_released_forward():
    torch.manual_seed(0)
    model = Transformer(23, 64, 2, [3, 5, 7, 9], 0.0, 0.59)
    tokens = torch.randint(1, 23, (4, 64))
    tokens[0, 20:] = 0
    logits, expected = model(tokens, mode="infer"), reference_logits(model, tokens)
    assert torch.allclose(logits, expected, atol=1e-5)
    grads = torch.autograd.grad(
        logits.square().mean(),
        list(model.linear5.parameters()) + list(model.enc_4.parameters()),
        allow_unused=True,
    )
    expected_grads = torch.autograd.grad(
        expected.square().mean(),
        list(model.linear5.parameters()) + list(model.enc_4.parameters()),
        allow_unused=True,
    )
    for got, want in zip(grads, expected_grads):
        assert (got is None) == (want is None)
        if got is not None:
            assert torch.allclose(got, want, atol=1e-6)


def test_dropout_forward_backward_runs():
    torch.manual_seed(0)
    module = train.HitecModule([2, 3, 4, 5]).train()
    tokens = torch.randint(1, 23, (2, 1024))
    target = torch.zeros(2, 14)
    target[:, 0] = 1
    first = module(tokens)
    assert not torch.allclose(first, module(tokens))
    loss = module.criterion(first, target).mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        torch.isfinite(p.grad).all() for p in module.model.enc_4.parameters() if p.grad is not None
    )


def test_param_groups_match_released_optimizer_rules():
    module = train.HitecModule([2, 3, 4, 5])
    groups = module.configure_optimizers().param_groups
    base = train.CONFIG["lr"]
    expected = []
    for name, _ in module.named_parameters():
        weight_decay = 0.0 if any(word in name for word in ("norm", "bias")) else 0.01
        if name.startswith(("model.embeddings", "model.enc_1")):
            lr = base * 0.9**3
        elif name.startswith("model.enc_2"):
            lr = base * 0.9**2
        elif name.startswith("model.enc_3"):
            lr = base * 0.9
        elif name.startswith("model.enc_4"):
            lr = base
        else:
            assert "linear" in name
            lr = base * 1.05
        expected.append((lr, weight_decay))
    assert [(group["lr"], group["weight_decay"]) for group in groups] == expected
    assert len(groups) == len(list(module.parameters()))


def test_manifest_dataset_targets_and_masks(tmp_path):
    manifest = tmp_path / "train.csv"
    pd.DataFrame(
        {
            "Sequence": ["MKV"],
            "target_indices": [json.dumps([[0], [1], [], []])],
            "observed_levels": [json.dumps([0, 1, 3])],
        }
    ).to_csv(manifest, index=False)
    dataset = train.ManifestDataset(manifest, [2, 3, 4, 5], [0, 2, 5, 9])
    tokens, target, mask, index = dataset[0]
    assert (
        tokens.shape == (1024,)
        and tokens[0] == 22
        and (tokens[4:] == 0).all()
        and (tokens[1:4] > 0).all()
    )
    assert target.nonzero().flatten().tolist() == [0, 1]
    assert mask.nonzero().flatten().tolist() == [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
    assert index == 0


VOCAB = {
    "levels": [["1"], ["1.1"], ["1.1.1"], ["1.1.1.1", "1.1.1.2"]],
    "level_starts": [0, 1, 2, 3],
    "output_dims": [1, 1, 1, 2],
}


def write_split(path, rows):
    pd.DataFrame(rows, columns=["uniprot_id", "sequence", "ec_number"]).to_parquet(path)


def test_records_keep_only_full_ecs_and_union_labels(tmp_path):
    split = tmp_path / "train.parquet"
    write_split(
        split,
        [
            ("P1", "MKV", "1.1.1.1"),
            ("P1", "MKV", "1.1.1.2"),
            ("P2", "AAA", "1.1.1.-;1.1.1.2"),
            ("P3", "CCC", None),
            ("P4", "DDD", "1.1.1.-"),
            ("P5", "", "1.1.1.1"),
        ],
    )
    records, stats = load_hitec_records(split, "train", VOCAB)
    assert records["Entry"].tolist() == ["P1", "P2"]
    first, second = records.iloc[0], records.iloc[1]
    assert first["EC number"] == "1.1.1.1;1.1.1.2"
    assert json.loads(first["target_indices"]) == [[0], [1], [2], [3, 4]]
    assert second["EC number"] == "1.1.1.2"
    assert json.loads(second["target_indices"]) == [[0], [1], [2], [4]]
    assert json.loads(second["observed_levels"]) == [0, 1, 2, 3]
    assert (
        stats["missing_required_rows"],
        stats["dropped_no_full_ec_rows"],
        stats["rows_after_label_filter"],
    ) == (
        1,
        1,
        2,
    )
    assert evaluate.dropped_rows(stats) == {
        "source_rows": 6,
        "missing_required_fields": 1,
        "empty_sequence": 1,
        "proteins_without_full_ec": 1,
        "proteins_kept": 2,
    }


def test_prepare_split_group_skips_matching_fingerprint(tmp_path):
    folder = tmp_path / "data" / "flat"
    folder.mkdir(parents=True)
    for name in ("train", "val", "test"):
        write_split(folder / f"{name}.parquet", [("P1", "MKV", "1.1.1.1")])
    group = SplitGroup("flat", folder)
    out = tmp_path / "out"
    first = prepare_split_group(group, tmp_path, VOCAB, out, tmp_path / "vocab.json")
    manifest = out / "enzyme_classification/flat/all/manifests/train.csv"
    before = manifest.stat().st_mtime_ns
    assert prepare_split_group(group, tmp_path, VOCAB, out, tmp_path / "vocab.json") == first
    assert manifest.stat().st_mtime_ns == before
    write_split(folder / "train.parquet", [("P1", "MKVA", "1.1.1.1")])
    prepare_split_group(group, tmp_path, VOCAB, out, tmp_path / "vocab.json")
    assert manifest.stat().st_mtime_ns > before


def test_finished_training_runs_evaluation_only(monkeypatch):
    modules = []
    monkeypatch.setattr(
        run_benchmarks, "run", lambda module, *args, gpu=None: modules.append(module)
    )
    args = types.SimpleNamespace(
        out_root="o",
        batch_size=64,
        num_workers=0,
        accelerator="cpu",
        amp=False,
        sdpa=False,
        stop_at=None,
        epochs=None,
        swa_epoch_start=None,
        swa_annealing_epochs=None,
    )
    run_benchmarks.run_seed(args, "g", 1, True, None)
    run_benchmarks.run_seed(args, "g", 1, False, None)
    assert modules == ["evaluate", "train", "evaluate"]


def test_defaults_sdpa_off_and_four_workers(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train", "--split-group", "g", "--seed", "1"])
    args = train.parse_args()
    assert (args.sdpa, args.num_workers) == (False, 4)
    monkeypatch.setattr(sys, "argv", ["run_benchmarks"])
    args = run_benchmarks.parse_args()
    assert (args.sdpa, args.num_workers) == (False, 4)
    hparams = json.loads((Path(train.__file__).parent / "default_hparams.json").read_text())
    assert hparams["training"]["sdpa"]["value"] is False
    assert hparams["training"]["num_workers"]["value"] == 4


def test_sdpa_matches_manual_attention():
    torch.manual_seed(0)
    manual = Transformer(23, 64, 2, [3, 5, 7, 9], 0.0, 0.59)
    fused = Transformer(23, 64, 2, [3, 5, 7, 9], 0.0, 0.59, sdpa=True)
    fused.load_state_dict(manual.state_dict())
    tokens = torch.randint(1, 23, (4, 64))
    out_manual, out_fused = manual(tokens, mode="infer"), fused(tokens, mode="infer")
    assert torch.allclose(out_manual, out_fused, atol=1e-5)
    grads = [
        torch.autograd.grad(out.square().mean(), model.enc_1.attention.query.weight)[0]
        for out, model in ((out_manual, manual), (out_fused, fused))
    ]
    assert torch.allclose(*grads, atol=1e-6)


class TinyModule(pl.LightningModule):
    def __init__(self, output_dims, sdpa=False):
        super().__init__()
        self.model = types.SimpleNamespace(compile=lambda **kwargs: None)
        self.layer = torch.nn.Linear(2, 1)

    def training_step(self, batch, batch_idx):
        return self.layer(batch[0]).square().mean()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


def run_tiny_train(monkeypatch, tmp_path, stop_at, epochs=6):
    """Runs train.main on CPU; the fake clock advances 10 s per call, so each epoch takes 10 s."""
    ticks = iter(range(0, 10_000, 10))
    metadata = {"run_root": str(tmp_path), "manifests": {"train": ""}, "output_dims": [1], "level_starts": [0]}
    argv = ["train", "--split-group", "g", "--seed", "1", "--num-workers", "0", "--accelerator", "cpu"]
    argv += ["--epochs", str(epochs), "--swa-epoch-start", "1", "--swa-annealing-epochs", "2"]
    argv += [] if stop_at is None else ["--stop-at", str(stop_at)]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(train, "time", types.SimpleNamespace(time=lambda: next(ticks)))
    monkeypatch.setattr(train, "load_run_metadata", lambda *args: metadata)
    monkeypatch.setattr(train, "ManifestDataset", lambda *args: TensorDataset(torch.ones(4, 2)))
    monkeypatch.setattr(train, "HitecModule", TinyModule)
    monkeypatch.setattr(train, "write_json", lambda *args: None)
    seed_root = tmp_path / "seed_1"
    try:
        train.main()
        code = 0
    except SystemExit as exit_:
        code = exit_.code
    return code, seed_root


def test_stop_at_stops_after_checkpoint_without_final_model(monkeypatch, tmp_path):
    # Epoch n ends at 10 n. Epoch 4 ends at 40 and 40 + 1.2 * 10 > 45 (epoch 3: 42 <= 45).
    code, seed_root = run_tiny_train(monkeypatch, tmp_path, 45)
    assert code == train.UNFINISHED == 75
    # The checkpoint of the fourth epoch stores the zero-based epoch index.
    assert torch.load(seed_root / "checkpoint_last.pt", weights_only=False)["epoch"] == 3
    assert not (seed_root / "lastmodel.pth").exists()


def test_stop_at_none_and_far_stop_at_finish(monkeypatch, tmp_path):
    for stop_at in (None, 1e9):
        code, seed_root = run_tiny_train(monkeypatch, tmp_path / str(stop_at), stop_at)
        assert code == 0
        assert (seed_root / "lastmodel.pth").exists()


def test_stop_at_does_not_stop_after_last_epoch(monkeypatch, tmp_path):
    code, seed_root = run_tiny_train(monkeypatch, tmp_path, 1, epochs=1)
    assert code == 0 and (seed_root / "lastmodel.pth").exists()


def test_unfinished_training_skips_evaluation_and_exits_75(monkeypatch):
    calls = []

    def fake_run(module, *args, gpu=None):
        calls.append(module)
        return 75 if module == "train" else 0

    monkeypatch.setattr(run_benchmarks, "run", fake_run)
    args = types.SimpleNamespace(
        out_root="o", batch_size=64, num_workers=0, accelerator="cpu", amp=False, sdpa=False,
        stop_at=5.0, epochs=None, swa_epoch_start=None, swa_annealing_epochs=None,
    )
    assert run_benchmarks.run_seed(args, "g", 1, False, None) is True
    assert calls == ["train"]

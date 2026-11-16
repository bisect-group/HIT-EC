from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import Callback, StochasticWeightAveraging
from torch.utils.data import DataLoader, Dataset

from model.model import Transformer

from .utils import (
    DEFAULT_OUT_ROOT,
    MODEL_DIMENSION,
    ensure_dir,
    load_residue_ids,
    load_run_metadata,
    seed_run_root,
    seed_train_metadata_path,
    tokenize_sequence,
    write_json,
)

# Settings of the released HIT-EC code (training/infer_training.py) and the paper (Results, Methods).
CONFIG = {
    "ah": 2,
    "dr": 0.1,
    "beta": 0.59,
    "lr": 8.75e-5,
    "vocab_size": 23,
    "epochs": 80,
    "swa_lrs": 3e-7,
    "swa_epoch_start": 49,
    "swa_annealing_epochs": 30,
}

UNFINISHED = 75  # exit code: stopped early, resume from checkpoint_last.pt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train HIT-EC on an EMULaToR split with the original settings"
    )
    parser.add_argument("--split-group", required=True)
    parser.add_argument("--out_root", "--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--accelerator", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument(
        "--amp", action="store_true", help="bf16 autocast instead of the original fp16"
    )
    parser.add_argument(
        "--sdpa",
        action="store_true",
        help="F.scaled_dot_product_attention in the three full attention layers",
    )
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"], help="Smoke tests only")
    parser.add_argument(
        "--swa-epoch-start", type=int, default=CONFIG["swa_epoch_start"], help="Smoke tests only"
    )
    parser.add_argument(
        "--swa-annealing-epochs",
        type=int,
        default=CONFIG["swa_annealing_epochs"],
        help="Smoke tests only",
    )
    parser.add_argument(
        "--stop-at",
        type=float,
        default=None,
        help="Unix time. Stop after an epoch if the next one may end later; exit code 75",
    )
    return parser.parse_args()


class ManifestDataset(Dataset):
    """Tokenized sequence, multi-hot target and observed-level mask for each manifest row."""

    def __init__(self, manifest_path, output_dims, level_starts):
        records = pd.read_csv(manifest_path)
        self.sequences = records["Sequence"].tolist()
        self.target_indices = records["target_indices"].map(json.loads).tolist()
        self.observed_levels = records["observed_levels"].map(json.loads).tolist()
        self.output_dims = output_dims
        self.level_starts = level_starts
        self.total_outputs = sum(output_dims)
        self.residue_ids = load_residue_ids()
        self.records = records

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        tokens = torch.tensor(
            tokenize_sequence(self.sequences[index], self.residue_ids), dtype=torch.long
        )
        target = torch.zeros(self.total_outputs)
        mask = torch.zeros(self.total_outputs)
        for level in self.observed_levels[index]:
            start = self.level_starts[level]
            mask[start : start + self.output_dims[level]] = 1.0
        for level_indices in self.target_indices[index]:
            target[level_indices] = 1.0
        return tokens, target, mask, index


class HitecModule(pl.LightningModule):
    def __init__(self, output_dims, sdpa=False):
        super().__init__()
        self.save_hyperparameters()
        self.model = Transformer(
            CONFIG["vocab_size"],
            MODEL_DIMENSION,
            CONFIG["ah"],
            output_dims,
            CONFIG["dr"],
            CONFIG["beta"],
            sdpa,
        )
        self.criterion = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, tokens):
        return self.model(tokens, mode="infer")

    def training_step(self, batch, batch_idx):
        tokens, target, mask, _ = batch
        loss = (self.criterion(self(tokens), target) * mask).sum() / mask.sum()
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        base = CONFIG["lr"]
        groups = []
        for name, params in self.named_parameters():
            weight_decay = 0.0 if "norm" in name or "bias" in name else 0.01
            if name.startswith(("model.embeddings", "model.enc_1")):
                lr = base * 0.9**3
            elif name.startswith("model.enc_2"):
                lr = base * 0.9**2
            elif name.startswith("model.enc_3"):
                lr = base * 0.9
            elif name.startswith("model.enc_4"):
                lr = base
            else:
                lr = base * 1.05
            groups.append({"params": params, "weight_decay": weight_decay, "lr": lr})
        return torch.optim.AdamW(groups, lr=base, fused=True)


class SaveLast(Callback):
    """Writes the resumable state after every epoch; the file is replaced atomically."""

    def __init__(self, path):
        self.path = Path(path)

    def on_train_epoch_end(self, trainer, pl_module):
        temp = self.path.with_suffix(".tmp")
        trainer.save_checkpoint(temp)
        os.replace(temp, self.path)


class StopAtGuard(Callback):
    """Stops after an epoch when the next one, at 1.2 x the longest epoch so far, would pass stop_at.

    Must come after SaveLast in the callback list so the checkpoint exists before the stop.
    The first epoch also counts the time since `started`.
    """

    def __init__(self, stop_at, started):
        self.stop_at = stop_at
        self.mark = started
        self.longest = 0.0
        self.stopped = False

    def on_train_epoch_end(self, trainer, pl_module):
        now = time.time()
        self.longest = max(self.longest, now - self.mark)
        self.mark = now
        if trainer.current_epoch + 1 < trainer.max_epochs and now + 1.2 * self.longest > self.stop_at:
            self.stopped = trainer.should_stop = True


def main():
    started = time.time()
    args = parse_args()
    config = {
        **CONFIG,
        "epochs": args.epochs,
        "swa_epoch_start": args.swa_epoch_start,
        "swa_annealing_epochs": args.swa_annealing_epochs,
    }
    pl.seed_everything(args.seed, workers=True)
    metadata = load_run_metadata(args.split_group, args.out_root)
    seed_root = ensure_dir(seed_run_root(metadata, args.seed))

    dataset = ManifestDataset(
        metadata["manifests"]["train"], metadata["output_dims"], metadata["level_starts"]
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.accelerator == "gpu",
    )
    # Resumable state, rewritten every epoch.
    resume = seed_root / "checkpoint_last.pt"
    guard = StopAtGuard(args.stop_at, started) if args.stop_at is not None else None
    trainer = pl.Trainer(
        max_epochs=config["epochs"],
        accelerator=args.accelerator,
        devices=1,
        precision=("bf16-mixed" if args.amp else "16-mixed") if args.accelerator == "gpu" else 32,
        callbacks=[
            StochasticWeightAveraging(
                swa_lrs=CONFIG["swa_lrs"],
                swa_epoch_start=config["swa_epoch_start"],
                annealing_epochs=config["swa_annealing_epochs"],
            ),
            SaveLast(resume),
            *([guard] if guard else []),
        ],
        enable_checkpointing=False,
        enable_progress_bar=True,
        num_sanity_val_steps=0,
        default_root_dir=str(seed_root),
        logger=False,
    )
    module = HitecModule(metadata["output_dims"], args.sdpa)
    # In-place compile keeps the state_dict keys. Static shapes: the short last batch compiles once more.
    module.model.compile(dynamic=False, mode="max-autotune-no-cudagraphs")
    trainer.fit(
        module,
        train_dataloaders=loader,
        ckpt_path=str(resume) if resume.exists() else None,
    )

    if guard and guard.stopped:
        print("[emulator_bench] stopped before stop-at; resume later", flush=True)
        sys.exit(UNFINISHED)

    # SWA copies the averaged weights into the module at the end of fit.
    checkpoint = seed_root / "lastmodel.pth"
    trainer.save_checkpoint(checkpoint)
    write_json(
        seed_train_metadata_path(metadata, args.seed),
        {
            "split_group": args.split_group,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "sdpa": args.sdpa,
            "config": config,
            "train_rows": len(dataset),
            "checkpoint": str(checkpoint),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    )
    print("[emulator_bench] checkpoint: {}".format(checkpoint), flush=True)


if __name__ == "__main__":
    main()

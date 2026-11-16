from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from src.utils.rich_progress import progress
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from src.utils.rich_progress import progress

from .results import (
    choose_f1_thresholds,
    compact_evaluation_summary,
    compute_care_metrics,
    compute_hitec_metrics,
    compute_supplemental_classification_metrics,
    compute_supplemental_ranking_metrics,
    sigmoid,
    write_care_ranked_csv,
)
from .train import HitecModule, ManifestDataset
from .utils import (
    DEFAULT_OUT_ROOT,
    ensure_dir,
    load_run_metadata,
    read_json,
    seed_results_root,
    seed_train_metadata_path,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained HIT-EC checkpoint")
    parser.add_argument("--split-group", required=True)
    parser.add_argument("--out_root", "--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--rank-limit", type=int, default=50)
    return parser.parse_args()


def choose_device(requested):
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_checkpoint(metadata, seed):
    return Path(read_json(seed_train_metadata_path(metadata, seed))["checkpoint"])


def load_vocab(metadata):
    vocab_path = Path(metadata["vocab_path"])
    if not vocab_path.exists():
        raise FileNotFoundError("Missing vocab: {}".format(vocab_path))
    return read_json(vocab_path)


def predict_split(model, metadata, split_name, batch_size, num_workers, device, result_root):
    dataset = ManifestDataset(
        metadata["manifests"][split_name], metadata["output_dims"], metadata["level_starts"]
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    logits, targets, masks = [], [], []
    with torch.no_grad():
        for tokens, target, mask, _ in progress(
            loader, desc="{} predictions".format(split_name), unit="batch"
        ):
            logits.append(model(tokens.to(device)).cpu().numpy())
            targets.append(target.numpy())
            masks.append(mask.numpy())
    arrays = {
        "logits": np.concatenate(logits),
        "targets": np.concatenate(targets),
        "masks": np.concatenate(masks),
        "manifest": dataset.records,
    }
    np.savez_compressed(
        result_root / "{}_logits.npz".format(split_name),
        logits=arrays["logits"],
        targets=arrays["targets"],
        masks=arrays["masks"],
    )
    return arrays


def dropped_rows(stats):
    """Source rows removed before the manifest: missing fields, empty sequence, or no full EC."""
    return {
        "source_rows": stats["raw_rows"],
        "missing_required_fields": stats["missing_required_rows"],
        "empty_sequence": stats["rows_after_required_drop"] - stats["rows_before_exact_dedup"],
        "proteins_without_full_ec": stats["dropped_no_full_ec_rows"],
        "proteins_kept": stats["rows_after_label_filter"],
    }


def evaluate_arrays(
    arrays, thresholds, metadata, vocab, result_root, split_name, seed, rank_limit
):
    probs = sigmoid(arrays["logits"])
    predictions = (probs >= thresholds.reshape(1, -1)).astype(np.int32)
    care_csv = result_root / "{}_results_df.csv".format(split_name)
    care_df = write_care_ranked_csv(
        arrays["manifest"], probs, vocab, care_csv, rank_limit=rank_limit
    )
    metrics = {
        "split_group": metadata["split_group"],
        "run_slug": metadata["run_slug"],
        "eval_split": split_name,
        "seed": int(seed),
        "hitec": compute_hitec_metrics(
            arrays["logits"], arrays["targets"], arrays["masks"], thresholds, vocab
        ),
        "care_task1": compute_care_metrics(care_df),
        "supplemental": {
            "classification": compute_supplemental_classification_metrics(
                arrays["targets"], predictions, arrays["masks"]
            ),
            "ranking": compute_supplemental_ranking_metrics(care_df),
        },
        "artifacts": {
            "care_ranked_csv": str(care_csv),
            "logits_npz": str(result_root / "{}_logits.npz".format(split_name)),
        },
    }
    write_json(result_root / "{}_metrics.json".format(split_name), metrics)
    return metrics


def main():
    args = parse_args()
    metadata = load_run_metadata(args.split_group, args.out_root)
    vocab = read_json(metadata["vocab_path"])
    result_root = ensure_dir(seed_results_root(metadata, args.seed))
    checkpoint = resolve_checkpoint(metadata, args.seed)

    device = choose_device(args.device)
    model = (
        HitecModule.load_from_checkpoint(str(checkpoint), map_location=device).to(device).eval()
    )

    # Per-class thresholds maximise F1 on the validation split (paper, "Repeated stratified hold-out").
    arrays = {
        split: predict_split(
            model, metadata, split, args.batch_size, args.num_workers, device, result_root
        )
        for split in ("val", "test")
    }
    thresholds, stats = choose_f1_thresholds(
        sigmoid(arrays["val"]["logits"]), arrays["val"]["targets"], arrays["val"]["masks"]
    )
    np.save(result_root / "validation_thresholds.npy", thresholds)
    metrics = [
        evaluate_arrays(
            arrays[split],
            thresholds,
            metadata,
            vocab,
            result_root,
            split,
            args.seed,
            args.rank_limit,
        )
        for split in ("val", "test")
    ]
    summary = compact_evaluation_summary(
        metrics,
        checkpoint=str(checkpoint),
        thresholds={"source_split": "val", "stats": stats, "classes": int(len(thresholds))},
    )
    summary["dropped_rows"] = {
        split: dropped_rows(metadata["stats"][split]) for split in ("train", "val", "test")
    }
    write_json(result_root / "evaluation_summary.json", summary)
    print(
        "[emulator_bench] evaluation summary: {}".format(result_root / "evaluation_summary.json"),
        flush=True,
    )


if __name__ == "__main__":
    main()

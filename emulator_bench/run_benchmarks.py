"""Multi-seed HIT-EC evaluation on the default EMULaToR enzyme classification splits.

Reads the dataset tree only. Everything else goes under --out_root.
Finished (split, seed) runs are skipped, interrupted training resumes from its last checkpoint,
and a run with finished training but no evaluation summary only runs the evaluation.
With --gpus, each run gets one GPU through CUDA_VISIBLE_DEVICES and up to len(gpus) * runs_per_gpu run at once.
"""

from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

from .dataset_adapter import select_split_groups
from .utils import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_OUT_ROOT,
    DEFAULT_SEEDS,
    load_run_metadata,
    seed_results_root,
    seed_train_metadata_path,
)

UNFINISHED = 75  # train exit code: stopped early, resume later


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--out_root", "--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument(
        "--split-group", action="append", help="Default: every default split folder on disk"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--accelerator", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument(
        "--gpus", type=int, nargs="+", help="GPU ids; default: the visible GPUs, one run at a time"
    )
    parser.add_argument("--runs_per_gpu", type=int, default=1)
    parser.add_argument(
        "--amp", action="store_true", help="bf16 autocast instead of the original fp16"
    )
    parser.add_argument(
        "--sdpa",
        action="store_true",
        help="F.scaled_dot_product_attention in the three full attention layers",
    )
    parser.add_argument(
        "--stop-at",
        type=float,
        default=None,
        help="Unix time. Training stops at an epoch end before it and the script exits with 75",
    )
    parser.add_argument("--limit-per-split", type=int, default=None, help="Smoke tests only")
    parser.add_argument("--epochs", type=int, default=None, help="Smoke tests only")
    parser.add_argument("--swa-epoch-start", type=int, default=None, help="Smoke tests only")
    parser.add_argument("--swa-annealing-epochs", type=int, default=None, help="Smoke tests only")
    return parser.parse_args()


def run(module, *args, gpu=None):
    command = [sys.executable, "-m", "emulator_bench." + module, *map(str, args)]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)} if gpu is not None else None
    print("[emulator_bench] running (gpu {}): {}".format(gpu, " ".join(command)), flush=True)
    code = subprocess.run(command, env=env).returncode
    if code not in (0, UNFINISHED):
        raise subprocess.CalledProcessError(code, command)
    return code


def run_seed(args, group, seed, train_done, gpu):
    """Returns True when training stopped early and the run must resume later."""
    common = ["--split-group", group, "--out_root", args.out_root, "--seed", seed]
    smoke = [
        arg
        for flag in ("epochs", "swa_epoch_start", "swa_annealing_epochs")
        if getattr(args, flag) is not None
        for arg in ("--" + flag.replace("_", "-"), getattr(args, flag))
    ]
    if not train_done:
        code = run(
            "train",
            *common,
            "--batch-size",
            args.batch_size,
            "--num-workers",
            args.num_workers,
            "--accelerator",
            args.accelerator,
            *(["--amp"] if args.amp else []),
            *(["--sdpa"] if args.sdpa else []),
            *(["--stop-at", args.stop_at] if args.stop_at is not None else []),
            *smoke,
            gpu=gpu,
        )
        if code == UNFINISHED:
            return True
    run(
        "evaluate",
        *common,
        "--num-workers",
        args.num_workers,
        "--device",
        "cuda" if args.accelerator == "gpu" else "cpu",
        gpu=gpu,
    )


def main():
    args = parse_args()
    groups = [group.name for group in select_split_groups(args.dataset_root, args.split_group)]
    prepare = ["--dataset-root", args.dataset_root, "--out_root", args.out_root]
    jobs = []
    for group in groups:
        run(
            "prepare_splits",
            *prepare,
            "--split-group",
            group,
            *(["--limit-per-split", args.limit_per_split] if args.limit_per_split else []),
        )
        metadata = load_run_metadata(group, args.out_root)
        jobs += [
            (group, seed, seed_train_metadata_path(metadata, seed).exists())
            for seed in args.seeds
            if not (seed_results_root(metadata, seed) / "evaluation_summary.json").exists()
        ]

    slots = queue.Queue()
    for gpu in (args.gpus or [None]) * args.runs_per_gpu:
        slots.put(gpu)

    unfinished = threading.Event()

    def work(job):
        if unfinished.is_set():
            return
        gpu = slots.get()
        try:
            if run_seed(args, *job, gpu):
                unfinished.set()
        finally:
            slots.put(gpu)

    with ThreadPoolExecutor(max_workers=slots.qsize()) as pool:
        list(pool.map(work, jobs))
    if unfinished.is_set():
        sys.exit(UNFINISHED)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse

from .dataset_adapter import build_ec_vocab, prepare_split_group, select_split_groups
from .utils import DEFAULT_DATASET_ROOT, DEFAULT_OUT_ROOT, ensure_dir, vocab_path


def parse_args():
    parser = argparse.ArgumentParser(description="Build the EC vocabulary and per-split manifests")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--out_root", "--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--split-group", action="append", help="Split folder, relative to the dataset root")
    parser.add_argument("--rebuild-vocab", action="store_true")
    parser.add_argument("--limit-per-split", type=int, default=None, help="Smoke tests only")
    return parser.parse_args()


def main():
    args = parse_args()
    vocab_file = vocab_path(args.out_root)
    ensure_dir(vocab_file.parent)
    vocab = build_ec_vocab(args.dataset_root, vocab_file, rebuild=args.rebuild_vocab)
    for group in select_split_groups(args.dataset_root, args.split_group):
        print("[emulator_bench] preparing {}".format(group.name), flush=True)
        prepare_split_group(
            group,
            args.dataset_root,
            vocab,
            args.out_root,
            vocab_file,
            limit_per_split=args.limit_per_split,
        )


if __name__ == "__main__":
    main()

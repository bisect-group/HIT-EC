from __future__ import annotations

import argparse

import pandas as pd

from .utils import DEFAULT_OUT_ROOT, flatten_dict, read_json, task_root


def parse_args():
    parser = argparse.ArgumentParser(description="Mean and std of HIT-EC metrics over seeds")
    parser.add_argument("--out_root", "--out-root", default=str(DEFAULT_OUT_ROOT))
    return parser.parse_args()


def main():
    args = parse_args()
    root = task_root(args.out_root)
    rows = [
        {**flatten_dict(read_json(path)), "metrics_path": str(path)}
        for path in sorted(root.glob("*/*/seed_*/*_metrics.json"))
    ]
    if not rows:
        raise FileNotFoundError("No metrics JSON files found under {}".format(root))

    df = pd.DataFrame(rows)
    df.to_csv(root / "aggregate_metrics.csv", index=False)
    numeric = [
        col for col in df.columns
        if col != "seed" and pd.api.types.is_numeric_dtype(df[col])
    ]
    summary = df.groupby(["split_group", "eval_split"])[numeric].agg(["mean", "std"])
    summary.columns = [".".join(col) for col in summary.columns]
    summary.reset_index().to_csv(root / "aggregate_summary.csv", index=False)
    print("[emulator_bench] wrote {} and aggregate_summary.csv".format(root / "aggregate_metrics.csv"), flush=True)


if __name__ == "__main__":
    main()

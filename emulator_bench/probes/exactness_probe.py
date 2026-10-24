"""Is the loader change exact?

Trains K steps on a tiny synthetic set whose size forces a short last batch, and compares
num_workers 0 vs 4.

Run from baselines/HITEC on one GPU:
python -m emulator_bench.probes.exactness_probe --device cuda:0 --out exactness_probe.json
"""

from __future__ import annotations

import argparse
import torch
from torch.utils.data import DataLoader

from .common import build, make_rows, step, write_report


def train_steps(args, num_workers):
    module, optimizer, scaler = build(args.seed, False, args.device, args.compile_mode)
    module.train()
    rows = make_rows(args.batch_size * 3 + args.batch_size // 2, args.seed)
    loader = DataLoader(rows, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    torch.manual_seed(args.seed)
    losses = []
    while len(losses) < args.steps:
        for batch in loader:
            losses.append(step(module, optimizer, scaler, batch).item())
            if len(losses) == args.steps:
                break
    params = torch.cat([p.detach().float().flatten().cpu() for p in module.parameters()])
    return torch.tensor(losses), params


def compare(a, b):
    loss_diff = (a[0] - b[0]).abs().max().item()
    param_diff = (a[1] - b[1]).abs().max().item()
    return {
        "max_abs_loss_diff": loss_diff,
        "max_abs_param_diff": param_diff,
        "bit_identical": loss_diff == 0 and param_diff == 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compile-mode", default="max-autotune-no-cudagraphs")
    args = parser.parse_args()

    torch.cuda.set_device(args.device)
    report = {"workers_0_vs_4": compare(train_steps(args, 0), train_steps(args, 4))}
    write_report(args.out, report)


if __name__ == "__main__":
    main()

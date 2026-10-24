"""Does inductor fuse the attention, and which attention variant is fastest?

Run from baselines/HITEC on one GPU:
python -m emulator_bench.probes.attention_probe --device cuda:0 --out attention_probe.json
"""

from __future__ import annotations

import argparse

import torch
from torch._dynamo.utils import counters

from .common import build, make_batches, timed_steps, write_report

BACKENDS = ("flash", "mem_efficient", "cudnn", "math")


def set_backends(allowed):
    """Allow only the named SDPA backends; None leaves the defaults."""
    if allowed is not None:
        torch.backends.cuda.enable_flash_sdp("flash" in allowed)
        torch.backends.cuda.enable_mem_efficient_sdp("mem_efficient" in allowed)
        torch.backends.cuda.enable_cudnn_sdp("cudnn" in allowed)
        torch.backends.cuda.enable_math_sdp("math" in allowed)


def run_variant(args, sdpa, allowed, batches):
    torch._dynamo.reset()
    counters.clear()
    set_backends(allowed)
    module, optimizer, scaler = build(args.seed, sdpa, args.device, args.compile_mode)
    module.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        first_loss = module.training_step(tuple(t.to(args.device) for t in batches[0]), 0).item()
    module.train()
    result = {
        "first_loss_eval": first_loss,
        "steps_per_s": timed_steps(module, optimizer, scaler, batches, args.warmup),
        "fuse_attention": counters["inductor"]["fuse_attention"],
    }
    set_backends(BACKENDS)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=50, help="Timed steps after the warm-up")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--loss-tol", type=float, default=1e-3)
    parser.add_argument("--compile-mode", default="max-autotune-no-cudagraphs")
    args = parser.parse_args()

    torch.cuda.set_device(args.device)
    batches = make_batches(args.warmup + args.steps, args.batch_size, args.seed)
    variants = {
        "default": (False, None),
        "default_no_mem_efficient": (False, ("flash", "cudnn", "math")),
    }
    for backend in BACKENDS:
        variants[f"sdpa_{backend}_only"] = (True, (backend,))
    variants["sdpa_all_backends"] = (True, BACKENDS)

    report = {"device": torch.cuda.get_device_name(args.device), "variants": {}}
    for name, (sdpa, allowed) in variants.items():
        try:
            report["variants"][name] = run_variant(args, sdpa, allowed, batches)
        except Exception as error:
            report["variants"][name] = {"error": f"{type(error).__name__}: {str(error)[:300]}"}
            set_backends(BACKENDS)
    base = report["variants"]["default"]
    for result in report["variants"].values():
        if "error" not in result:
            result["loss_diff_vs_default"] = abs(
                result["first_loss_eval"] - base["first_loss_eval"]
            )
            result["loss_matches_default"] = result["loss_diff_vs_default"] <= args.loss_tol
            result["speedup_vs_default"] = result["steps_per_s"] / base["steps_per_s"]
    write_report(args.out, report)


if __name__ == "__main__":
    main()

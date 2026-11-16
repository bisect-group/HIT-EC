"""Shared setup for the GPU probes: the real model shape, synthetic batches, fp16 training steps."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from torch.utils.data import default_collate

from emulator_bench.train import CONFIG, HitecModule
from emulator_bench.utils import MODEL_DIMENSION

OUTPUT_DIMS = [8, 64, 512, 2048]


def make_rows(count, seed=0):
    gen = torch.Generator().manual_seed(seed)
    total = sum(OUTPUT_DIMS)
    rows = []
    for index in range(count):
        mask = torch.zeros(total)
        mask[: total // 2] = 1
        target = (torch.rand(total, generator=gen) > 0.99).float()
        tokens = torch.randint(1, CONFIG["vocab_size"], (MODEL_DIMENSION,), generator=gen)
        rows.append((tokens, target, mask, index))
    return rows


def build(seed, sdpa=False, device="cuda:0", compile_mode="max-autotune-no-cudagraphs"):
    """Module, fused AdamW and GradScaler as in train.py; the module is compiled like train.py."""
    torch.manual_seed(seed)
    module = HitecModule(OUTPUT_DIMS, sdpa).to(device)
    module.log = lambda *args, **kwargs: None
    if compile_mode:
        module.model.compile(dynamic=False, mode=compile_mode)
    optimizer = module.configure_optimizers()
    return module, optimizer, torch.amp.GradScaler(device.split(":")[0])


def step(module, optimizer, scaler, batch):
    tokens, target, mask, _ = (t.to(next(module.parameters()).device) for t in batch)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16):
        loss = module.training_step((tokens, target, mask, None), 0)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return loss.detach()


def make_batches(count, batch_size, seed=0):
    rows = make_rows(count * batch_size, seed)
    return [default_collate(rows[i * batch_size : (i + 1) * batch_size]) for i in range(count)]


def timed_steps(module, optimizer, scaler, batches, warmup):
    """Steps per second over the batches after `warmup` untimed steps."""
    for batch in batches[:warmup]:
        step(module, optimizer, scaler, batch)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for batch in batches[warmup:]:
        step(module, optimizer, scaler, batch)
    torch.cuda.synchronize()
    return (len(batches) - warmup) / (time.perf_counter() - start)


def write_report(path, report):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

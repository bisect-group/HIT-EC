#!/usr/bin/env bash
# HIT-EC paper runs: 4 split groups, batch 64, 80 epochs, original fp16 (no --amp, no --sdpa). train also evaluates.
# Usage: emulator_bench/run.sh [--dry-run] [prepare|train|aggregate|all]
# Env: PY GPUS SEEDS RUNS_PER_GPU DATA_ROOT OUT_ROOT EMULATOR_ROOT DRY_RUN=1
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
EMULATOR_ROOT=${EMULATOR_ROOT:-$(realpath ../..)}
DATA_ROOT=${DATA_ROOT:-$EMULATOR_ROOT/data/processed/datasets/enzyme_classification_dataset}
OUT_ROOT=${OUT_ROOT:-$EMULATOR_ROOT/outputs/baselines/HITEC}
PY=${PY:-python}
GPUS=${GPUS:-0}
RUNS_PER_GPU=${RUNS_PER_GPU:-1}
SEEDS=${SEEDS:-"666 777 888"}
read -ra gpus <<<"$GPUS"
read -ra seeds <<<"$SEEDS"
STEP=all
for a in "$@"; do
  case $a in
    --dry-run) DRY_RUN=1 ;;
    prepare|train|aggregate|all) STEP=$a ;;
    *) echo "usage: $0 [--dry-run] [prepare|train|aggregate|all]" >&2; exit 2 ;;
  esac
done
run() { echo "+ $*"; [[ -n ${DRY_RUN:-} ]] || "$@"; }
want() { [[ $STEP == all || $STEP == "$1" ]]; }

want prepare && run $PY -m emulator_bench.prepare_splits --dataset-root "$DATA_ROOT" --out_root "$OUT_ROOT"
want train && run $PY -m emulator_bench.run_benchmarks --dataset-root "$DATA_ROOT" --out_root "$OUT_ROOT" \
  --seeds "${seeds[@]}" --gpus "${gpus[@]}" --runs_per_gpu "$RUNS_PER_GPU"
want aggregate && run $PY -m emulator_bench.aggregate_results --out_root "$OUT_ROOT"
exit 0

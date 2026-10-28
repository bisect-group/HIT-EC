# HIT-EC EMULaToR wrapper

Multi-seed evaluation of HIT-EC with its original settings on the default EMULaToR EC classification splits.
The wrapper reads `data/processed/datasets/enzyme_classification_dataset` and never writes there.
All outputs go to `--out_root` (default `<EMULaToR>/outputs/baselines/HITEC`).

Split folders come from disk: `random_splits_grouped_sequence`, `uniprot_time_splits`, and every
`enzyme_sequence_splits/threshold_*` and `enzyme_structure_splits/threshold_*`. `backup_*` folders are never read.
Each run goes to `<out_root>/enzyme_classification/<split_group>/<split_name>/seed_<s>/`.
A flat split group uses `all` as split name. Summaries go to `<out_root>/enzyme_classification/`.

Training uses the settings of the released code: AdamW (lr 8.75e-5, layer-wise lr x0.9, weight decay 0.01),
80 epochs, fp16 mixed precision, SWA from epoch 49 (30 annealing epochs, lr 3e-7), attention heads 2,
dropout 0.1, beta 0.59, dimension 1024. Sequences are cut to 1023 residues plus a BOS token.
Only fully specified EC numbers (four levels, no "-") are used in train, val and test. A partial EC is dropped
from a protein's labels, and a protein left without a full EC is dropped and counted per split in `metadata.json` and
`evaluation_summary.json` (`dropped_rows`). The label vocabulary holds only full ECs and their prefixes.
Class thresholds maximise F1 on the validation split. The test split reuses them.

Default seeds are 666, 777 and 888. Run from this folder (`baselines/HITEC`) with the mldb Python:

```bash
python -m emulator_bench.run_benchmarks --gpus 0 1 2 --runs_per_gpu 1
python -m emulator_bench.run_benchmarks --gpus 0 --split-group uniprot_time_splits
python -m emulator_bench.aggregate_results
```

`run_benchmarks` skips finished (split, seed) runs. Training resumes from `checkpoint_last.pt` (written every epoch) after an interruption.
`--stop-at <unix time>` (on `train` and `run_benchmarks`) ends training after an epoch when the next one, at 1.2 x the longest epoch so far, may pass that time. The checkpoint is written first. Nothing else is written: no `lastmodel.pth`, no evaluation. Both scripts then exit with code 75. Rerun the same command in the next job to resume.
A run with finished training but no `evaluation_summary.json` only runs the evaluation. `prepare_splits` skips split folders whose inputs did not change.
`lastmodel.pth` is the final (SWA) model. `--amp` trains in bf16 instead of the original fp16. `--sdpa` (off by default) uses `F.scaled_dot_product_attention` in the full attention layers. `--num-workers` defaults to 4. Single steps:
`emulator_bench.prepare_splits`, `emulator_bench.train`, `emulator_bench.evaluate`.
Do not start two `prepare_splits` runs for the same split folder at once.

`--batch-size` (default 64) is not given by the paper. The original default of 2 is too slow. `--limit-per-split`, `--epochs`, `--swa-epoch-start` and `--swa-annealing-epochs` are for smoke tests only.
See `REPORT.md`, `commands.txt` and `default_hparams.json`.

## Verify equivalence

```bash
conda activate mldb
emulator_bench/verify_equivalence.sh        # CPU, about 30 s
emulator_bench/verify_equivalence.sh --gpu  # also the fp16 tolerance test
emulator_bench/verify_equivalence.sh --negative-control  # the GPU test must FAIL
```

The CPU mode compares the wrapper with the released `model/model.py` and `training/infer_training.py`.
The tests read the released code with `git show`, so they need a clone with the full git history.
They use a tiny model (dimension 64) in fp64 and check the forward pass, gradients, optimizer groups,
weights after Adam steps, 6-epoch SWA training and stop-and-resume training. The tolerance is 1e-10
(1e-5 for the compiled fp32 path). A second test checks that the loader gives the same batches for
`--num-workers` 0 and 4 on a synthetic manifest, with the same `DataLoader` arguments as `train.py`.

Not exact: the wrapper loss omits the constant log(2) the released loss adds for each unobserved entry.
This does not change gradients. One test marks the difference as an expected failure and another checks it.
The `--gpu` test trains 20 steps with the production path (compile, fp16 autocast, GradScaler, fused AdamW) and with eager fp32 and plain AdamW (the released optimizer; production uses fused AdamW), with dropout off (compiled dropout draws another random stream). The noise control is the production run with initial weights perturbed by one rounding of the production dtype (2**-11, fp16). Per step, the gap |production - fp32| must be at most 10 times the floor |perturbed production - production|. The same holds for the mean final prediction gap. Step 0 is judged like every other step: its gap must be at most 10 times its floor. Each test prints `max gap`, `max floor` and `ratio`; run with `-s` to see the line.
`--negative-control` reruns that test with the production run's initial weights multiplied by 1.05 (`EQUIV_NEGATIVE_CONTROL=1`). The test must fail, and the script exits 0 only then.
The released code also uses fp16, so the production run has the same precision as the original.

# HIT-EC wrapper report

## 1. Method summary

HIT-EC predicts EC numbers from the protein sequence. A 4-block transformer encodes up to 1023 residues
plus a BOS token. Each block feeds a local head for one EC level. A global head covers all levels.
The final logits are `0.59 * global + 0.41 * local`. Training uses a masked binary cross-entropy:
a level that a protein does not observe adds no loss. Per-class thresholds maximise F1 on the validation split.
The wrapper trains and evaluates the model on the EMULaToR EC classification splits with the original settings.

## 2. Original default settings

See `default_hparams.json` for every value and its source. In short, from `training/infer_training.py`:
AdamW, lr 8.75e-5, layer-wise lr x0.9 per block (heads x1.05), weight decay 0.01 (0 for norm and bias),
80 epochs, fp16, SWA (lr 3e-7, from epoch 49, 30 annealing epochs), 2 heads, dropout 0.1, beta 0.59.
Batch size and seeds are not given. The wrapper uses batch size 64 and seeds 666, 777, 888.

## 3. What the wrapper does

- `prepare_splits.py` builds the EC vocabulary and one manifest per split. It skips a split folder when
  the split files, vocabulary, limit and `dataset_adapter.py` match the stored fingerprint.
- `train.py` trains with the original settings. It writes `checkpoint_last.pt` after every epoch (atomic replace)
  and resumes from it. `lastmodel.pth` holds the final SWA weights.
- `evaluate.py` predicts val and test, picks thresholds on val, and writes metrics and `evaluation_summary.json`.
  The summary has a `dropped_rows` entry per split (see section 10).
- `run_benchmarks.py` runs everything per (split, seed) and per GPU. A run with `train.json` but no
  `evaluation_summary.json` runs the evaluation only.
- **Label rule.** Only fully specified ECs (four levels, no "-" or "n" part) are used in train, val and test.
  A partial EC is dropped from the protein's label set. A protein left without a full EC is dropped and counted per split
  (section 10). The vocabulary holds only full ECs and their prefixes (levels of 7, 76, 270 and 6182 classes). A cached
  vocabulary from another label policy is rebuilt, and the prepare fingerprint covers the label code, so old manifests rebuild.
- Runs go to `<out_root>/enzyme_classification/<split_group>/<split_name>/seed_<s>/`.

## 4. Files

Added: `prepare_splits.py`, `run_benchmarks.py`, `REPORT.md`, `commands.txt`, `default_hparams.json`, `tests/`.
Removed: `cache_features.py`, `queue_pipeline.py`, `queue_multiple_seeds.sh`.
Changed: `train.py`, `evaluate.py`, `dataset_adapter.py`, `results.py`, `utils.py`, `aggregate_results.py`,
`README.md`, and `model/model.py` (speed-ups only, section 6).

## 5. Bugs fixed

- The first wrapper never wrote `checkpoint_last.pt` during training. Lightning 2.6.1 skips the "last" file when
  `save_top_k=0`. A small `SaveLast` callback now writes it every epoch. The smoke test below proves the resume.
- `enable_checkpointing=False` stops Lightning from adding its own per-epoch checkpoint.

## 6. Deviations, and why behaviour is unchanged

Speed-ups (all exact up to fp noise):

1. `model/model.py`, attention. The original views the `(b, h, L, dim)` attention output as `(L, b, dim, h)`
   and sums over h. That sums h adjacent value columns. The wrapper sums the value weights and bias first,
   so the value projection and the P*V product are half as wide. The output is the same.
2. `model/model.py`, layer 4. Only row 0 of the last block reaches the heads, and it depends on the
   first h query rows of head 0. The wrapper computes only that part.
3. `torch.compile` (`max-autotune-no-cudagraphs`, static shapes) and fused AdamW. State-dict keys are unchanged.
4. Four data-loader workers (`--num-workers`, default 4). Sample order and `__getitem__` hold no randomness
   beyond the seeded shuffle, so the batches do not change.
5. `--sdpa` (default off, opt-in) calls `F.scaled_dot_product_attention` in the three full attention layers.
   It uses the default scale 1/sqrt(512), no mask and no dropout, so it is the same function.
   Use `emulator_bench/probes/attention_probe.py` to check speed before use. Results change only by fp noise.

Measured exactness:

- fp32, CPU, dimension 64 and 1024 with padded rows: max logit difference 1.9e-6 (scale about 9),
  max gradient difference 3.7e-9. `tests/test_emulator_bench.py::test_model_matches_released_forward` checks this.
- fp16, GPU, 192 steps with SWA, dropout 0, against the original eager code: loss differs by at most 4.1e-6,
  parameters by at most 1.9e-4. Original eager against itself differs by 1.4e-6 and 4.4e-5.
- A reviewer reran the comparison on several shapes and confirmed the model changes are exact.

Documented differences that do not change what the model learns:

- **Dropout RNG order differs** from the released model. The compiled model draws masks in another order and the
  layer-4 shortcut draws fewer masks. The distribution is the same. With dropout 0.1 the mean loss over a short
  run was 0.034564 (original) and 0.034569 (compiled).
- **Logged train loss differs by a constant.** The original replaces unobserved predictions and labels by 0, so each
  unobserved entry adds BCE(0, 0) = ln 2 to the sum. The wrapper masks those entries out. The gradients are the
  same because the replaced entries carry no gradient.
- **No scheduler, like the original in effect.** The released code returns the scheduler under the key
  `scheduler`. Lightning 2.6.1 only reads `lr_scheduler`. A check with a `MultiplicativeLR(0.95)` returned that way
  gave an empty scheduler list and an unchanged lr after 3 epochs. So the original never decays the lr either.
  SWA sets its own lr from the SWA start epoch.
- **No bestmodel.** The original selects no checkpoint on validation. The wrapper uses the final SWA weights.
- **Classes absent from val get threshold 1.0.** A class with no observed or no positive validation rows
  is never predicted positive. The count is in `thresholds.stats` of `evaluation_summary.json`.
- Batch size 64 and fp16 `GradScaler` (Lightning "16-mixed") replace the original 4-GPU DDP setup.

## 7. Commands

See `commands.txt`. All commands run from `baselines/HITEC`.

## 8. Smoke test

GPU 3, batch size 4, 64 rows per split, `enzyme_structure_splits/threshold_0.85`, seed 666, 80 epochs with SWA
start 2 and 2 annealing epochs, compile + fused AdamW + fp16.

1. `python -m emulator_bench.prepare_splits --out_root $OUT --split-group enzyme_structure_splits/threshold_0.85 --limit-per-split 64`
2. `python -m emulator_bench.train ... --batch-size 4 --epochs 80 --swa-epoch-start 2 --swa-annealing-epochs 2`,
   killed by its PID after the sixth `checkpoint_last.pt` write (epoch 6, past the SWA start).
3. `python -m emulator_bench.run_benchmarks --out_root $OUT --split-group ... --seeds 666 --batch-size 4 --gpus 3 --limit-per-split 64 --epochs 80 --swa-epoch-start 2 --swa-annealing-epochs 2`.
   Output: prepare skipped (manifest mtime unchanged), `Restoring states from the checkpoint path ...`,
   `Restored all states from the checkpoint`, training ran to epoch 80, `lastmodel.pth` written, evaluation ran.
4. Moving `evaluation_summary.json` away and rerunning `run_benchmarks` ran only `evaluate` and wrote the same summary.

Result: val micro F1 0.137, CARE k=1 level-4 accuracy 12.5, k=20 31.25. These numbers are meaningless
(64 rows, random-like training). The check shows that the pipeline runs and resumes.

## 9. Equivalence verdict

EQUIVALENT-EXCEPT-DOCUMENTED. The model, optimizer groups, loss gradients and dataset construction match the
released code (unit tests and the numbers in section 6). I did not run the released script end to end. It ships no
data module, and its `training_step` uses an undefined `observed` mask.

## 10. Open issues

- **Dropped rows.** `evaluation_summary.json` counts per split (`dropped_rows`, train, val and test): source rows, rows with
  missing fields, rows with an empty sequence, proteins without a full EC, and proteins kept. `metadata.json` holds the
  same counts in `stats`. The wrapper drops proteins without a full EC by the user's rule (full ECs only). Counts from a
  CPU `prepare_splits` run on all four split groups (proteins after merging rows of the same sequence, dropped for no full EC):

  | Split group | train | val | test |
  |---|---|---|---|
  | random_splits_grouped_sequence | 13,698 of 236,617 | 767 of 13,217 | 738 of 13,164 |
  | uniprot_time_splits | 13,443 of 237,451 | 780 of 12,211 | 980 of 13,336 |
  | enzyme_sequence_splits/threshold_0.39 | 13,499 of 238,649 | 782 of 12,901 | 922 of 11,448 |
  | enzyme_structure_splits/threshold_0.85 | 13,252 of 239,151 | 777 of 11,853 | 1,174 of 11,994 |

  No row had missing fields or an empty sequence in any split. The counts are proteins, not source rows. A label such
  as `3.4.21.n1` has an "n" part and counts as not full.
- Source rows of the same protein and sequence merge into one row with the union of their labels. Sequences
  above 1023 residues are cut. Both change the row count, so "dropped" and "kept" are not a plain subtraction.
- Speed: about 2.76 steps/s on an A10, so about 30 h per run on the 0.85 structure split (about 226k train proteins
  after the full-EC rule, 80 epochs, batch 64; the 30 h was measured with 239k rows, so expect about 28 h). Other splits scale with their train size. 12 runs need about 340 h on one A10.
- Batch size 64 and the seeds are wrapper choices.
- Padding to 1024 tokens is part of the model function (no attention mask), so it cannot be trimmed.

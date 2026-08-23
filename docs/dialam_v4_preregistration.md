# DialAM v4 class-balanced rehearsal preregistration

Status: **FROZEN BEFORE TRAINING**.

V3 materially fixed false-positive calibration but left a class-coverage
bottleneck: only 189/4,096 training rows were ATTACK positives. ATTACK recall
was 0% on development and 12.5% on the reused frozen benchmark. V4 tests one
data-only correction while retaining the same Qwen3-0.6B base, prompt,
per-example loss, QLoRA hyperparameters, seed, three epochs, and greedy decoding.

## Intervention

Build 8,192 rows with exactly 4,096 positive and 4,096 NONE targets:

- retain all 4,096 rows from the v3 same-update paired foundation;
- balance the final positive side to 1,344 SUPPORT, 1,344 ATTACK, 1,344
  REPHRASE, and 64 mixed-label rows;
- use every unused ATTACK example before deterministic ATTACK-only repetition;
- add 2,048 unique NONE blocks, exhausting unused same-update hard negatives
  before adding lexically difficult all-negative blocks; and
- keep every development/frozen parent episode excluded from training.

No semantic candidate retrieval, model-generated label, new source dataset, or
hyperparameter search is introduced.

## Development gate

The existing four-episode, 30-case v3 development set is the selection gate.
V4 reaches the reused frozen benchmark only if all of these hold:

- exact patch accuracy at least `0.26666666666666666`;
- edge F1 at least `0.25`;
- ATTACK F1 at least `0.10`;
- false edges/update at most `0.3333333333333333`;
- at most two of six NONE cases have a false edge; and
- JSON and schema validity remain 100%.

## Reused frozen-benchmark promotion rule

V3 frozen errors informed this intervention, so the 30-case frozen set is now a
reused comparison benchmark, not an untouched model-selection holdout. The
external staff-held-out set remains the final unbiased evaluation. V4 replaces
v3 only if every reused-benchmark condition holds:

- exact patch accuracy at least `0.43333333333333335`;
- edge F1 at least `0.40`;
- relation macro-F1 at least `0.39`;
- ATTACK F1 at least `0.30`;
- false edges/update at most `0.30`;
- at most one of six NONE cases has a false edge; and
- JSON and schema validity remain 100%.

All results will be reported even if v4 fails. V3 remains selected unless the
complete promotion rule passes.

## Reproduction

```bash
.venv/bin/python scripts/build_dialam_training_v4.py
modal run scripts/modal_dialam_qlora.py --action train --size 8192 --dataset-version v4
modal run scripts/modal_dialam_qlora.py --action evaluate --target tuned --size 8192 --dataset-version v4 --eval-split v3_dev --output-path results/dialam_model_generation/v4_n8192_dev/predictions.jsonl
.venv/bin/python scripts/score_dialam_predictions.py results/dialam_model_generation/v4_n8192_dev/predictions.jsonl --examples data/dialam/training/v3_dev_eval_examples.jsonl
```

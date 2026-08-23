# DialAM v4 class-balanced rehearsal result

Decision: **RETAIN_V3_V4_FAILED_DEVELOPMENT_GATE**.

V4 improved semantic edge recovery on the episode-disjoint development set,
especially ATTACK, but failed its preregistered NONE-calibration condition. The
reused frozen benchmark and judge were therefore **not run**. V3 remains the
selected public model.

| Development metric | V3 / n=4096 | V4 / n=8192 | Delta |
|---|---:|---:|---:|
| Exact patch accuracy | 26.7% | **30.0%** | 3.3% |
| Edge precision | 28.6% | **41.2%** | 12.6% |
| Edge recall | 16.7% | **29.2%** | 12.5% |
| Edge F1 | 21.1% | **34.1%** | 13.1% |
| Relation macro-F1 | 17.3% | **31.5%** | 14.2% |
| ATTACK F1 | 0.0% | **54.5%** | 54.5% |
| REPHRASE F1 | 16.7% | 0.0% | -16.7% |
| SUPPORT F1 | 35.3% | **40.0%** | 4.7% |
| False edges/update | 0.333 | 0.333 | +0.000 |
| NONE cases with false edges | 2/6 | **4/6** | +2 |
| JSON/schema validity | 100.0% / 100.0% | 100.0% / 100.0% | unchanged |

## Gate decision

V4 passed exact accuracy, edge F1, ATTACK F1, false edges/update, JSON validity,
and schema validity. It failed the condition allowing at most two of six NONE
cases to contain a false edge: v4 produced false edges on 4/6, all SUPPORT.
Following the preregistration, no reused-frozen candidate calls or judge calls
were made.

## What this learned

The intervention achieved its intended rare-class effect: ATTACK F1 rose from
0.0% to 54.5%, edge recall rose from 16.7% to 29.2%, and edge F1 rose from
21.1% to 34.1%. But it weakened calibrated sparsity and label balance:
REPHRASE F1 fell from 16.7% to 0.0%, and NONE false-positive cases doubled.

This rules out "just add more class-balanced rows" as the final fix. The
joint, variable-length patch objective still trades recall against NONE
precision. If work continues, the strongest next experiment is a separately
preregistered pairwise `SUPPORT|ATTACK|REPHRASE|NONE` classifier followed by
deterministic patch assembly. That is a formulation change, not a v4 retune.

## Reproduce

```bash
.venv/bin/python scripts/build_dialam_training_v4.py
modal run scripts/modal_dialam_qlora.py --action train --size 8192 --dataset-version v4
modal run scripts/modal_dialam_qlora.py --action evaluate --target tuned --size 8192 --dataset-version v4 --eval-split v3_dev --output-path results/dialam_model_generation/v4_n8192_dev/predictions.jsonl
.venv/bin/python scripts/score_dialam_predictions.py results/dialam_model_generation/v4_n8192_dev/predictions.jsonl --examples data/dialam/training/v3_dev_eval_examples.jsonl
.venv/bin/python scripts/build_dialam_v4_report.py
```

The v4 checkpoint is preserved locally and in the authenticated Modal volume.
QT30-derived training/evaluation text and predictions remain ignored. The JSON
report publishes only aggregate metrics, configuration, paths, and hashes.

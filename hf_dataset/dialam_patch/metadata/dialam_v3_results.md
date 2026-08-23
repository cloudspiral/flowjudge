# DialAM v3 paired-loss result

Decision: **PROMOTE_V3_AS_FINAL_DIRECTION**.

V3 clears every preregistered material-improvement condition and becomes the selected model direction. It does **not** clear the original frozen reliability bar, so it remains a research/assignment artifact rather than a reliable argument-mining system.

| Metric | Base | v1 / n=2048 | v2 / n=2048 | v3 / n=4096 |
|---|---:|---:|---:|---:|
| Exact patch accuracy | 0.0% | 26.7% | 23.3% | **43.3%** |
| Edge precision | 0.0% | 16.7% | 18.8% | **43.8%** |
| Edge recall | 0.0% | 16.7% | 25.0% | **29.2%** |
| Edge F1 | 0.0% | 16.7% | 21.4% | **35.0%** |
| Relation macro-F1 | 0.0% | 16.7% | 18.8% | **33.9%** |
| False edges/update | 0.000 | 0.667 | 0.867 | **0.300** |
| NONE cases with false edges | n/a | 2/6 | 5/6 | **0/6** |
| Judge Robustness /4 | 1.40 | 2.10 | 1.43 | **2.97** |

## What changed

- 4,096 rows arranged as 2,048 same-update positive/NONE pairs.
- One pair per update, including all 189 paired ATTACK examples available after the development split.
- Prompt-masked loss averaged over each example before averaging the batch, eliminating the 4.50x output-length weighting imbalance.
- Four original parent episodes reserved for a new 30-case development set; the frozen six episodes and judge rubric stayed unchanged.

The separate development set reached 8/30 exact, 21.1% edge F1, 0.333 false edges/update, and 2/6 NONE false-positive cases. The single frozen pass improved further to 13/30 exact, 35.0% edge F1, 0.300 false edges/update, and 0/6 NONE false-positive cases.

## Before to after

Versus selected v1, v3 gains +0.183 edge F1, +0.167 exact accuracy, and +0.867 judge Robustness while reducing false edges/update by 0.367. Versus v2, it gains +0.136 edge F1 and reduces false edges/update by 0.567.

The original false-positive SUPPORT problem is substantially corrected. The remaining failure is conservative underprediction plus label/target confusion: ATTACK is still weakest at 18.2% F1. The model fails the original 85% edge-F1, 80% exact-patch, 75% macro-F1, 0.2 false-edge, and 3.5 Robustness reliability thresholds.

## Reproduce

```bash
.venv/bin/python scripts/build_dialam_training_v3.py
modal run scripts/modal_dialam_qlora.py --action train --size 4096 --dataset-version v3
modal run scripts/modal_dialam_qlora.py --action evaluate --target tuned --size 4096 --dataset-version v3 --eval-split v3_dev --output-path results/dialam_model_generation/v3_n4096_dev/predictions.jsonl
.venv/bin/python scripts/score_dialam_predictions.py results/dialam_model_generation/v3_n4096_dev/predictions.jsonl --examples data/dialam/training/v3_dev_eval_examples.jsonl
modal run scripts/modal_dialam_qlora.py --action evaluate --target tuned --size 4096 --dataset-version v3 --eval-split frozen --output-path results/dialam_model_generation/v3_n4096/predictions.jsonl
.venv/bin/python scripts/evaluate_dialam_model.py --predictions results/dialam_model_generation/v3_n4096/predictions.jsonl --output-dir results/dialam_model_eval/v3_n4096 --approval APPROVE_DIALAM_MODEL_JUDGING
.venv/bin/python scripts/build_dialam_v3_report.py
```

QT30-derived rows, predictions, records, and judge transcripts remain local and ignored. This report publishes only aggregate metrics, paths, hashes, configuration, and checkpoint file hashes.

# DialAM v5.1 calibrated pairwise result

Final selection decision: **PROMOTE_V5_1_AS_FINAL_DIRECTION**.

V5 converted the task from free JSON generation to one fixed four-label
decision per supplied candidate. V5.1 applies the preregistered 3.0 NONE
margin to correct the known 50.0% training versus 7.379% natural positive
prior shift; it does not alter the checkpoint or training data.

## Episode-disjoint development gate

| Metric | Raw v5 / 8192 | Calibrated v5.1 / 8192 |
|---|---:|---:|
| Exact patch accuracy | 40.0% | **53.3%** |
| Edge F1 | 48.3% | **53.3%** |
| Relation macro-F1 | 49.5% | **54.3%** |
| False edges/update | 0.667 | **0.300** |
| NONE cases with false edges | 6/6 | **2/6** |

V5.1 passed all seven unchanged development checks before the frozen set
was evaluated. The selected margin and complete 0.00–3.00 grid are
preserved in `reports/dialam_v5_1_calibration.json`.

## Reused frozen benchmark

| Metric | Previous v3 / 4096 | Selected v5.1 / 8192 | Delta |
|---|---:|---:|---:|
| Exact patch accuracy | 43.3% | **53.3%** | +10.0 pp |
| Edge precision | 43.8% | **55.6%** | +11.8 pp |
| Edge recall | 29.2% | **41.7%** | +12.5 pp |
| Edge F1 | 35.0% | **47.6%** | +12.6 pp |
| Relation macro-F1 | 33.9% | **47.4%** | +13.5 pp |
| ATTACK F1 | 18.2% | **30.8%** | +12.6 pp |
| False edges/update | 0.300 | **0.267** | -0.033 |
| NONE cases with false edges | 0/6 | **0/6** | 0 |
| Judge Robustness /4 | 2.967 | **3.033** | +0.067 |

All nine preregistered promotion checks passed:

- PASS: `exact_patch_accuracy`
- PASS: `edge_f1`
- PASS: `relation_macro_f1`
- PASS: `attack_f1`
- PASS: `false_edges_per_update`
- PASS: `none_scenarios_with_false_edges`
- PASS: `judge_robustness`
- PASS: `json_validity`
- PASS: `schema_validity`

The model still does not clear the original production-like reliability
bar (80% exact, 85% edge F1, 75% macro-F1, at most 0.20 false
edges/update, and 3.5/4 Robustness). It is the strongest tested
assignment artifact, not a production-ready argument miner.

The chronological ledger and generated chart are in
[`docs/dialam_experiment_history.md`](dialam_experiment_history.md).

## Reproduce

```bash
.venv/bin/python scripts/calibrate_dialam_v5.py
modal run scripts/modal_dialam_qlora.py --action evaluate --target tuned --size 8192 --dataset-version v5 --eval-split frozen --output-path results/dialam_model_generation/v5_n8192_frozen_raw_scores/predictions.jsonl
.venv/bin/python scripts/apply_dialam_v5_margin.py --predictions results/dialam_model_generation/v5_n8192_frozen_raw_scores/predictions.jsonl --examples data/dialam/balanced_diagnostic_eval_examples.jsonl --margin 3.0 --output results/dialam_model_generation/v5_1_n8192/predictions.jsonl --summary reports/dialam_v5_1_frozen_application.json
.venv/bin/python scripts/evaluate_dialam_model.py --predictions results/dialam_model_generation/v5_1_n8192/predictions.jsonl --output-dir results/dialam_model_eval/v5_1_n8192 --approval APPROVE_DIALAM_MODEL_JUDGING
.venv/bin/python scripts/build_dialam_v5_1_report.py
```

# DialAM v6 Vives→QT30 transfer result

Selection decision: **RETAIN_V5_1_V6_FAILED_DEVELOPMENT_GATE**.

V6 tested one preregistered intervention: a balanced VivesDebate
relation-classification warm-up followed by the exact unchanged QT30 v5
target corpus. The Qwen3-0.6B base, QLoRA configuration, pairwise prompt,
scorer, frozen benchmark, and judge rubric were held fixed.

## Episode-disjoint development gate

| Metric | v5.1 / 8192 | v6.1 / 12288 | Delta |
|---|---:|---:|---:|
| Exact patch accuracy | 53.3% | **33.3%** | -20.0 pp |
| Edge F1 | 53.3% | **40.0%** | -13.3 pp |
| Relation macro-F1 | 54.3% | **39.7%** | -14.5 pp |
| False edges/update | 0.300 | **0.667** | +0.367 |
| NONE cases with false edges | 2/6 | **5/6** | +3 |

Development checks:

- FAIL: `edge_f1`
- FAIL: `exact_patch_accuracy`
- FAIL: `false_edges_per_update`
- PASS: `json_validity`
- FAIL: `none_scenarios_with_false_edges`
- FAIL: `relation_macro_f1`
- PASS: `schema_validity`

## Failure diagnosis

V6 produced 20 false-positive edges versus 13 false negatives. SUPPORT accounted for 13 of the false positives (65%). Compared with v5.1 development, false positives rose from 9 to 20 and NONE cases with any false edge rose from 2/6 to 5/6.

The observed regression is consistent with cross-domain relation warm-up
strengthening the model's tendency to choose a semantic relation when the
correct direct-edge label is NONE. It does not prove a language-transfer
cause, but it rules out this naive Vives curriculum as the next direction.
The next controlled experiment should stay QT30-only and optimize the
known decision boundary directly with hard candidate-level preference or
margin pairs, keeping the same development-first gate.

## Frozen benchmark

Not run. The preregistered development gate failed, so no v6
frozen predictions were generated or inspected and v5.1 remains
the selected submission model.

## Reproduce

```bash
.venv/bin/python scripts/build_dialam_training_v6.py
modal run scripts/modal_dialam_qlora.py --action train --size 12288 --dataset-version v6
modal run scripts/modal_dialam_qlora.py --action evaluate --target tuned --size 12288 --dataset-version v6 --eval-split v3_dev --output-path results/dialam_model_generation/v6_n12288_dev_raw/predictions.jsonl
.venv/bin/python scripts/calibrate_dialam_v6.py
# Run the locked frozen sequence below only when the development report says PASS.
modal run scripts/modal_dialam_qlora.py --action evaluate --target tuned --size 12288 --dataset-version v6 --eval-split frozen --output-path results/dialam_model_generation/v6_n12288_frozen_raw/predictions.jsonl
.venv/bin/python scripts/apply_dialam_v5_margin.py --predictions results/dialam_model_generation/v6_n12288_frozen_raw/predictions.jsonl --examples data/dialam/balanced_diagnostic_eval_examples.jsonl --margin <locked-development-margin> --dataset-version v6.1 --output results/dialam_model_generation/v6_1_n12288/predictions.jsonl --summary reports/dialam_v6_frozen_application.json
# Run the judge only when every deterministic frozen promotion check passes.
.venv/bin/python scripts/evaluate_dialam_model.py --predictions results/dialam_model_generation/v6_1_n12288/predictions.jsonl --output-dir results/dialam_model_eval/v6_1_n12288 --approval APPROVE_DIALAM_MODEL_JUDGING
.venv/bin/python scripts/build_dialam_v6_report.py
```

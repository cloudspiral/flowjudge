# DialAM v7 QT30 reciprocal-preference result

Selection decision: **RETAIN_V5_1_V7_FAILED_DEVELOPMENT_GATE**.

V7 tested one preregistered intervention: one epoch of reciprocal
candidate-preference continuation from the frozen v5 adapter on the exact
QT30 v5 pair corpus. No new corpus or frozen-derived training signal was used.

The interruption/resume smoke passed before the full run. Full Trainer state
was committed periodically, and the final adapter was reload-verified.

## Episode-disjoint development gate

| Metric | v5.1 / 8192 | v7.1 / 8192 | Delta |
|---|---:|---:|---:|
| Exact patch accuracy | 53.3% | **40.0%** | -13.3 pp |
| Edge F1 | 53.3% | **44.4%** | -8.9 pp |
| Relation macro-F1 | 54.3% | **43.4%** | -10.9 pp |
| False edges/update | 0.300 | **0.600** | +0.300 |
| NONE cases with false edges | 2/6 | **5/6** | +3 |

Development checks:

- FAIL: `edge_f1`
- FAIL: `exact_patch_accuracy`
- FAIL: `false_edges_per_update`
- PASS: `json_validity`
- FAIL: `none_scenarios_with_false_edges`
- FAIL: `relation_macro_f1`
- PASS: `schema_validity`

## Development diagnosis

V7 produced 18 false-positive edges and 12 false negatives. SUPPORT caused 12 of the false positives, and 5/6 all-NONE scenarios received at least one false edge.

The reciprocal preference continuation therefore moved the relation-vs-NONE
boundary in the wrong direction on development. The selected fixed-grid
margin also saturated at its registered maximum of 3.0, so a separately
preregistered score-scale calibration is the only warranted zero-training
follow-up before rejecting the checkpoint itself.

## Frozen benchmark

Not run. The preregistered development gate failed, so no v7
frozen predictions were generated or inspected and v5.1 remains
the selected submission model.

## Reproduce

```bash
.venv/bin/python scripts/build_dialam_training_v7.py
modal run --detach scripts/modal_dialam_qlora.py --action train --size 8192 --dataset-version v7 --resume-mode auto
modal run scripts/modal_dialam_qlora.py --action evaluate --target tuned --size 8192 --dataset-version v7 --eval-split v3_dev --output-path results/dialam_model_generation/v7_n8192_dev_raw/predictions.jsonl
.venv/bin/python scripts/calibrate_dialam_v7.py
# Run frozen generation only if the development report says PASS.
modal run scripts/modal_dialam_qlora.py --action evaluate --target tuned --size 8192 --dataset-version v7 --eval-split frozen --output-path results/dialam_model_generation/v7_n8192_frozen_raw/predictions.jsonl
.venv/bin/python scripts/apply_dialam_v5_margin.py --predictions results/dialam_model_generation/v7_n8192_frozen_raw/predictions.jsonl --examples data/dialam/balanced_diagnostic_eval_examples.jsonl --margin <locked-development-margin> --dataset-version v7.1 --output results/dialam_model_generation/v7_1_n8192/predictions.jsonl --summary reports/dialam_v7_frozen_application.json
# Run the judge only if every deterministic promotion condition passes.
.venv/bin/python scripts/evaluate_dialam_model.py --predictions results/dialam_model_generation/v7_1_n8192/predictions.jsonl --output-dir results/dialam_model_eval/v7_1_n8192 --approval APPROVE_DIALAM_MODEL_JUDGING
.venv/bin/python scripts/build_dialam_v7_report.py
```

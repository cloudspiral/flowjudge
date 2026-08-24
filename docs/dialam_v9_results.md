# DialAM V9 selected-model hard-negative result

Selection decision: **RETAIN_V5_1_V9_FAILED_DEVELOPMENT_GATE**.

V9 tested one preregistered QT30-only intervention: replace generic
NONE sampling with 4,096 training-only NONE comparisons that the retained
V5.1 checkpoint scored as most relation-like. The exact 4,096 V5 positive
rows were rehearsed; the model, split, loss, and evaluation gates stayed fixed.

Mining, training, and evaluation are interruption-safe. Mining commits every
128 candidates, training commits full Trainer state every 100 steps, and
evaluation commits each scenario independently.

## Episode-disjoint development gate

| Metric | v5.1 / 8192 | v9.1 / 8192 | Delta |
|---|---:|---:|---:|
| Exact patch accuracy | 53.3% | **30.0%** | -23.3 pp |
| Edge F1 | 53.3% | **22.2%** | -31.1 pp |
| Relation macro-F1 | 54.3% | **18.2%** | -36.1 pp |
| False edges/update | 0.300 | **0.000** | -0.300 |
| NONE cases with false edges | 2/6 | **0/6** | -2 |

Development checks:

- FAIL: `edge_f1`
- FAIL: `exact_patch_accuracy`
- PASS: `false_edges_per_update`
- PASS: `json_validity`
- PASS: `none_scenarios_with_false_edges`
- FAIL: `relation_macro_f1`
- PASS: `schema_validity`

## Development diagnosis

V9 eliminated false-positive edges, including all-NONE failures, but
produced 21 false negatives. It found
only 3/24 gold edges and no SUPPORT or REPHRASE true positives. The
selected-model hard-NONE correction therefore overcorrected into
underprediction rather than preserving V5.1's positive recall.

## Frozen benchmark

Not run. The preregistered development gate failed, so no V9 frozen
predictions or judge transcripts were generated and V5.1 remains the
selected submission model.

## Reproduce

```bash
PYTHONPATH=src .venv/bin/python scripts/build_dialam_v9_mining_inputs.py
modal run --detach scripts/modal_dialam_qlora.py --action mine-v9 --output-path data/dialam/training/v9_mining_scores_n12288.jsonl
PYTHONPATH=src .venv/bin/python scripts/build_dialam_training_v9.py
modal run --detach scripts/modal_dialam_qlora.py --action train --size 8192 --dataset-version v9 --resume-mode auto --output-path artifacts/dialam_qlora/v9_n8192/remote_training_result.json
modal run --detach scripts/modal_dialam_qlora.py --action evaluate --target tuned --size 8192 --dataset-version v9 --eval-split v3_dev --output-path results/dialam_model_generation/v9_n8192_dev_raw/predictions.jsonl
PYTHONPATH=src .venv/bin/python scripts/calibrate_dialam_v9.py
# Continue only when the development report says PASS_RUN_FROZEN_ONCE.
modal run --detach scripts/modal_dialam_qlora.py --action evaluate --target tuned --size 8192 --dataset-version v9 --eval-split frozen --output-path results/dialam_model_generation/v9_n8192_frozen_raw/predictions.jsonl
PYTHONPATH=src .venv/bin/python scripts/apply_dialam_v5_margin.py --predictions results/dialam_model_generation/v9_n8192_frozen_raw/predictions.jsonl --examples data/dialam/balanced_diagnostic_eval_examples.jsonl --margin <locked-development-margin> --dataset-version v9.1 --output results/dialam_model_generation/v9_1_n8192/predictions.jsonl --summary reports/dialam_v9_frozen_application.json
# Run the unchanged judge only when every deterministic check passes.
PYTHONPATH=src .venv/bin/python scripts/evaluate_dialam_model.py --predictions results/dialam_model_generation/v9_1_n8192/predictions.jsonl --output-dir results/dialam_model_eval/v9_1_n8192 --approval APPROVE_DIALAM_MODEL_JUDGING
PYTHONPATH=src .venv/bin/python scripts/build_dialam_v9_report.py
```

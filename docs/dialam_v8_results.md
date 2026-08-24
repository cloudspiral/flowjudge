# DialAM v8 prior-aware listwise result

Selection decision: **RETAIN_V5_1_V8_FAILED_DEVELOPMENT_GATE**.

V8 tested one preregistered QT30-only intervention: two distinct hard
same-block NONE candidates per balanced positive, optimized with
cross-entropy over the exact four allowed inference-label scores.

Training and evaluation are interruption-safe. Full Trainer state was
committed every 100 steps, completed runs return idempotently, and each
evaluated scenario is committed separately before the next scenario.

## Episode-disjoint development gate

| Metric | v5.1 / 8192 | v8.1 / 12288 | Delta |
|---|---:|---:|---:|
| Exact patch accuracy | 53.3% | **50.0%** | -3.3 pp |
| Edge F1 | 53.3% | **52.0%** | -1.3 pp |
| Relation macro-F1 | 54.3% | **53.2%** | -1.1 pp |
| False edges/update | 0.300 | **0.433** | +0.133 |
| NONE cases with false edges | 2/6 | **3/6** | +1 |

Development checks:

- FAIL: `edge_f1`
- FAIL: `exact_patch_accuracy`
- FAIL: `false_edges_per_update`
- PASS: `json_validity`
- FAIL: `none_scenarios_with_false_edges`
- FAIL: `relation_macro_f1`
- PASS: `schema_validity`

## Development diagnosis

V8 produced 13 false-positive and
11 false-negative edges. The largest
false-positive class was SUPPORT
with 10 edges;
3/6 all-NONE scenarios
received a false edge.

## Frozen benchmark

Not run. The preregistered development gate failed, so no v8
frozen predictions or judge transcripts were generated and v5.1
remains the selected submission model.

## Reproduce

```bash
PYTHONPATH=src .venv/bin/python scripts/build_dialam_training_v8.py
modal run --detach scripts/modal_dialam_qlora.py --action train --size 12288 --dataset-version v8 --resume-mode auto
modal run --detach scripts/modal_dialam_qlora.py --action evaluate --target tuned --size 12288 --dataset-version v8 --eval-split v3_dev --output-path results/dialam_model_generation/v8_n12288_dev_raw/predictions.jsonl
PYTHONPATH=src .venv/bin/python scripts/calibrate_dialam_v8.py
# Continue only when the development report says PASS_RUN_FROZEN_ONCE.
modal run --detach scripts/modal_dialam_qlora.py --action evaluate --target tuned --size 12288 --dataset-version v8 --eval-split frozen --output-path results/dialam_model_generation/v8_n12288_frozen_raw/predictions.jsonl
PYTHONPATH=src .venv/bin/python scripts/apply_dialam_v5_margin.py --predictions results/dialam_model_generation/v8_n12288_frozen_raw/predictions.jsonl --examples data/dialam/balanced_diagnostic_eval_examples.jsonl --margin <locked-development-margin> --dataset-version v8.1 --output results/dialam_model_generation/v8_1_n12288/predictions.jsonl --summary reports/dialam_v8_frozen_application.json
# Run the unchanged judge only when every deterministic check passes.
PYTHONPATH=src .venv/bin/python scripts/evaluate_dialam_model.py --predictions results/dialam_model_generation/v8_1_n12288/predictions.jsonl --output-dir results/dialam_model_eval/v8_1_n12288 --approval APPROVE_DIALAM_MODEL_JUDGING
PYTHONPATH=src .venv/bin/python scripts/build_dialam_v8_report.py
```

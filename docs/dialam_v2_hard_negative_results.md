# DialAM v2 hard-negative result

Decision: **V2_DID_NOT_CLEAR_MATERIAL_IMPROVEMENT_BAR**.

The sole intervention is the v2 data mix and deterministic hard-negative selection. Qwen3-0.6B, n=2048, the QLoRA configuration and seed, frozen 30-scenario eval, deterministic metrics, and blinded judge rubric are unchanged.

| Metric | v1 / n=2048 | v2 / n=2048 | Delta |
|---|---:|---:|---:|
| Exact patch accuracy | 26.7% | 23.3% | -0.033 |
| Edge precision | 16.7% | 18.8% | +0.021 |
| Edge recall | 16.7% | 25.0% | +0.083 |
| Edge F1 | 16.7% | 21.4% | +0.048 |
| Relation macro-F1 | 16.7% | 18.8% | +0.021 |
| False edges/update | 0.667 | 0.867 | +0.200 |
| False edges | 20 | 26 | +6 |
| NONE cases with false edges | 2/6 | 5/6 | +3 |
| SUPPORT F1 | 0.0% | 23.1% | +0.231 |
| ATTACK F1 | 26.7% | 0.0% | -0.267 |
| REPHRASE F1 | 23.5% | 33.3% | +0.098 |
| Judge robustness /4 | 2.10 | 1.43 | -0.667 |

## Data intervention

v2 raises NONE coverage from 819 to 1024 examples. Every selected NONE example is a no-edge fixed block from an update whose true direct edge occurs in another block, and every one has lexical content overlap with an in-block candidate. No easy-negative duplication, embeddings, semantic retrieval, new source data, model change, or hyperparameter search was used.

## Material-improvement rule

Before training, material improvement was defined as at least +0.05 absolute edge F1, no increase in false edges/update, and retained 100% JSON/schema validity. v2 does not clear that rule or the original frozen reliability bar. The machine-readable report records every condition, along with model/data/checkpoint hashes and the private artifact paths and hashes.

## Reproduce

```bash
.venv/bin/python scripts/build_dialam_training_v2.py
modal run scripts/modal_dialam_qlora.py --action train --size 2048 --dataset-version v2
modal run scripts/modal_dialam_qlora.py --action evaluate --target tuned --size 2048 --dataset-version v2 --output-path results/dialam_model_generation/v2_n2048/predictions.jsonl
.venv/bin/python scripts/evaluate_dialam_model.py --predictions results/dialam_model_generation/v2_n2048/predictions.jsonl --output-dir results/dialam_model_eval/v2_n2048 --approval APPROVE_DIALAM_MODEL_JUDGING
.venv/bin/python scripts/build_dialam_v2_report.py
```

QT30-derived rows, candidate responses, judge transcripts, and checkpoints remain local/ignored. Only aggregate reports, hashes, schemas, and reconstruction/publication scripts are committed.

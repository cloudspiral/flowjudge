# DialAM v5 pairwise-classification result

Development decision: **RETAIN_V3_V5_FAILED_DEVELOPMENT_GATE**.

| Development metric | Pairwise base | v3 / 4096 | v4 / 8192 | v5 / 8192 |
|---|---:|---:|---:|---:|
| Exact patch accuracy | 3.3% | 26.7% | 30.0% | **40.0%** |
| Edge precision | 6.3% | 28.6% | 41.2% | **41.2%** |
| Edge recall | 37.5% | 16.7% | 29.2% | **58.3%** |
| Edge F1 | 10.8% | 21.1% | 34.1% | **48.3%** |
| Relation macro-F1 | 7.5% | 17.3% | 31.5% | **49.5%** |
| False edges/update | 4.467 | 0.333 | 0.333 | **0.667** |
| NONE cases with false edges | 6/6 | 2/6 | 4/6 | **6/6** |

## Gate

- PASS: `exact_patch_accuracy`
- PASS: `edge_f1`
- PASS: `relation_macro_f1`
- FAIL: `false_edges_per_update`
- FAIL: `none_scenarios_with_false_edges`
- PASS: `json_validity`
- PASS: `schema_validity`

## Failure diagnosis

V5 produced 20 false-positive and 10 false-negative development edges. Its six NONE scenarios contained 6 cases with a false edge. The naturally eligible training-candidate positive rate is 7.379%, versus 50.0% in the exact contrast corpus.

The external block-level contract is unchanged. V5 scores one allowed label
for every supplied candidate ID and deterministically assembles non-NONE
decisions into the exact patch schema.

The complete chronological record and chart are in
[`docs/dialam_experiment_history.md`](dialam_experiment_history.md).

## Reused frozen benchmark

Raw argmax v5 did not advance because its development gate failed.
The separately preregistered v5.1 prior-correction fallback did pass
development, completed one frozen evaluation, and replaced v3; see
[`docs/dialam_v5_1_results.md`](dialam_v5_1_results.md).

## Reproduce

```bash
.venv/bin/python scripts/build_dialam_training_v5.py
modal run scripts/modal_dialam_qlora.py --action train --size 8192 --dataset-version v5
modal run scripts/modal_dialam_qlora.py --action evaluate --target base --size 8192 --dataset-version v5 --eval-split v3_dev --output-path results/dialam_model_generation/v5_base_dev/predictions.jsonl
modal run scripts/modal_dialam_qlora.py --action evaluate --target tuned --size 8192 --dataset-version v5 --eval-split v3_dev --output-path results/dialam_model_generation/v5_n8192_dev/predictions.jsonl
.venv/bin/python scripts/build_dialam_v5_report.py
```

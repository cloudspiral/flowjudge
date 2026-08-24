# DialAM v7.2 transition-complete calibration result

Selection decision: **RETAIN_V5_1_V7_2_FAILED_DEVELOPMENT_GATE**.

V7.2 made no model calls and performed no training. It reused the exact
v7 raw development scores and evaluated every possible NONE-margin output
configuration above the v7.1 boundary.

## Development result

The score-only rescue evaluated 31 unique
decision states and selected margin `5.505193236283958`.

| Metric | v7.1 / 8192 | v7.2 / 8192 | Delta |
|---|---:|---:|---:|
| Exact patch accuracy | 40.0% | **53.3%** | +13.3 pp |
| Edge F1 | 44.4% | **52.2%** | +7.7 pp |
| Relation macro-F1 | 43.4% | **50.7%** | +7.3 pp |
| False edges/update | 0.600 | **0.333** | -0.267 |
| NONE cases with false edges | 5/6 | **2/6** | -3 |

Development checks:

- FAIL: `edge_f1`
- PASS: `exact_patch_accuracy`
- FAIL: `false_edges_per_update`
- PASS: `json_validity`
- PASS: `none_scenarios_with_false_edges`
- FAIL: `relation_macro_f1`
- PASS: `schema_validity`

## Interpretation

The larger margin recovered exact accuracy to the v5.1 development
baseline (53.3%) and reduced false positives
from 18 to 10.
However, edge F1 remained 52.2% versus the locked
56.0% gate, macro-F1 remained below its gate, and false edges/update
remained above 0.300. The checkpoint therefore was not eligible for
frozen evaluation.

SUPPORT still accounted for 7 of
10 false positives. Calibration corrected
much of the score-scale shift, but cannot repair the remaining relation
ranking errors—especially weak REPHRASE recall—without changing training.

## Frozen benchmark

Not run. The unchanged development gate failed. Candidate calls: 0;
judge calls: 0. V5.1 remains the selected submission model.

## Reproduce

```bash
PYTHONPATH=src .venv/bin/python scripts/calibrate_dialam_v7_2.py
PYTHONPATH=src .venv/bin/python scripts/build_dialam_v7_2_report.py
```

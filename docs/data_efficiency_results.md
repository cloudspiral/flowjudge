# FlowJudge data-efficiency curve

Every checkpoint changes only the number of nested v1 training examples; the base, hyperparameters, own evaluation set, prompt, deterministic scorer, and fixed judge remain constant.

| Training examples | Judge | Spec adherence | Robustness | Exact graph | Edge F1 | Topical FP |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 12 | complete | 0.78/4 | 1.67/4 | 0.0% | 0.0% | 0.0% |
| 24 | complete | 0.67/4 | 1.78/4 | 0.0% | 0.0% | 0.0% |
| 48 | pending | n/a | n/a | 0.0% | 0.0% | 0.0% |
| 96 | pending | n/a | n/a | 0.0% | 16.0% | 0.0% |

## Untuned base reference

Qwen3 0.6B base: 0.67/4 Spec adherence, 0.89/4 Robustness, 0.0% exact graph match, and 0.0% edge F1.

The fixed Sol judge completed n=12 and n=24. It is explicitly pending for later points because the OpenAI account returned `insufficient_quota`; no weaker substitute judge was used. Deterministic graph metrics and raw local responses are complete for all four points.

## Minimum viable dataset size

No tested size reliably holds the full behavior. N=96 is the first point with a nonzero held-out edge F1, but 0% exact graph match and 16% edge F1 do not justify calling it viable. The minimum is therefore **not established at N ≤ 96**, and must also be confirmed on the unavailable staff-held-out set.

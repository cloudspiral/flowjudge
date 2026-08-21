# FlowJudge prompt-ceiling ablation

**Verdict: SURVIVES PROMPT-CEILING GATE.**

Best combination: `openai:gpt-5.4-mini-2026-03-17 + few_shot` with 58.6% mean Spec adherence, 69.5% mean Robustness, and 15.6% deterministic exact graph match.

## Required 2 × 3 table

Each cell is **mean Spec adherence / mean Robustness**, with each 0–4 judge score normalized to a percentage.

| Candidate model | Zero-shot | Few-shot | Structured checklist |
| --- | ---: | ---: | ---: |
| `anthropic:claude-haiku-4-5-20251001` | 53.9% / 55.5% | 53.1% / 58.6% | 48.4% / 55.5% |
| `openai:gpt-5.4-mini-2026-03-17` | 52.3% / 55.5% | 58.6% / 69.5% | 54.7% / 64.8% |

Gate used for the assignment decision: mean Spec adherence ≥95% and mean Robustness ≥90% in one model/prompt cell.

## Deterministic cross-check

| Candidate model | Prompt | Valid JSON | Exact graph | Edge precision | Edge recall | Edge F1 | Topical-nonresponse FP |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `anthropic:claude-haiku-4-5-20251001` | few_shot | 100.0% | 18.8% | 59.5% | 67.1% | 63.1% | 25.0% |
| `anthropic:claude-haiku-4-5-20251001` | structured_checklist | 100.0% | 12.5% | 51.8% | 62.9% | 56.8% | 12.5% |
| `anthropic:claude-haiku-4-5-20251001` | zero_shot | 96.9% | 21.9% | 52.6% | 72.9% | 61.1% | 25.0% |
| `openai:gpt-5.4-mini-2026-03-17` | few_shot | 100.0% | 15.6% | 46.1% | 58.6% | 51.6% | 37.5% |
| `openai:gpt-5.4-mini-2026-03-17` | structured_checklist | 100.0% | 12.5% | 44.6% | 58.6% | 50.6% | 12.5% |
| `openai:gpt-5.4-mini-2026-03-17` | zero_shot | 100.0% | 12.5% | 38.5% | 67.1% | 49.0% | 37.5% |

## Failure that survives the best prompt

The best cell still produced 27 non-exact scenarios, with 29 missed and 48 spurious edges. Its dominant recurring failure was **spurious response edges**, concentrated most often in scenarios tagged `same_side_extension_trap`. This is the specific behavior the distilled dataset and error-driven v2 examples should target.

## Representative failures

| Scenario | Phenomena | Missed edges | Spurious edges |
| --- | --- | --- | --- |
| `vives_debate1` | local_direct, same_side_extension_trap, topical_distractor | `[]` | `[{"source":"U83","target":"U82","type":"responds_to"}]` |
| `vives_debate1_b` | local_direct, same_side_extension_trap | `[{"source":"U165","target":"U164","type":"responds_to"}]` | `[{"source":"U161","target":"U160","type":"responds_to"},{"source":"U164","target":"U161","type":"responds_to"},{"source":"U164","target":"U162","type":"responds_to"},{"source":"U166","target":"U165","type":"responds_to"}]` |
| `vives_debate1_c` | long_distance, same_side_extension_trap, topical_distractor | `[{"source":"U167","target":"U23","type":"responds_to"}]` | `[{"source":"U166","target":"U22","type":"responds_to"},{"source":"U167","target":"U22","type":"responds_to"},{"source":"U168","target":"U22","type":"responds_to"},{"source":"U21","target":"U18","type":"responds_to"}]` |
| `vives_debate2_b` | long_distance, branching, response_chain, same_side_extension_trap | `[{"source":"U185","target":"U169","type":"responds_to"}]` | `[]` |
| `vives_debate2_c` | long_distance, branching, rephrase_trap | `[{"source":"U304","target":"U242","type":"responds_to"},{"source":"U304","target":"U243","type":"responds_to"},{"source":"U304","target":"U244","type":"responds_to"}]` | `[]` |
| `vives_debate3` | branching, rephrase_trap | `[{"source":"U152","target":"U150","type":"responds_to"},{"source":"U152","target":"U151","type":"responds_to"}]` | `[{"source":"U153","target":"U150","type":"responds_to"}]` |

## Reproduction record

- Run directory: `results/20260821T055241.543347Z`
- Candidate calls: 192
- Fixed-judge calls: 192
- Judge model: `gpt-5.6-sol`
- Fixed-judge valid rubric rate: 100.0%
- Raw candidate text, judge text, and complete SDK envelopes are preserved under the run directory.

This ablation does not train or fine-tune any model.

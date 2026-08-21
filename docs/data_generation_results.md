# FlowJudge v1 data-generation report

## Outcome

The first real filtered dataset is complete. A fixed GPT-5.6 Sol teacher and a
separate GPT-5.6 Sol quality-filter call processed 115 source-derived candidates
from VivesDebate Debates 1–7. The filter accepted 113 and rejected 2 after at
most three feedback-driven rewrite attempts. The published nested v1 slices use
the first 96 accepted candidates in deterministic candidate order.

| Measure | Result |
| --- | ---: |
| Source candidates | 115 |
| Accepted before cap | 113 |
| Rejected after three attempts | 2 |
| Accepted on attempt 1 | 74 |
| Accepted on attempt 2 | 32 |
| Accepted on attempt 3 | 7 |
| Published v1 examples | 96 |
| Model-facing ADUs | 902 |
| Gold response edges | 226 |
| Mean ADUs per example | 9.40 |
| Mean gold edges per example | 2.35 |

Every published row passed all five filter booleans: overall acceptance,
meaning preservation, edge-support preservation, no new response cues, and
fluent/contextually clear atomic units. Pydantic validation separately enforces
unit IDs, chronology, stance, graph endpoints, later-to-earlier direction, and
cross-side edges.

## Leakage and split checks

- Training sources are complete Debates 1–7 only.
- Own evaluation sources are complete Debates 8–10 only.
- The staff-held-out set remains external and unseen.
- The internal source anchor pair is not included in model-facing prompts.
- Teacher and filter calls can see the fixed graph for preservation and
  validation; the student receives only the neutral transcript and target JSON.
- The four efficiency sets are nested deterministic prefixes: 12, 24, 48, 96.

## Preserved evidence

The ignored local run `results/data_generation/20260821T061839.058137Z/`
contains all 115 records, every teacher/filter response, every complete SDK
envelope, retry feedback, the final manifest, and the two rejected cases. The
committed `data/training/v1_manifest.json` records the source run and materialized
paths without exposing credentials.

The two rejected windows were not backfilled with synthetic text. Their source
ADUs combined claims that the filter could not make both atomic and faithful
without adding an unsupported relation, so later valid real candidates filled
the 96-example cap.

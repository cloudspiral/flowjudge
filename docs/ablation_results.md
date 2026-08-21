# FlowJudge formal ablation results

## Outcome

The formal weak-baseline ablation did **not** meet the FlowJudge kill condition.
The strongest candidate/prompt combination, Claude Haiku 4.5 zero-shot, achieved
8/32 normalized exact graph matches (25.0%) and 2/8 on the held-out split. The
best OpenAI combination achieved 3/32 exact matches (9.4%). Repeated failures
occurred in edge orientation, target attachment, branching, long-distance
cross-application, and distinguishing a response from a rephrase or merely
related statement.

This supports continuing toward an SLM proof of concept, but only after a small
gold-validity adjudication described below. It does **not** show that flagship
frontier models are unreliable: the candidate models were intentionally weak,
and the strong `gpt-5.6-sol` model was used only as a judge that received the
gold graph.

## Run and artifact integrity

- Run: `results/20260821T020430.050266Z/`
- OpenAI candidate: `gpt-5.4-nano-2026-03-17`
- Anthropic candidate: `claude-haiku-4-5-20251001`
- Fixed judge: `gpt-5.6-sol`
- Matrix: 32 scenarios × 2 candidate models × 3 prompts
- Completed: 192/192 candidate calls and 192/192 judge calls
- Preserved: 192 candidate texts, 192 judge texts, and 384 valid SDK envelopes
- Integrity check: every preserved text exactly matches its JSONL record

## Primary deterministic results

Strict scoring requires the response to be schema-valid bare JSON. Normalized
scoring permits exactly one surrounding Markdown code fence and makes no other
repair.

| Metric | Strict | Fence-normalized |
|---|---:|---:|
| Schema-valid output | 94/192 (49.0%) | 190/192 (99.0%) |
| Exact graph match | 4/192 (2.1%) | 24/192 (12.5%) |
| Edge precision | 28.5% | 43.3% |
| Edge recall | 15.0% | 49.0% |
| Edge F1 | 19.7% | 46.0% |
| Topical-nonresponse false-positive rate | 12.5% | 18.8% |

Claude wrapped every response in a Markdown fence, so its strict validity and
strict exact-match rates were zero. All 96 Claude responses became schema-valid
after fence removal. Two OpenAI zero-shot responses remained invalid because
they contained duplicate edges.

## Results by candidate and prompt

| Candidate and prompt | Strict valid | Strict exact | Normalized exact | Precision | Recall | F1 | Topical FP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Claude Haiku 4.5, zero-shot | 0/32 | 0/32 | **8/32 (25.0%)** | 53.5% | 75.7% | **62.7%** | 12.5% |
| Claude Haiku 4.5, few-shot | 0/32 | 0/32 | 6/32 (18.8%) | **59.2%** | 64.3% | 61.6% | 25.0% |
| Claude Haiku 4.5, checklist | 0/32 | 0/32 | 6/32 (18.8%) | 56.3% | 64.3% | 60.0% | **0.0%** |
| GPT-5.4 nano, zero-shot | 30/32 | 0/32 | 0/32 | 8.3% | 10.0% | 9.1% | 0.0% |
| GPT-5.4 nano, few-shot | 32/32 | 1/32 (3.1%) | 1/32 (3.1%) | 40.8% | 57.1% | 47.6% | 62.5% |
| GPT-5.4 nano, checklist | 32/32 | **3/32 (9.4%)** | **3/32 (9.4%)** | 41.0% | 22.9% | 29.4% | 12.5% |

No prompt consistently fixed the task. The checklist reduced some false
positives, but OpenAI checklist recall collapsed. Claude zero-shot produced the
best exact and F1 results.

## Held-out and category results

The development and held-out exact rates were identical at 12.5%:

| Split | Assignments | Normalized exact | Precision | Recall | F1 | Topical FP |
|---|---:|---:|---:|---:|---:|---:|
| Development | 144 | 18/144 (12.5%) | 41.9% | 52.8% | 46.7% | 23.3% |
| Held-out | 48 | 6/48 (12.5%) | 47.5% | 41.3% | 44.2% | 11.1% |

The two synthetic cases were easier: each was solved by 3/6 combinations. The
30 real VivesDebate excerpts were solved in only 18/180 assignments (10.0%).
Eighteen of the 30 real excerpts were never solved exactly by any combination.

The strongest held-out results were Claude zero-shot and Claude checklist at
2/8 each. Claude few-shot and OpenAI checklist reached 1/8; both remaining
OpenAI prompts reached 0/8.

## Results by structural phenomenon

Tags overlap, so these rows are diagnostics rather than independent buckets.

| Phenomenon | Assignments | Normalized exact | Edge F1 |
|---|---:|---:|---:|
| Rephrase trap | 54 | 1/54 (1.9%) | 40.7% |
| Local direct | 54 | 3/54 (5.6%) | 47.3% |
| Long distance | 72 | 4/72 (5.6%) | 34.1% |
| Response chain | 60 | 7/60 (11.7%) | 45.3% |
| Same-side-extension trap | 150 | 18/150 (12.0%) | 44.7% |
| Topical distractor | 48 | 6/48 (12.5%) | 45.3% |
| Branching | 84 | 15/84 (17.9%) | 45.5% |
| Dropped argument (synthetic) | 6 | 3/6 (50.0%) | 75.0% |
| Cross-application (synthetic) | 6 | 3/6 (50.0%) | 68.3% |

## Recurring failure structure

Across 192 normalized assignments:

- 137 (71.4%) over-linked with at least one spurious edge;
- 106 (55.2%) under-linked with at least one missed edge;
- 75 (39.1%) did both;
- the 270 spurious edges included 76 forward-in-time edges, 29 direct reversals
  of a gold edge, 47 wrong-target attachments from an otherwise relevant source,
  and 38 same-side links;
- models selected typed hard negatives 22 times: 9 topical nonresponses, 9
  repetitions/rephrases, and 4 same-side extensions;
- among 42 model-by-source branching opportunities, only 5 reconstructed every
  branch, 18 recovered only part of the branch, and 19 missed the branch entirely.

Representative failures:

1. **Orientation collapse:** on `vives_debate2_b`, OpenAI zero-shot reversed
   three gold arrows, including `U185 → U54` and `U186 → U185`.
2. **Branch truncation:** `vives_debate3_c` contains six gold targets for U259.
   Every combination missed at least five of them, and no combination reconstructed the
   complete eight-edge graph.
3. **Rephrase/response over-linking:** on `vives_debate9_c`, five combinations
   added both `U246 → U65` and `U248 → U65`; U248 is the pre-annotated rephrase
   hard negative, while the gold responses begin at U250.
4. **Prompt-example leakage:** three OpenAI zero-shot responses emitted `U4 → U2`
   or `U2 → U4` even though those unit IDs did not exist in the transcript.
5. **Duplicate serialization:** two OpenAI zero-shot outputs repeated an edge
   and therefore failed the strict and normalized Pydantic schema.

## Fixed-judge results

The fixed judge produced valid JSON on 192/192 assignments. Its correctness
decision agreed with fence-normalized deterministic exact match on 192/192 and
its missed/spurious edge diagnosis matched on 190/192. The two diagnosis
differences were the duplicate-edge OpenAI outputs: deterministic parsing
rejected the whole graph, while the judge interpreted the deduplicated semantic
content. This confirms the judge was useful as a secondary check, but the
deterministic scorer remains authoritative.

## Gold-validity warning

The ablation also exposed a validity risk in treating VivesDebate `CA` as a
complete response graph. Several frequently predicted spurious edges are
plausible direct responses under FlowJudge's written definition:

- `vives_debate1_c` U168 may mitigate U23's elite-access objection by proposing
  public healthcare (predicted by four combinations);
- `vives_debate4_c` U277 may directly support the answer to U275's paternalism
  objection (four combinations);
- `vives_debate5_b` U175 appears to attack U174's legalization solution (four
  combinations);
- `vives_debate7_b` U145 appears to answer U139's pronatalist objection (three
  combinations);
- `vives_debate9_c` U248 explicitly challenges the premise that the child is a
  product (five combinations), although it was pre-registered as a rephrase.

Conversely, some universally missed CA-derived branch edges attach a broad
denial to every supporting fragment, which may be stricter than a human
response-graph annotation would be. These observations do not alter the
pre-registered primary score, but training directly on the current CA mapping
could teach annotation artifacts rather than the intended relation.

## Recommendation

**Do not kill FlowJudge. Do not fine-tune yet.** The weak baselines exhibit a
large, repeated structural gap, but the next step should be a compact independent
adjudication—not a review of all 32 cases—covering:

1. the roughly 10–15 highest-consensus spurious pairs;
2. the gold edges missed by all six combinations; and
3. the multi-target CA branches that may represent grouped rather than direct
   unit-level responses.

Keep the pre-registered score unchanged and report any adjudicated graph as a
separate sensitivity analysis. If the core failures remain after that check,
proceed to an SLM proof of concept using a new debate-level train/test split;
the current prompt-ablation split is not safe for fine-tuning because excerpts
from the same debate occur in both partitions.

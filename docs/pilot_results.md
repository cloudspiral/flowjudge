# FlowJudge pilot results

## Run identity

- Run directory: `results/20260821T012454.328398Z`
- Benchmark commit: `1d67f3c`
- Candidate assignments: 72
- Fixed-judge assessments: 72
- Candidate models: `gpt-5.4-nano-2026-03-17` and
  `claude-haiku-4-5-20251001`
- Fixed judge: `gpt-5.6-sol`
- Prompts: zero-shot, few-shot, and structured checklist
- Cases: 10 VivesDebate excerpts and 2 targeted synthetic cases

The experiment completed in one run. The artifact audit found 72 unique records,
72 raw-assignment directories, and all 288 expected raw text and SDK-envelope
files. Every envelope is valid JSON, and every preserved candidate and judge
text file exactly matches its corresponding record.

## Primary deterministic result

Strict Pydantic parsing and exact graph comparison remain primary. Markdown
fences, commentary, missing fields, extra fields, duplicate edges, and invalid
relation types are invalid output.

| Metric | Result |
|---|---:|
| Valid JSON | 35/72 (48.6%) |
| Exact graph match | 2/72 (2.8%) |
| Edge precision | 37.7% |
| Edge recall | 16.0% |
| Edge F1 | 22.5% |
| True-positive edges | 26 |
| False-positive edges | 43 |
| False-negative edges | 136 |
| Curated hard-negative false-positive rate | 4/84 (4.8%) |

The curated hard-negative metric covers topically related nonresponses,
same-side extensions, repetition, and independent counterarguments. The four
false positives were Debate1 `U83 -> U82`, Debate10 `U253 -> U251`, the dropped
argument case `U3 -> U1`, and Debate9 `U200 -> U199`.

## Results by model and prompt

| Candidate and prompt | Valid JSON | Exact | Precision | Recall | F1 | Hard-negative FP |
|---|---:|---:|---:|---:|---:|---:|
| GPT-5.4 nano, zero-shot | 12/12 | 0/12 | 13.8% | 14.8% | 14.3% | 0.0% |
| GPT-5.4 nano, few-shot | 11/12 | 1/12 | 53.8% | 51.9% | 52.8% | 21.4% |
| GPT-5.4 nano, checklist | 12/12 | 1/12 | 57.1% | 29.6% | 39.0% | 7.1% |
| Claude Haiku 4.5, zero-shot | 0/12 | 0/12 | n/a | 0.0% | n/a | 0.0% |
| Claude Haiku 4.5, few-shot | 0/12 | 0/12 | n/a | 0.0% | n/a | 0.0% |
| Claude Haiku 4.5, checklist | 0/12 | 0/12 | n/a | 0.0% | n/a | 0.0% |

The only strict exact matches were GPT-5.4 nano on VivesDebate Debate5 under
the few-shot and checklist prompts.

## Results by scenario category

| Category | Assignments | Valid JSON | Exact | Precision | Recall | F1 | Hard-negative FP |
|---|---:|---:|---:|---:|---:|---:|---:|
| VivesDebate real | 60 | 48.3% | 3.3% | 36.4% | 15.9% | 22.1% | 4.5% |
| Synthetic dropped argument | 6 | 50.0% | 0.0% | 44.4% | 33.3% | 38.1% | 8.3% |
| Synthetic cross-application | 6 | 50.0% | 0.0% | 40.0% | 8.3% | 13.8% | 0.0% |

## Secondary judge

The fixed judge returned schema-valid JSON for 72/72 assignments. Its binary
correct/incorrect decision agreed with the deterministic result on 59/72
(81.9%), while its exact missed-edge and spurious-edge diagnosis matched on
35/72 (48.6%). This supports retaining deterministic graph comparison as the
primary evaluation.

The judge often evaluated the graph inside Claude's Markdown fences rather than
enforcing the benchmark's strict serialization rule. That explains much of its
disagreement with deterministic validity.

## Diagnostic normalization

All 36 Claude responses wrapped otherwise parseable JSON in Markdown code
fences despite explicit instructions to return JSON only. This is a real
protocol-compliance failure under the primary metric, but it can obscure the
underlying structure result. A read-only diagnostic removed only the outer
fence and changed no graph content:

| Claude prompt after fence removal | Exact | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Zero-shot | 6/12 | 73.5% | 92.6% | 82.0% |
| Few-shot | 4/12 | 70.0% | 77.8% | 73.7% |
| Checklist | 3/12 | 69.0% | 74.1% | 71.4% |

Across all normalized Claude assignments, exact match was 13/36 (36.1%) and
edge F1 was 75.9%. Thus Claude's strict 0/36 is not solely a debate-structure
failure, but its best semantic result still falls well below the 11/12 kill
threshold.

## Representative failures

1. **Temporal direction reversal.** GPT-5.4 nano zero-shot produced 22 edges
   whose source was earlier than the target. On Debate2 it emitted
   `U144 -> U145` and `U144 -> U146` instead of the gold `U145 -> U144` and
   `U146 -> U144`.
2. **Attaching a rebuttal to the wrong claim.** GPT-5.4 nano few-shot correctly
   found three Debate10 edges but linked `U253 -> U251`, a same-side extension,
   instead of `U253 -> U252`, the intervening opposing challenge.
3. **Missing branching and cross-application.** On the synthetic
   cross-application case, GPT-5.4 nano with the checklist omitted both
   `U3 -> U1` and `U3 -> U2`, missed `U6 -> U4`, and invented `U6 -> U3`.
4. **Treating rephrases as fresh responses.** Claude zero-shot found every
   Debate3 gold edge from `U152`, then duplicated all three from `U153`, even
   though `U153` merely rephrases the same-side challenge.
5. **Over-linking an extension.** Claude zero-shot found Debate1's gold
   `U85 -> U83` but also linked the broader freedom extension `U87 -> U83`.
6. **Serialization noncompliance.** Claude wrapped every one of its 36 answers
   in Markdown fences. GPT-5.4 nano produced one additional schema-invalid
   answer by omitting `type` from all four edges in Debate7 few-shot.

## Decision

Do **not** kill FlowJudge. No strict model/prompt combination exceeded 1/12
exact, and the strongest fence-normalized diagnostic reached only 6/12. The
failures repeat around edge orientation, exact target attachment, distinguishing
rephrases/extensions from new responses, and reconstructing one-to-many
cross-applications.

Proceed to a formal 30+ scenario ablation, but report two evaluation layers:

1. strict end-to-end protocol accuracy; and
2. graph accuracy after a narrowly documented serialization normalization or
   provider-native schema constraint.

This prevents a specialized model from appearing superior merely because it
avoids Markdown fences. The expanded set should deliberately vary response
distance, branch count, rephrase density, same-side adjacency, and the number of
eligible opposing pairs.

## Token use and estimated cost

The preserved envelopes report 14,421 input and 1,462 output tokens for GPT-5.4
nano, 17,043 input and 2,635 output tokens for Claude Haiku 4.5, and 29,767 input
and 6,542 output tokens for GPT-5.6 Sol. Using the providers' posted standard
token prices at the time of the run gives an estimated total of approximately
**$0.38**. This is an estimate, not a billing-console reconciliation.

# DialAM hosted-model prompt-ceiling results

**Verdict: PASS — proceed to the small-model baseline and QLoRA decision.**

The complete six-cell matrix contains 30 frozen, one-block scenarios per cell.
No GPT Mini or Claude Haiku prompt strategy cleared every preregistered
reliability threshold. This report stops at the prompt-ceiling decision; no
DialAM training was started.

## Formal matrix

| Candidate | Prompt | JSON valid | Exact patch | Edge P / R / F1 | Relation macro-F1 | False edges/update | Judge Spec / Robustness |
|---|---|---:|---:|---:|---:|---:|---:|
| Claude Haiku 4.5 | zero-shot | 0.0% | 0.0% | 0.0% / 0.0% / 0.0% | 0.0% | 0.00 | 2.80 / 2.10 |
| Claude Haiku 4.5 | few-shot | 33.3% | 20.0% | 55.6% / 20.8% / 30.3% | 29.8% | 0.13 | 3.07 / 2.43 |
| Claude Haiku 4.5 | strong structured | 0.0% | 0.0% | 0.0% / 0.0% / 0.0% | 0.0% | 0.00 | 2.43 / 2.17 |
| GPT-5.4 Mini | zero-shot | 100.0% | 26.7% | 26.1% / 50.0% / 34.3% | 31.3% | 1.13 | 4.00 / 1.83 |
| GPT-5.4 Mini | few-shot | 100.0% | 26.7% | 25.0% / 37.5% / 30.0% | 24.0% | 0.90 | 4.00 / 1.80 |
| GPT-5.4 Mini | strong structured | 100.0% | 26.7% | 25.0% / 45.8% / 32.4% | 31.7% | 1.10 | 4.00 / 2.00 |

All cells contain exactly 30 scenarios. Every cell had 100% valid judge JSON,
zero invalid candidate IDs, and no reversed evaluable edges. Haiku's zero-shot
and structured outputs consistently wrapped JSON in Markdown fences, which is
invalid under the frozen bare-object Behavior Spec. Few-shot reduced but did
not eliminate that failure: 20/30 responses remained invalid JSON.

## Surviving failure

The strongest cell by edge F1 was GPT-5.4 Mini zero-shot. It produced correct
bare JSON and valid IDs, but overpredicted semantic relations: 34 false-positive
edges versus 12 false negatives, only 26.1% edge precision, and 1.13 false edges
per update. It emitted at least one false edge on all six all-negative
scenarios. The few-shot and strong-structured GPT Mini cells exhibited the same
pattern on five and six of the six all-negative scenarios, respectively.

This is the behavior that survives the best prompting attempt: the hosted
baseline recognizes plausible topical or argumentative relatedness but does not
reliably restrict its patch to corpus-annotated direct relations.

## Gate decision

The measurement is complete and no cell cleared every threshold. The corrected
decision rule therefore returns **PASS**: prompting remains below the frozen
reliability bar, so the next decision may consider an untuned small-model
baseline and QLoRA. If any cell had cleared the bar, this gate would have
returned FAIL instead.

## Reproduction evidence

- Run: `results/dialam_prompt_ceiling/20260823T043624.705184Z`
- Candidates: 90 `gpt-5.4-mini-2026-03-17` and 90
  `claude-haiku-4-5-20251001`
- Fixed judge: 180 `gpt-5.4-mini-2026-03-17` calls using the unchanged rubric
- Calls completed: 180 candidate + 180 judge
- Persisted artifacts: 180 candidate prompts/responses/envelopes and 180 judge
  prompts/responses/envelopes
- Judge identity scan: no provider, model, or prompt-strategy identity found
- Forbidden-model scan: no Sol or Opus model IDs found anywhere in the run
- Frozen judge-rubric SHA-256:
  `b27fe77a9adce4fe586327fa556a51352efb76d402b93676df2173225be35f66`

The run directory is intentionally Git-ignored because it contains raw corpus
text and model transcripts. Aggregate metrics and this report contain no raw
QT30 dialogue text.

# DialAM prompt-ceiling gate

## Status

**PASS — formal matrix complete.** No DialAM QLoRA training has started. All
180 hosted-baseline candidate calls and 180 fixed-judge calls completed; no
model/prompt cell cleared the preregistered reliability bar. Full metrics and
failure analysis are in [`dialam_prompt_ceiling_results.md`](dialam_prompt_ceiling_results.md).

## Frozen experiment

- Evaluation: 30 one-block update scenarios from the six frozen held-out QT30
  parent episodes.
- Diagnostic mix: 8 SUPPORT, 8 ATTACK, 8 REPHRASE, 6 all-negative.
- Hosted-model baselines: OpenAI `gpt-5.4-mini-2026-03-17` and Anthropic
  `claude-haiku-4-5-20251001`.
- Prompts: zero-shot, few-shot, and strong structured.
- Fixed judge: OpenAI `gpt-5.4-mini-2026-03-17`, scoring only Spec adherence
  and Robustness with the unchanged frozen rubric.
- Relation correctness: deterministic gold metrics only.
- Raw preservation: exact candidate prompt/response/envelope and judge
  prompt/response/envelope for every assignment under the ignored run folder.
- Judge blinding: the judge prompt receives task input, gold patch, candidate
  response, and deterministic diagnostics, but no provider, model ID, or prompt
  name.
- Frozen judge rubric SHA-256:
  `b27fe77a9adce4fe586327fa556a51352efb76d402b93676df2173225be35f66`.
  DialAM base-vs-tuned evaluation must call the same prompt builder and verify
  this exact digest; runtime construction now fails if the rubric changes, and
  it may not be tuned after viewing ceiling results.

This is 30 scenarios × 2 families × 3 prompts = 180 candidate calls and 180
judge calls.

The older smoke view also contains 30 updates, but those updates span 46
blocks. It is not the frozen prompt-ceiling view. The frozen balanced diagnostic
view contains exactly 30 scenarios and exactly 30 blocks; each selected update
has all earlier propositions in its one complete block. No block was sampled
from a multi-block update.

The runner always calls a model once per block. It then unions all separately
parsed block predictions by `update_id` before computing scenario-level exact
patch accuracy. This is generic and remains correct if a later evaluation view
contains multi-block updates.

## Preregistered reliability threshold

This threshold was frozen before any DialAM prompt-ceiling model call. A
combination passes only if all conditions hold:

| Metric | Threshold |
|---|---:|
| Scenarios | at least 30 |
| JSON validity | 100% |
| Schema validity | 100% |
| Invalid IDs | 0 |
| Exact patch accuracy | at least 80% |
| Edge F1 | at least 85% |
| Relation macro-F1 | at least 75% |
| Direction accuracy | at least 95% |
| False edges/update | at most 0.20 |
| Judge validity | 100% |
| Mean Spec adherence | at least 3.5/4 |
| Mean Robustness | at least 3.5/4 |

The prompt-ceiling gate is **PASS** only when the complete six-cell matrix has
valid judge outputs and no Mini/Haiku prompt combination clears every threshold.
If either hosted-model baseline clears the bar under any prompt, the gate is
**FAIL**. An incomplete matrix also fails closed.

## Commands

```bash
uv run python scripts/run_dialam_prompt_ceiling.py
uv run python scripts/run_dialam_prompt_ceiling.py --preflight \
  --approval APPROVE_DIALAM_PREFLIGHT
uv run python scripts/run_dialam_prompt_ceiling.py --preflight \
  --preflight-provider anthropic \
  --approval APPROVE_DIALAM_PREFLIGHT
uv run python scripts/run_dialam_prompt_ceiling.py \
  --approval APPROVE_DIALAM_PROMPT_CEILING
```

The first command performs no network calls. The approved command writes a
timestamped ignored run directory containing `manifest.json`, `records.jsonl`,
`metrics.json`, and every raw candidate/judge transcript.

The preflight command makes exactly six candidate calls and six blinded-judge
calls: one shared frozen diagnostic block for every model × prompt cell. It
persists response envelopes or structured failure records for every attempt.
The provider-scoped form makes three candidate calls and three blinded-judge
calls and exists only for targeted recovery of a failed provider preflight.

## Superseded exploratory preflight (excluded from the formal matrix)

- The initial six-cell run made 6 candidate and 6 judge calls. All three
  `gpt-5.6-sol` candidates and all six `gpt-5.6-sol` judges resolved. The stale
  Anthropic ID `claude-opus-4-1-20250805` returned three persisted HTTP 404
  failures.
- The configured Anthropic model catalog identified `claude-opus-5`. A scoped
  retry made 3 candidate and 3 judge calls; all resolved to the exact requested
  IDs, all candidate responses were schema-valid, and all judge responses
  passed the judge schema.
- A literal scan of all nine persisted judge prompts found no provider name,
  model ID, or prompt-configuration name.
- These stronger-model calls are not part of the formal Mini/Haiku ablation and
  will not be reused in its metrics. Nominal exploratory accounting was 6
  candidate + 6 judge calls. Correcting
  the stale Anthropic ID brought the actual one-time total to 9 + 9 attempts;
  the failed artifacts were retained rather than overwritten.

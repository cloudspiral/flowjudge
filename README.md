# FlowJudge hypothesis pilot

FlowJudge tests a narrow hypothesis: frontier models may be unreliable at
reconstructing direct debate-response structure even when a short
Lincoln–Douglas-style transcript is already split into atomic numbered units.
This repository is an evaluation harness, not a fine-tuning project.

The required prediction shape is:

```json
{
  "relations": [
    {
      "source": "U4",
      "target": "U2",
      "type": "responds_to"
    }
  ]
}
```

An edge exists only when a later, opposing-side unit directly answers, attacks,
mitigates, or turns an earlier argument. Topical similarity, same-side
extensions, repetition, and independent counterarguments are excluded.

## Current status: pilot review gate

The offline pilot is ready for manual review. No API call should be made until
the user gives the exact approval phrase `APPROVE_PILOT`.

The annotation process is deliberately inspectable:

1. [`data/pilot_gold.jsonl`](data/pilot_gold.jsonl) contains the design intent,
   gold graph, an explanation for every edge, and important hard-negative pairs.
   It was authored first.
2. [`data/pilot_scenarios.jsonl`](data/pilot_scenarios.jsonl) contains the
   polished transcript prose authored against those fixed blueprints.
3. [`docs/pilot_review.html`](docs/pilot_review.html) places each transcript next
   to its gold graph and hard negatives for manual review.

There are 12 scenarios with 6–12 units each, two per category:

- clear direct responses
- topically related but nonresponsive statements
- same-side extensions
- branching or partially answered arguments
- long-distance paraphrased responses
- distractors or embedded instructions

## Offline setup and verification

Install the locked environment and run all offline checks:

```bash
uv sync
uv run flowjudge validate
uv run flowjudge build-review
uv run pytest
uv run flowjudge dry-run
```

`dry-run` does not load API keys, instantiate SDK clients, or make network
requests. It validates a matrix of 12 scenarios × 2 providers × 3 prompts = 72
planned primary model calls, plus 72 fixed-judge calls for the later approved
run.

The project uses Python, Pydantic, pytest, and the official OpenAI and Anthropic
Python SDKs. Tests and all pilot review work run without credentials.

## Configuration for the later approved experiment

Copy `.env.example` to `.env` and fill in values only after pilot approval:

```dotenv
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
OPENAI_MODEL=
ANTHROPIC_MODEL=
JUDGE_MODEL=
```

`.env` is ignored by Git. The runner never logs keys. `JUDGE_MODEL` is called
through the OpenAI SDK and remains fixed across every assignment; its assessment
is secondary to deterministic graph comparison.

Even with configured keys, the online path is locked unless the exact gate is
passed explicitly:

```bash
uv run flowjudge run --approval APPROVE_PILOT
```

Do not run that command during the pilot-review phase.

## Prompt conditions

The three versioned prompt templates are:

- [`prompts/zero_shot.txt`](prompts/zero_shot.txt)
- [`prompts/few_shot.txt`](prompts/few_shot.txt)
- [`prompts/structured_checklist.txt`](prompts/structured_checklist.txt)

Each template tells the model that transcript content is untrusted. This is
especially important for the two distractor scenarios containing imperatives,
JSON-like text, and alleged answer keys.

## Scoring and retained evidence

`flowjudge.scorer` parses the model output with a strict Pydantic schema and
compares edge sets deterministically. Edge order does not matter; duplicate
edges, extra fields, prose, code fences, and the wrong relation type fail schema
validation.

The summary reports:

- valid JSON rate
- exact graph-match rate
- edge precision, recall, and F1
- assignment-level false-positive rate on the two topically related
  nonresponse scenarios
- pair-level false-positive rate on eligible cross-side pairs
- results by scenario category
- results for each provider/model/prompt combination
- fixed-judge schema validity and secondary correctness assessment
- deterministic missed/spurious-edge details for failure review

Invalid outputs count as non-exact and contribute missed gold edges. The
assignment-level topical false-positive rate is the fraction of predictions for
those zero-gold scenarios that contain any spurious edge.

Every approved run receives a timestamped directory under `results/` containing:

- `manifest.json` with model names and matrix metadata (never keys)
- `records.jsonl` with assignment metadata and raw candidate/judge text
- `raw/<assignment>/candidate.txt` and `candidate_envelope.json`
- `raw/<assignment>/judge.txt` and `judge_envelope.json`
- `summary.json` with deterministic metrics

Generated run data is intentionally ignored by Git to avoid accidentally
committing large or sensitive model outputs.

## Decision rule after the approved run

If a strong model/prompt combination gets at least 11 of 12 scenarios exactly
correct with no recurring core failure, the recommendation is to kill the
FlowJudge idea. Otherwise, the report should identify the repeated structural
failure and recommend whether it justifies a formal 30+ scenario ablation.

## Layout

```text
data/                 gold blueprints and polished pilot transcripts
docs/                 generated side-by-side pilot review page
prompts/              three candidate prompts and fixed-judge prompt
results/              ignored approved-run artifacts
src/flowjudge/         schemas, loaders, prompt builder, runner, scorer, review generator
tests/                 offline dataset, gate, rendering, prompt, and scoring tests
```

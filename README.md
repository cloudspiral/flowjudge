# FlowJudge real-data hypothesis benchmark

FlowJudge tests whether small, inexpensive general-purpose models reliably
reconstruct direct debate-response structure from chronologically ordered
argumentative discourse units (ADUs). The immediate hypothesis is whether a
future specialized SLM could outperform these weak baselines. This repository
is an evaluation harness, not a fine-tuning project.

The required prediction shape remains:

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

## Current status: revised benchmark review gate

No model API calls have been made. The benchmark now contains:

- one curated 6–12 ADU excerpt from each of Debates 1–10 in
  [VivesDebate version 3](https://doi.org/10.5281/zenodo.6531487), for 70 real
  model-facing ADUs total;
- readable curated English produced after source IDs and gold pairs were fixed;
- original FAVOUR/AGAINST stance, Catalan/Spanish/raw-English text, phase,
  argument number, all 2,842 valid RA/CA/MA relations, and complete jury
  scores/outcomes preserved as provenance;
- one malformed source relation slot preserved verbatim as a structured issue,
  never repaired or guessed;
- two concise synthetic cases covering a dropped argument and a cross-applied
  rebuttal.

The old 12-scenario synthetic pilot has been removed. The compact
[`docs/pilot_review.html`](docs/pilot_review.html) now reports mapping rules,
automated checks, per-debate counts, known source issues, and two complete
representative excerpts. It does not require manual review of all cases.

## Source, license, and provenance

VivesDebate is described by Ruiz-Dolz et al. (2021):
[paper](https://doi.org/10.3390/app11157160),
[official version 3 data](https://doi.org/10.5281/zenodo.6531487). The source dataset is
licensed CC BY-NC-SA 4.0.

The selected CSVs are retained unchanged under
[`data/source/vivesdebate/`](data/source/vivesdebate/). Its `manifest.json`
records the release DOI, retrieval date, selection rule, license, and official
MD5 checksums. The converted benchmark data derived from VivesDebate is subject
to that dataset license.

The record DOI pins version 3 explicitly. Debates 1–10 all have jury outcomes
and complete stance annotations. Excerpt IDs and gold graphs are recorded in a
separate blueprint file before the curated English text; no selection is based
on model behavior.

## Explicit conversion mapping

The full documented mapping is in
[`docs/vivesdebate_mapping.md`](docs/vivesdebate_mapping.md). In brief:

- chronological source ID `N` becomes `UN`; source ID gaps remain gaps;
- `FAVOUR → AFF` and `AGAINST → NEG`;
- `unit.text` is curated English from the clearer `ADU_ES` and `ADU_CAT`
  fields, while all three raw text fields remain attached to the unit;
- every real scenario contains 6–12 selected source ADUs with original IDs,
  order, stance, phase, and argument number;
- every prompt includes the excerpt's human-written topic label so an atomic
  ADU is not presented without the subject of its exchange;
- all valid `RA`, `CA`, and `MA` annotations remain in `source_relations` as
  inference, conflict, and rephrase, including relations outside the excerpt;
- only an opposite-stance `CA` whose endpoints are both in the excerpt becomes
  `responds_to`;
- for FlowJudge, that conflict is oriented from the later ADU to the earlier
  ADU, while the original VivesDebate direction remains preserved separately;
- same-stance conflicts, inferences, and rephrases do not become response edges;
- both jury score records, component scores, winner, and score margin are
  retained.

The curated blueprint must exactly match all internal opposite-stance `CA`
relations or generation fails. It also explains at least one important hard
negative in every excerpt. This remains conservative: it does not invent
unannotated responses.

## Offline setup and verification

```bash
uv sync
uv run python scripts/build_benchmark.py
uv run flowjudge validate
uv run flowjudge build-review
uv run pytest
uv run flowjudge dry-run
```

The converter verifies source checksums and deterministically regenerates:

- `data/benchmark_scenarios.jsonl`
- `data/benchmark_gold.jsonl`
- `data/benchmark_manifest.json`

`dry-run` does not load API keys, instantiate SDK clients, or make network
requests. The experiment matrix remains 12 scenarios × 2 providers × 3 prompts
= 72 planned primary model calls, followed by 72 fixed-judge calls.

## Configuration for a later approved experiment

Copy `.env.example` to `.env` and fill values only after benchmark approval:

```dotenv
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
OPENAI_MODEL=
ANTHROPIC_MODEL=
JUDGE_MODEL=
```

`.env` is ignored by Git. The runner never logs keys. `JUDGE_MODEL` is called
through the OpenAI SDK and remains fixed across assignments; deterministic graph
comparison is primary.

Even with configured keys, online execution requires the exact gate:

```bash
uv run flowjudge run --approval APPROVE_PILOT
```

Do not run that command during benchmark review.

## Prompts and scoring

Candidate prompts are versioned in `prompts/`:

- zero-shot
- few-shot
- structured checklist

The strict Pydantic output schema rejects duplicate edges, extra fields, prose,
code fences, and invalid relation types. Deterministic scoring reports:

- valid JSON rate;
- exact graph-match rate;
- edge precision, recall, and F1;
- false-positive rate on explicitly annotated hard-negative pairs;
- results by source category and provider/model/prompt combination;
- fixed-judge schema validity and secondary correctness;
- deterministic missed/spurious-edge details for failure analysis.

Every approved run preserves its manifest, assignment records, candidate and
judge text, complete SDK response envelopes, and summary under `results/`.

## Decision rule

If even a weak model/prompt combination gets at least 11 of 12 scenarios exactly
correct with no recurring core failure, recommend killing the FlowJudge idea.
Otherwise, identify the repeated structural failure and decide whether it
justifies training an SLM and expanding to a formal 30+ scenario ablation. This
pilot supports a claim about beating inexpensive general-purpose baselines, not
a claim that flagship frontier models are unreliable.

## Layout

```text
data/benchmark_*.json*       converted benchmark and validation manifest
data/curation/               pre-text excerpt blueprints and curated English
data/source/vivesdebate/     unchanged selected source CSVs and provenance
docs/                        compact review and mapping documentation
prompts/                     candidate and fixed-judge prompts
results/                     ignored approved-run artifacts
scripts/build_benchmark.py   deterministic source converter
src/flowjudge/               schemas, loader, runner, scorer, review generator
tests/                       offline provenance, conversion, gate, and scoring tests
```

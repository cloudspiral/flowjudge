# FlowJudge real-data hypothesis benchmark

FlowJudge tests whether frontier models reliably reconstruct direct
debate-response structure from chronologically ordered argumentative discourse
units (ADUs). This repository is an evaluation harness, not a fine-tuning
project.

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

- 10 complete real debates from
  [VivesDebate version 3](https://doi.org/10.5281/zenodo.6531487): Debates 1–10;
- all 2,932 source ADUs in chronological order;
- original FAVOUR/AGAINST stance, Catalan/Spanish/English text, phase, argument
  number, all valid RA/CA/MA relations, and complete jury scores/outcomes;
- one malformed source relation slot preserved verbatim as a structured issue,
  never repaired or guessed;
- two concise synthetic cases covering a dropped argument and a cross-applied
  rebuttal.

The old 12-scenario synthetic pilot has been removed. The compact
[`docs/pilot_review.html`](docs/pilot_review.html) now reports mapping rules,
automated checks, per-debate counts, known source issues, and two representative
conversions. It does not require manual review of all cases.

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
and complete stance annotations; no debate selection is based on model behavior
or gold-edge density.

## Explicit conversion mapping

The full documented mapping is in
[`docs/vivesdebate_mapping.md`](docs/vivesdebate_mapping.md). In brief:

- chronological source ID `N` becomes `UN`; source ID gaps remain gaps;
- `FAVOUR → AFF` and `AGAINST → NEG`;
- `ADU_EN` is the model-facing text, while `ADU_CAT` and `ADU_ES` remain on the
  unit;
- all valid `RA`, `CA`, and `MA` annotations remain in `source_relations` as
  inference, conflict, and rephrase;
- only an opposite-stance `CA` conflict becomes `responds_to`;
- for FlowJudge, that conflict is oriented from the later ADU to the earlier
  ADU, while the original VivesDebate direction remains preserved separately;
- same-stance conflicts, inferences, and rephrases do not become response edges;
- both jury score records, component scores, winner, and score margin are
  retained.

This is intentionally conservative: it derives gold only from explicit corpus
conflict annotations and does not invent unannotated responses.

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

If a strong model/prompt combination gets at least 11 of 12 scenarios exactly
correct with no recurring core failure, recommend killing the FlowJudge idea.
Otherwise, identify the repeated structural failure and decide whether it
justifies a formal 30+ scenario ablation.

## Layout

```text
data/benchmark_*.json*       converted benchmark and validation manifest
data/source/vivesdebate/     unchanged selected source CSVs and provenance
docs/                        compact review and mapping documentation
prompts/                     candidate and fixed-judge prompts
results/                     ignored approved-run artifacts
scripts/build_benchmark.py   deterministic source converter
src/flowjudge/               schemas, loader, runner, scorer, review generator
tests/                       offline provenance, conversion, gate, and scoring tests
```

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

## Current status: formal ablation ready for approval

The approved pilot completed all 72 candidate assignments and 72 fixed-judge
assessments. The primary exact graph-match rate was 2/72 (2.8%); no
model/prompt combination exceeded 1/12. The full analysis and expansion
recommendation are in [`docs/pilot_results.md`](docs/pilot_results.md). Every raw
candidate response, judge response, and SDK envelope remains in the ignored
local run directory `results/20260821T012454.328398Z/`.

The pilot justified expansion, so the now-locked formal benchmark contains:

- three curated 6–12 ADU excerpts from each of Debates 1–10 in
  [VivesDebate version 3](https://doi.org/10.5281/zenodo.6531487), for 30 real
  scenarios and 210 real model-facing ADUs total;
- a locked 24-case development split and an 8-case held-out test split; all
  held-out cases are previously unused real excerpts;
- scenario-level tags for local and long-distance responses, branching,
  response chains, rephrase traps, same-side-extension traps, and topical
  distractors;
- readable curated English produced after source IDs and gold pairs were fixed;
- original FAVOUR/AGAINST stance, Catalan/Spanish/raw-English text, phase,
  argument number, all 2,842 valid RA/CA/MA relations, and complete jury
  scores/outcomes preserved as provenance;
- one malformed source relation slot preserved verbatim as a structured issue,
  never repaired or guessed;
- two concise synthetic cases covering a dropped argument and a cross-applied
  rebuttal.

The compact
[`docs/pilot_review.html`](docs/pilot_review.html) now reports mapping rules,
automated checks, per-excerpt counts, the split, known source issues, and two complete
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
and complete stance annotations. Excerpt IDs, splits, phenomenon tags, and gold
graphs are recorded in a separate blueprint file before the curated English
text. The original ten pilot excerpts remain development data; the added
excerpts were selected from source annotations rather than model behavior.
This split is for the prompt-only ablation. It is not a leakage-safe future
fine-tuning split because development and held-out excerpts can come from the
same debate; any later SLM phase must create a debate-level training/test split.

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
requests. The formal matrix is 32 scenarios × 2 providers × 3 prompts = 192
planned primary model calls, followed by 192 fixed-judge calls.

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
uv run flowjudge run --approval APPROVE_ABLATION
```

Do not run that command during benchmark review.

## Prompts and scoring

Candidate prompts are versioned in `prompts/`:

- zero-shot
- few-shot
- structured checklist

The strict Pydantic output schema rejects duplicate edges, extra fields, prose,
code fences, and invalid relation types. A second deterministic view removes
only one complete Markdown code fence; it does not repair prose, malformed JSON,
field names, or graph content. Scoring reports:

- valid JSON rate;
- strict and fence-normalized exact graph-match rate;
- strict and fence-normalized edge precision, recall, and F1;
- false-positive rates on explicitly typed hard-negative pairs, including
  topically related nonresponses;
- results by source category, development/held-out split, structural phenomenon,
  and provider/model/prompt combination;
- fixed-judge schema validity and secondary correctness;
- deterministic missed/spurious-edge details for failure analysis.

Every approved run preserves its manifest, assignment records, candidate and
judge text, complete SDK response envelopes, and summary under `results/`.

## Decision rule

The pilot kill condition was not met. In the formal run, treat the held-out
split and fence-normalized graph metrics as the primary reasoning diagnostic,
while reporting strict compliance separately. Repeated structural errors must
be identified by phenomenon before any SLM training is justified. This
benchmark can support a claim about outperforming the chosen inexpensive
general-purpose baselines; it does not establish that flagship frontier models
are unreliable unless flagship models are separately tested.

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

# FlowJudge response-graph specialization experiment

FlowJudge tests whether small, inexpensive general-purpose models reliably
reconstruct direct debate-response structure from chronologically ordered
argumentative discourse units (ADUs). The prompt-ceiling gate is complete; the
repository now includes the leakage-safe data-generation, QLoRA, and
base-versus-tuned evaluation pipeline for the specialized SLM phase.

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

The locked, falsifiable behavior and fixed 0–4 judge rubric are in
[`docs/behavior_spec.md`](docs/behavior_spec.md).

## DialAM incremental-patch feasibility gate

The repository now also contains a data/evaluation gate for the narrower
incremental behavior in [`BEHAVIOR_SPEC.md`](BEHAVIOR_SPEC.md): given one new
proposition and a complete fixed-size block of earlier propositions, emit the
direct `SUPPORT`, `ATTACK`, and `REPHRASE` patch as one bare JSON object.

The official DialAM-2024 QT30 archive is checksum-pinned, downloaded on demand,
and kept out of Git. The canonical parser preserves episode/map identity,
chronology, speaker and raw locution grounding, proposition text, original
RA/CA/MA labels and direction, normalized labels, and n-ary structure. The
smoke data uses only singly grounded, unambiguous binary direct relations and a
24-dialogue train / 6-dialogue held-out split. No semantic retrieval is used;
the gate itself remains frozen independently of the later QLoRA phase.

The post-audit correction confirms that splits use original parent episodes,
not individual map IDs. Identity-based deduplication leaves 12,767 unique
RA/CA/MA relations; it does not reproduce the published 10,818 headline. The
full retention funnel, grounding/chronology defects, exact exclusion policy,
and conservative publication matrix are in the audit.

```bash
uv sync --frozen
uv run python scripts/build_dialam_gate.py
uv run python scripts/validate_dialam_gate.py
uv run python scripts/run_dialam_prompt_ceiling.py
uv run pytest
```

The last command above is an offline dry-run: 30 balanced diagnostic scenarios
× 2 cross-family hosted-model baselines × 3 prompts = 180 candidate calls and
180 fixed-judge calls. The approved Mini/Haiku matrix completed and the
corrected prompt-ceiling gate passed because no cell cleared every reliability
threshold. See [`docs/dialam_prompt_ceiling_results.md`](docs/dialam_prompt_ceiling_results.md)
for the six-cell table and surviving false-edge failure.

The small-model phase now uses only `Qwen/Qwen3-0.6B` and one fixed Unsloth
QLoRA configuration. Nested 256/512/1024/2048 slices are deterministic prefixes
of the same leakage-safe corpus. The n=256 smoke checkpoint trained, saved,
reloaded, and completed the unchanged frozen evaluation. It raised schema
validity from 0% to 100%, but edge F1 is only 9.5%; see
[`docs/dialam_qlora_smoke_results.md`](docs/dialam_qlora_smoke_results.md).

The fixed v1 curve is complete. Its best point, n=2048, reached 8/30 exact
patches, 16.7% edge F1, and 0.667 false edges/update. A controlled v2 n=2048
hard-negative run raised edge F1 to 21.4% but worsened false edges to
0.867/update and was rejected by its preregistered gate.

The preregistered v3 correction then succeeded. It uses 4,096 rows as 2,048
exact same-update positive/NONE pairs and averages assistant-token loss within
each row before averaging rows, removing the 4.50× output-length weighting
imbalance. On the unchanged frozen set, v3 reaches 13/30 exact patches, 35.0%
edge F1, 33.9% macro-F1, 0.300 false edges/update, 0/6 NONE false-positive
cases, and 2.97/4 judge Robustness. V3 cleared every material-improvement
condition and became the first selected direction.

The later v5/v5.1 pipeline improves further. V5 trains 8,192 candidate-level
rows as exact positive/NONE pairs and replaces unconstrained JSON generation
with fixed four-label likelihood scoring. Raw v5 exposed the expected 50.0%
training-versus-7.379%-natural prior shift; the preregistered v5.1 3.0 NONE
margin corrected it without retraining. On the same frozen set, v5.1 reaches
16/30 exact patches (53.3%), 47.6% edge F1, 47.4% macro-F1, 0.267 false
edges/update, 0/6 NONE false-positive cases, and 3.03/4 judge Robustness. It
passes every frozen promotion check and is now selected, while still failing
the original high reliability bar. The selected adapter is published at
[`mr-mc/flowjudge-dialam-qwen3-0.6b-v5-1-n8192`](https://huggingface.co/mr-mc/flowjudge-dialam-qwen3-0.6b-v5-1-n8192),
the permission-cleared transformed dataset is at
[`mr-mc/flowjudge-dialam`](https://huggingface.co/datasets/mr-mc/flowjudge-dialam),
and the live base-versus-tuned demo remains
[`mr-mc/flowjudge-dialam-demo`](https://huggingface.co/spaces/mr-mc/flowjudge-dialam-demo).
See [`docs/dialam_v1_efficiency_results.md`](docs/dialam_v1_efficiency_results.md),
[`docs/dialam_v2_hard_negative_results.md`](docs/dialam_v2_hard_negative_results.md),
[`docs/dialam_v3_results.md`](docs/dialam_v3_results.md), and
[`docs/dialam_v5_1_results.md`](docs/dialam_v5_1_results.md).

The versioned [`DialAM experiment history`](docs/dialam_experiment_history.md)
preserves every base/v1/v2/v3 milestone and every later development gate, with
a generated two-panel chart that never mixes development-only evidence with
the frozen benchmark. Regenerate the ledger, Markdown table, and SVG with
`python scripts/build_dialam_experiment_history.py`.

A v4 experiment doubled the corpus to 8,192 rows and balanced
SUPPORT, ATTACK, and REPHRASE exposure while retaining the complete v3
foundation. On the episode-disjoint development set it improved edge F1 from
21.1% to 34.1% and ATTACK F1 from 0% to 54.5%, but NONE cases with a false edge
worsened from 2/6 to 4/6. V4 therefore failed its preregistered development
gate; the reused frozen set and judge were not run. Its failed point is retained
in the experiment ledger rather than hidden.
See [`docs/dialam_v4_results.md`](docs/dialam_v4_results.md).

The assignment-prescribed base-versus-tuned evaluator now auto-detects DialAM
`PatchExample` JSONL and writes the complete deterministic table plus blinded
judge transcripts:

```bash
uv sync --group train
uv run python eval.py \
  --model mr-mc/flowjudge-dialam-qwen3-0.6b-v5-1-n8192 \
  --eval-set <dialam-patch-example-jsonl>
```

The official raw archive/maps remain Git-ignored. Under the project owner's
explicit project-specific redistribution permission, the vetted
[`hf_dataset/dialam_patch/`](hf_dataset/dialam_patch/) package contains the
actual transformed training/evaluation JSONL, candidate/judge evidence,
metadata, schemas, hashes, and reconstruction scripts. No general QT30 license
is claimed and the package does not mirror the official raw source.

## Current status: local end-to-end loop completed

The approved pilot completed all 72 candidate assignments and 72 fixed-judge
assessments. The primary exact graph-match rate was 2/72 (2.8%); no
model/prompt combination exceeded 1/12. The full analysis and expansion
recommendation are in [`docs/pilot_results.md`](docs/pilot_results.md). Every raw
candidate response, judge response, and SDK envelope remains in the ignored
local run directory `results/20260821T012454.328398Z/`.

The approved formal ablation then completed all 192 candidate assignments and
192 fixed-judge assessments. Strict exact match was 4/192 (2.1%), while exact
match after removing only a surrounding Markdown fence was 24/192 (12.5%). The
best combination reached 8/32 overall and 2/8 held-out. Full results, recurring
failure analysis, and the gold-validity warning are in
[`docs/ablation_results.md`](docs/ablation_results.md). Raw artifacts remain in
the ignored local directory `results/20260821T020430.050266Z/`.

The assignment prompt-ceiling rerun completed another 192 candidate calls and
192 fixed-judge calls with GPT-5.4 Mini and Claude Haiku 4.5. The best cell
reached only 58.6% mean Spec adherence, 69.5% mean Robustness, and 15.6%
deterministic exact graph match, so the behavior survives the pre-registered
95%/90% gate. See [`docs/prompt_ceiling_results.md`](docs/prompt_ceiling_results.md).
The recurring best-cell failure is spurious response edges, especially on
same-side extensions.

The first real distillation run then accepted 113/115 VivesDebate-derived
examples and published leak-free nested 12/24/48/96 training slices. All four
Qwen3 0.6B QLoRA checkpoints trained locally with complete logs. On the
nine-case own evaluation set, the best n=96 checkpoint improved valid JSON from
66.7% to 100% and edge F1 from 0% to 16%, but exact graph match remained 0%.
Two data-only v2 revisions and a Qwen3 1.7B capacity check failed to improve the
result. See [`docs/data_generation_results.md`](docs/data_generation_results.md),
[`docs/data_efficiency_results.md`](docs/data_efficiency_results.md), and
[`docs/v2_results.md`](docs/v2_results.md).

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

## Frontier prompt-ceiling rerun

The assignment-critical run used 32 scenarios × 2 frontier providers × 3
prompts, with the fixed judge returning explicit Spec-adherence and Robustness
scores. To match the pragmatic low-cost interpretation used by the companion
Minesweeper assignment, the selected candidates were
`gpt-5.4-mini-2026-03-17` and `claude-haiku-4-5-20251001`. The fixed judge
was the stronger `gpt-5.6-sol` so measurement quality was not intentionally
weakened with the candidates.

Copy `.env.example` to `.env` and add keys if they are not already present:

```dotenv
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
OPENAI_MODEL=gpt-5.4-mini-2026-03-17
ANTHROPIC_MODEL=claude-haiku-4-5-20251001
JUDGE_MODEL=gpt-5.6-sol
```

`.env` is ignored by Git. The runner never logs keys. `JUDGE_MODEL` is called
through the OpenAI SDK and remains fixed across assignments; deterministic graph
comparison is primary.

The exact rerun command is:

```bash
uv run flowjudge run \
  --approval APPROVE_FRONTIER_ABLATION \
  --openai-model gpt-5.4-mini-2026-03-17 \
  --anthropic-model claude-haiku-4-5-20251001 \
  --judge-model gpt-5.6-sol
```

Do not rerun that paid command without the exact approval phrase. Its raw
artifacts are preserved locally in `results/20260821T055241.543347Z/`.

## Distillation, training, and evaluation

Training candidates come only from complete Debates 1–7; the own evaluation
set comes only from Debates 8–10. A fixed strong teacher rewrites the noisy
source-derived excerpts, and a separate strict call checks meaning, gold-edge
support, absence of label cues, and contextual clarity. Dataset sizes 12, 24,
48, and 96 are nested and deterministic.

The canonical open base is `Qwen/Qwen3-0.6B`. On Apple Silicon the project uses
the corresponding `mlx-community/Qwen3-0.6B-4bit` checkpoint and MLX-LM LoRA;
training adapters over the quantized base is QLoRA. The CUDA path uses
Transformers, PEFT, and bitsandbytes NF4. Both paths mask prompt tokens.

```bash
uv run flowjudge build-training-data --limit 115
uv run flowjudge distill-training-data --max-workers 4 --required-examples 96
uv sync --frozen --group mlx-train
uv run --group mlx-train python scripts/run_efficiency_curve.py --backend mlx
uv run python eval.py \
  --model artifacts/efficiency/n96/adapter \
  --eval-set data/eval/own_eval.jsonl
```

`eval.py` auto-detects local MLX-LM or PEFT adapters, evaluates the base and
tuned model, and preserves raw generations plus one full judge JSON object per
example. A grader can replace the eval path with staff-provided `BenchmarkCase`
JSONL without changing the harness.

The best n=96 adapter has also been fused and dequantized into a portable local
Transformers checkpoint at `artifacts/publish/flowjudge-qwen3-0.6b/`. A real
CPU load and generation verified all 596,049,920 parameters. Public upload is
prepared by `scripts/publish_hf.py` but requires Hugging Face authentication and
user-selected public repository IDs.

The OpenAI account exhausted its credit after completing rubric judgments for
n=12 and n=24. Later local generations, pending-judge rows, and deterministic
metrics are preserved without substituting a weaker judge. Replenish the same
key to complete the fixed-judge columns.

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
- mean fixed-judge Spec adherence and Robustness on the documented 0–4 rubric;
- deterministic missed/spurious-edge details for failure analysis.

Every approved run preserves its manifest, assignment records, candidate and
judge text, complete SDK response envelopes, and summary under `results/`.

## Decision rule

The prompt-ceiling kill condition was not met, but the SLM also did not learn
the behavior reliably. The data produced strong schema compliance and a small
nonzero graph-reconstruction gain, not superiority to the chosen weak frontier
baselines. No tested dataset size reliably holds the behavior, so minimum viable
N is not established at N ≤ 120. The next evidence-driven step is gold-label
adjudication and more diverse human-checked attachment contrasts, not
hyperparameter tuning.

## Layout

```text
data/benchmark_*.json*       converted benchmark and validation manifest
data/curation/               pre-text excerpt blueprints and curated English
data/source/vivesdebate/     unchanged selected source CSVs and provenance
data/training/               filtered v1 slices and error-driven v2 datasets
docs/                        compact review and mapping documentation
prompts/                     candidate and fixed-judge prompts
results/                     ignored approved-run artifacts
scripts/                     conversion, QLoRA, evaluation, and publication tools
src/flowjudge/               schemas, loader, runner, scorer, review generator
tests/                       offline provenance, conversion, gate, and scoring tests
```

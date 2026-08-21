# VivesDebate to FlowJudge mapping

## Source and selection

The real-data benchmark is derived from the official
[VivesDebate version 3 Zenodo release](https://doi.org/10.5281/zenodo.6531487), described
in [Ruiz-Dolz et al. (2021)](https://doi.org/10.3390/app11157160). The corpus
contains competitive debates on the resolution “Should surrogacy be
legalised?” The source license is CC BY-NC-SA 4.0.

FlowJudge uses Debates 1–10 from the pinned version 3 release. The unchanged
debate CSVs and jury-evaluation CSV remain under `data/source/vivesdebate/`;
their official MD5 checksums are recorded in the source manifest.

The model-facing benchmark does not send an entire 198–371 ADU debate to a
model. It uses three self-contained 6–12 ADU exchanges from each debate and gives
the model a short human-written topic label. This restores the short,
manually reviewable task shape and prevents the evaluation from becoming
primarily a test of poor machine translation and very-long-context search.

## Curation order and audit trail

The curation is split into two files so that relation labels are fixed before
the model-facing wording:

1. `data/curation/vives_excerpt_blueprints.json` fixes each excerpt's source ADU
   IDs, development or held-out split, structural phenomenon tags, CA-derived
   gold pairs, content-specific edge explanations, and typed hard-negative pairs.
2. `data/curation/vives_reviewed_english.json` supplies a curated English
   rendering of those already-selected ADUs using the clearer `ADU_ES` and
   `ADU_CAT` fields.

The converter requires the configured gold pairs to equal all internal,
opposite-stance VivesDebate `CA` relations exactly. Changing the English cannot
add or remove a gold edge. ADU boundaries, source IDs, chronological order, and
stance are unchanged.

## Field mapping

| VivesDebate field | FlowJudge field | Rule |
|---|---|---|
| `ID (Chronological)` | `units[].id` and `units[].source_id` | `N` becomes `UN`. Selected units stay in source order; original gaps remain gaps. |
| `TEAM STANCE=FAVOUR` | `units[].side=AFF` | The original value remains in `source_stance`. |
| `TEAM STANCE=AGAINST` | `units[].side=NEG` | The original value remains in `source_stance`. |
| Curated English | `units[].text` | Model-facing text rendered from `ADU_ES` and `ADU_CAT` without merging or splitting ADUs. |
| `ADU_EN` | `units[].text_en` | Original corpus machine translation, preserved verbatim but not used as model input. |
| `ADU_CAT`, `ADU_ES` | `units[].text_ca`, `units[].text_es` | Preserved verbatim for provenance and review. |
| Excerpt blueprint title | Prompt `Excerpt topic` | Human-written shared context; it does not merge, split, or replace any ADU. |
| `TYPE (Part + Person)` | `units[].phase` | Preserved exactly; blank values remain null. |
| `ARGUMENT NUMBER` | `units[].argument_number` | Preserved exactly; blank values remain null. |
| `RA` | `source_relations[].type=inference` | Preserved but never converted to `responds_to`. |
| `MA` | `source_relations[].type=rephrase` | Preserved but never converted to `responds_to`. |
| `CA` | `source_relations[].type=conflict` | Preserved in original source direction and casing. |
| Jury evaluation rows | `jury_outcome` | Both stance scores and component scores are preserved; winner and margin are derived deterministically. |

Each real scenario retains its debate's complete valid RA/CA/MA graph as
provenance, including relations whose endpoints fall outside the model-facing
excerpt. Only relations with both endpoints inside the excerpt are eligible for
gold derivation. This keeps the source annotations auditable without exposing
out-of-context ADUs to the tested model.

## Gold response derivation

A `responds_to` edge exists only when a valid VivesDebate `CA` relation:

- has both endpoints in the selected excerpt;
- connects opposing stances; and
- matches the prewritten excerpt blueprint.

The later chronological ADU becomes the FlowJudge source and the earlier ADU
becomes the target. Treating `CA` as symmetric is only a temporal-orientation
convention; the original direction remains in `source_relations`.

Same-side conflicts, inference links, rephrases, and relations crossing the
excerpt boundary never become response edges. Each excerpt also contains one or
more explicitly explained hard negatives tagged as topical nonresponse,
same-side extension, repetition/rephrase, or independent counterargument.

## Split and ablation design

The ten excerpts used in the completed pilot stay in the development split. Two
additional excerpts per debate broaden the structural coverage. Eight new real
excerpts—one each from Debates 1–8—form a held-out test split that was not used
in the pilot; the remaining real and synthetic cases are development data. This
produces 24 development scenarios and 8 held-out scenarios without placing a
synthetic case in the held-out score.

This is a held-out split for the prompt-only baseline ablation, not a future
fine-tuning split: excerpts from the same source debate can occur in both
partitions. Before training any SLM, create a debate-level split that prevents
source-debate leakage.

Scenario tags identify the intended structural stressors: local direct replies,
long-distance replies, branching, response chains, rephrase traps, same-side
extension traps, and topical distractors. The two synthetic development cases
add dropped arguments and cross-application because those phenomena are not
cleanly isolated in the selected VivesDebate excerpts.

## Curated-English boundary

The curated English fixes grammar and supplies omitted function words when the
Spanish or Catalan makes them clear. It does not merge ADUs, add warrants, add
relation language, or rewrite a unit to make its gold edge easier. Raw Catalan,
Spanish, and `ADU_EN` remain attached to every selected unit and appear in
expandable sections on the review page.

This is a curation layer, not a claim that the English text is an official
VivesDebate translation. The original CSV is always the authoritative source.

## Malformed source annotations

The selected source files contain one malformed relation slot: Debate7 U25 has
an `RA` type but no related ID. It remains verbatim in
`source_annotation_issues` and creates no relation or gold edge. The converter
does not repair or guess malformed annotations.

## Validation contract

`scripts/build_benchmark.py` fails on checksum drift, duplicate JSON keys,
duplicate or unordered source IDs, missing stance or multilingual text,
incomplete curation coverage, an excerpt outside the 6–12 ADU limit, or any
disagreement between configured gold and internal opposite-stance CA
annotations.

The generated manifest and tests verify:

- all 2,932 source ADUs remain in unchanged checksum-verified CSVs;
- all 2,842 valid source relations and every relation slot remain accounted for;
- all 210 selected real ADUs preserve source ID, order, stance, phase, argument
  number, and raw Catalan/Spanish/English text;
- every real excerpt has 6–12 ADUs and at least one hard negative;
- every gold edge is later-to-earlier, cross-stance, and backed by an internal
  `CA` annotation;
- all jury outcomes are preserved.

`docs/pilot_review.html` shows the mapping, these automated checks, the 30-excerpt
inventory, and two complete representative transcripts. It is a compact audit
report, not a request to manually review all cases.

# VivesDebate to FlowJudge mapping

## Source and selection

The real-data benchmark is derived from the official
[VivesDebate version 3 Zenodo release](https://doi.org/10.5281/zenodo.6531487), described
in [Ruiz-Dolz et al. (2021)](https://doi.org/10.3390/app11157160). The corpus
contains complete competitive debates on the resolution “Should surrogacy be
legalised?” The source license is CC BY-NC-SA 4.0.

FlowJudge selects Debates 1–10 from the pinned version 3 release. All 10 have
jury outcomes and a non-empty stance annotation for every ADU. Pinning the
record DOI, rather than only the concept DOI, prevents silent source drift.

The raw selected CSVs and jury evaluation CSV are retained unchanged under
`data/source/vivesdebate/`. `manifest.json` records official MD5 checksums and
the retrieval date. The converted benchmark uses the corpus-provided `ADU_EN`
machine translation for model input while also preserving `ADU_CAT` and
`ADU_ES` on every unit.

## Field mapping

| VivesDebate field | FlowJudge field | Rule |
|---|---|---|
| `ID (Chronological)` | `units[].id` and `units[].source_id` | `N` becomes `UN`. Original gaps are retained; units remain sorted by source ID. |
| `TEAM STANCE=FAVOUR` | `units[].side=AFF` | Canonical stance is also preserved as `source_stance=FAVOUR`. |
| `TEAM STANCE=AGAINST` | `units[].side=NEG` | Canonical stance is also preserved as `source_stance=AGAINST`. |
| `ADU_EN` | `units[].text` and `units[].text_en` | English model input; text is not rewritten. |
| `ADU_CAT`, `ADU_ES` | `units[].text_ca`, `units[].text_es` | Preserved source language and Spanish machine translation. |
| `TYPE (Part + Person)` | `units[].phase` | Preserved exactly, including source casing; blank source values remain null. |
| `ARGUMENT NUMBER` | `units[].argument_number` | Preserved exactly; blank values remain null. |
| `RA` | `source_relations[].type=inference` | Preserved but never converted to `responds_to`. |
| `MA` | `source_relations[].type=rephrase` | Preserved but never converted to `responds_to`. |
| `CA` | `source_relations[].type=conflict` | Preserved in its original source direction and casing. |
| Jury evaluation rows | `jury_outcome` | Both stance scores and component scores are preserved; winner and margin are derived deterministically. |

Some CSVs have multiple `RELATED ID` / `ARGUMENTAL RELATION TYPE` column pairs.
The converter reads every pair and expands semicolon-separated target IDs into
individual `source_relations` while retaining the originating column and raw
relation label.

## Gold response derivation

FlowJudge derives a `responds_to` edge only when a valid VivesDebate `CA`
relation links ADUs with opposing stances. The later chronological ADU becomes
the FlowJudge source and the earlier ADU becomes the target.

This is a benchmark conversion convention: conflict is treated as symmetric for
the purpose of temporal orientation. The original source direction remains in
`source_relations`. Same-stance conflicts remain preserved but do not become
FlowJudge responses. Inference and rephrase relations also remain preserved but
do not become responses.

This mapping is intentionally conservative. It captures corpus-annotated direct
conflict and does not claim that every possible natural-language answer,
mitigation, or turn was annotated as `CA` by VivesDebate.

## Malformed source annotations

The selected source files contain one malformed relation slot: Debate7 U25 has
an `RA` type but no related ID. This slot is retained verbatim in
`source_annotation_issues` with:

- source ADU;
- source and type column names;
- raw related-ID and relation-type strings;
- a deterministic reason (`incomplete_pair`, `unknown_relation_type`, or
  `unknown_target`).

Malformed slots do not create source relations or gold edges. The converter does
not repair or guess them. The unchanged CSV remains the ultimate source record.

## Validation contract

`scripts/build_benchmark.py` fails on checksum drift, duplicate or unsorted ADU
IDs, missing selected stances, missing text, unrecognised valid relation labels,
or relations to unknown units that are not captured as source issues. The
generated `data/benchmark_manifest.json` reports:

- selected-file checksum verification;
- source and converted ADU counts;
- source relation-slot accounting;
- valid RA/CA/MA counts and malformed-slot counts;
- chronological ID and stance checks;
- jury outcome presence;
- the invariant that every gold edge is later-to-earlier and cross-stance.

The compact `docs/pilot_review.html` presents these checks, the per-debate
inventory, and two representative converted examples. It deliberately does not
ask a reviewer to inspect all 2,932 real ADUs manually.

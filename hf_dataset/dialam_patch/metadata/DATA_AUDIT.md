# DialAM-2024 / QT30 data audit

## Status and scope

**Feasibility gate completed on 2026-08-22.** The project owner reported that
they obtained approval to use QT30 for this educational toy project. That
project-specific permission resolves the use gate but is not represented here
as a general public corpus license.

This audit parsed every one of the 1,478 official DialAM training maps, well
above the requested 20-map minimum. Counts below are reproduced directly from
the pinned archive unless explicitly marked as figures published by the corpus
site.

The smoke task uses one complete argument map as the annotated comparison
scope. Episode identity is retained for leakage-safe splitting, but propositions
from other maps in the same episode are not treated as negative candidates.
That avoids asserting cross-map negatives the source annotation does not make.

## Source, version, permission, and format

- Publisher: ARG-tech, University of Dundee
- Landing page: <http://dialam.arg.tech/>
- Training archive: <http://dialam.arg.tech/res/files/dataset.zip>
- Corpus record: <http://corpora.aifdb.org/qt30>
- Corpus paper: <https://aclanthology.org/2022.lrec-1.352/>
- Retrieval date: 2026-08-22

The download has no semantic version or release tag, so the exact snapshot is
pinned by its transport metadata and digest:

| Field | Value |
|---|---|
| HTTP `Last-Modified` | `Thu, 15 Feb 2024 21:45:17 GMT` |
| HTTP `ETag` | `"4d3421-611728b0e30f9"` |
| Bytes | `5,059,617` |
| SHA-256 | `9d3d70c5ad86c815f821928dc515ee80a8e2a24d3f5d224557045f50f7a73791` |
| JSON maps | `1,478` |

The format is Argument Interchange Format JSON. Every map has `nodes`, `edges`,
and `locutions` arrays. `L` nodes hold raw locutions, `I` nodes reconstructed
propositions, `RA`/`CA`/`MA` nodes propositional relations, `YA` nodes
illocutionary grounding, and `TA` nodes dialogue transitions. Directed edges
are stored as `fromID -> toID`.

Neither the official download page, archive, format guide, annotation guide,
nor AIFdb's `qt30` record states a corpus license. The project therefore records
the owner's reported direct approval as its permission basis without inferring
general redistribution terms. Exact provenance is in
[`data/dialam/source_manifest.json`](data/dialam/source_manifest.json).

Publication policy under the evidence currently available:

| Artifact | Publishability |
|---|---|
| Raw archive or raw dialogue/proposition text | **Not cleared** |
| Transformed JSONL containing corpus text | **Not cleared** |
| IDs and labels only | **Unclear; obtain written redistribution permission** |
| Original parser/build/eval scripts | **Publishable**, if text- and secret-free |
| Aggregate manifests/statistics | **Publishable** with corpus citation |

The text-bearing generated JSONL and raw archive are therefore Git-ignored.
This is a conservative project policy, not legal advice.

## Dialogue and map counts

The AIFdb QT30 record contains 30 episode subcorpora. Their 1,478 map IDs align
with the archive except for one documented source mismatch: AIFdb currently
lists absent `nodeset19137`, while the archive contains `nodeset19149` with the
matching 2020-11-05 episode content. The explicit replacement is recorded in
[`data/dialam/dialogues.json`](data/dialam/dialogues.json).

| Dialogue ID | Maps | Dialogue ID | Maps |
|---|---:|---|---:|
| `cutietestrun28May2020` | 37 | `cutiestestrun14January2021` | 40 |
| `cutietestrun4June2020` | 47 | `cutiestestrun28january2021` | 43 |
| `cutietestrun18June2020` | 46 | `cutiestestrun18february2021` | 45 |
| `cutietestrun30July2020` | 50 | `cutiestestrun4march2021` | 56 |
| `cutietestrun2September2020` | 50 | `cutiestestrun18march2021` | 47 |
| `cutietestrun22October2020` | 44 | `cutiestestrun15april2021` | 74 |
| `cutiestestrun5November2020` | 42 | `cutiestestrun29april2021` | 85 |
| `cutiestestrun19November2020` | 36 | `cutietestrun20may2021` | 60 |
| `cutiestestrun10December2020` | 45 | `cutietestrun27may2021` | 58 |
| `cutietestrun10June2021` | 56 | `qt11112021wtxt` | 42 |
| `cutietestrun24June2021` | 48 | `qt28102021wtxt` | 59 |
| `cutietestrun8july2021` | 41 | `qt14102021wtxt` | 40 |
| `qt30092021wtxt` | 59 | `qt16092021wtxt` | 48 |
| `qt02092021wtxt` | 44 | `qt19082021wtxt` | 42 |
| `qt05082021wtxt` | 45 | `qt22072021wtxt` | 49 |

## Locutions, propositions, chronology, and grounding

The archive contains overlapping map context and duplicated `locutions`
metadata, so occurrence counts and unique node counts differ:

| Structure | Occurrences | Unique node IDs |
|---|---:|---:|
| Raw `L` nodes | 49,550 | 45,166 |
| `I` propositions | 23,465 | 20,451 |
| `YA` nodes | 34,430 | 34,430 |
| `TA` nodes | 23,038 | 23,038 |
| Raw locution metadata records | 39,496 | 21,136 |
| Valid `L` IDs in locution metadata | — | 21,131 |

There are five locution metadata records pointing to non-`L` nodes. In 535
maps, repeated locution IDs are deduplicated deterministically; their speaker
and start time agree, though some annotation timestamps differ.

Chronological turn is reconstructed within each map from the directed
`L -> TA -> L` transition graph, using source node order only as a deterministic
tie-break. Thirty maps contain cyclic or unresolved TA precedence and are kept
in the canonical audit but excluded from smoke-data selection.

Direct locution-to-proposition grounding is the path `L -> YA -> I`, restricted
to `L` IDs present in the source `locutions` metadata:

| Direct groundings per proposition occurrence | Proposition occurrences |
|---:|---:|
| 0 | 3,063 |
| 1 | 20,179 |
| 2 | 219 |
| 3 | 4 |

The canonical parser preserves every direct grounding, including raw locution
ID/text, speaker ID/name, YA ID/label, and chronological turn. Smoke scenarios
are deliberately stricter: every proposition in a selected map must have
exactly one direct grounding.

## RA, CA, and MA relations and corrected retention funnel

All 12,767 RA/CA/MA relation-node occurrences are preserved with original AIF
direction and a normalized patch label:

| AIF type | Original label | Normalized label | Count |
|---|---|---|---:|
| `RA` | Default Inference | `SUPPORT` | 5,687 |
| `CA` | Default Conflict | `ATTACK` | 1,232 |
| `MA` | Default Rephrase | `REPHRASE` | 5,848 |

Arity is measured only on proposition endpoints. YA anchoring edges into a
relation node are not incorrectly counted as extra argumentative premises.

| Proposition endpoint structure | Count |
|---|---:|
| Binary `1 -> relation -> 1` | 11,961 |
| N-ary, two or more proposition inputs | 674 |
| Incomplete, zero proposition inputs or outputs | 132 |

The requested deduplication identity is parent episode plus ordered original
source proposition IDs, original relation-node ID, and ordered original target
proposition IDs. It removes **zero** occurrences: raw and unique are both
12,767. This directly contradicts the 10,818 headline, leaving a gap of 1,949.
The shared-task paper reports 10,818, but a later data audit of the same 1,478
files independently reports 12,767 covered S relations. The discrepancy is
therefore a published-statistic/version-definition mismatch, not overlap that
can be removed by the requested identity key.

Sources: [DialAM site](http://dialam.arg.tech/),
[shared-task paper](https://aclanthology.org/2024.argmining-1.8/), and
[post-release relation audit](https://gist.github.com/ArneBinder/114b9472c2140f3e0ffc85f2b9247a06).

The exact relation-level retention funnel is:

| Stage | SUPPORT | ATTACK | REPHRASE | Total | Removed |
|---|---:|---:|---:|---:|---:|
| Raw occurrences | 5,687 | 1,232 | 5,848 | 12,767 | — |
| Unique identity keys | 5,687 | 1,232 | 5,848 | 12,767 | 0 |
| Binary direct `I -> S -> I` | 4,980 | 1,210 | 5,771 | 11,961 | 806 |
| Both endpoints singly grounded | 4,504 | 940 | 4,532 | 9,976 | 1,985 |
| Chronology resolvable | 4,439 | 919 | 4,433 | 9,791 | 185 |
| Behavior-compatible update context | 3,902 | 771 | 3,777 | 8,450 | 1,341 |
| Retained after overlapping-update resolution | 3,897 | 771 | 3,775 | 8,443 | 7 |

“Behavior-compatible” means the containing map has at least two propositions,
no unresolved TA precedence, exactly one direct `L -> YA -> I` grounding for
every proposition, and no ambiguous/n-ary relation touching that update.
“Retained” then keeps one map for a repeated parent-episode/proposition update:
most complete direct patch, widest earlier context, earliest map, then map ID.

Original binary edge direction is preserved separately from the normalized
incremental edge:

| Original AIF direction relative to chronology | Count |
|---|---:|
| Later proposition to earlier proposition | 7,963 |
| Earlier proposition to later proposition | 2,231 |
| Same turn | 52 |
| Unknown because grounding/chronology is incomplete | 1,715 |

The patch direction is always normalized from the new/later proposition to the
earlier proposition. For example, an RA whose premise is earlier and conclusion
later retains `earlier_to_later` as its original direction while producing a
later-to-earlier `SUPPORT` patch.

## Three complete source examples

These examples include every proposition and propositional relation in their
source map. `L` is the raw locution; `I` is its reconstructed proposition.

### `nodeset21469` — `cutietestrun10June2021`

| Turn | I ID | Speaker | L ID and raw locution | Reconstructed proposition |
|---:|---|---|---|---|
| 1 | `719346` | Lucy Frazer | `719340`: In my constituency I went to visit a community land trust | in her constituency Lucy Frazer went to visit a community land trust |
| 2 | `719357` | Lucy Frazer | `719351`: is a fantastic idea where the community has a proportion of land in a development which the community owns and is affordable housing for people who live and work locally | a community land trust is a fantastic area where the community has a proportion of land in a development which the community owns and is affordable housing for people who live and work locally |
| 3 | `719371` | Lucy Frazer | `719365`: the government has a variety of measures to tackle what is a very, very difficult problem | the government has a variety of measures to tackle what is a very, very difficult problem |

Relations:

- `719361` MA: original `719357 -> 719346`; normalized
  `719357 -REPHRASE-> 719346`.
- `719374` RA: original `719357 -> 719371` (`earlier_to_later`);
  normalized `719371 -SUPPORT-> 719357`.

### `nodeset21407` — `cutietestrun27may2021`

| Turn | I ID | Speaker | L ID and raw locution | Reconstructed proposition |
|---:|---|---|---|---|
| 1 | `714713` | Devi Sridhar | `714710`: this is a final reflection | this is a final reflection |
| 2 | `714721` | Devi Sridhar | `714718`: whether it's Israel and Palestine, whether it's about what's happening in the States among black men and the police, it's just such a shame that so many of these issues we forget the humanity in each other | whether it's Israel and Palestine, whether it's about what's happening in the States among black men and the police, it's just such a shame that so many of these issues we forget the humanity in each other |
| 3 | `714735` | Devi Sridhar | `714732`: actually we're all people, we want to be happy and have meaning and live our lives | actually we're all people, we want to be happy and have meaning and live our lives |
| 4 | `714746` | Devi Sridhar | `714743`: we get caught up on religion and beliefs and skin colour and our differences | we get caught up on religion and beliefs and skin colour and our differences |
| 5 | `714754` | Devi Sridhar | `714751`: when you travel across the world most people are pretty similar in terms of what they want out of their lives | when you travel across the world most people are pretty similar in terms of what they want out of their lives |

Relations, all originally and canonically later-to-earlier:

- `714727`: `714721 -REPHRASE-> 714713`.
- `714757`: `714754 -REPHRASE-> 714735`.
- `714760`: `714746 -REPHRASE-> 714721`.
- `714763`: `714754 -ATTACK-> 714746`.

### `nodeset23570` — `qt28102021wtxt`

| Turn | I ID | Speaker | L ID and raw locution | Reconstructed proposition |
|---:|---|---|---|---|
| 1 | `803431` | AudienceMember 20211028QT44 | `768056`: One of the things that I always think that the Scottish Government does is put a plaster over things | one of the things that Audience Member 20211028QT44 always thinks that the Scottish Government does is put a plaster over things |
| 2 | `803435` | AudienceMember 20211028QT44 | `803433`: There will be funding put into this for about three weeks and then when everyone leaves, when this international event has ended, we'll come back to there's litter everywhere | there will be funding put into this for about three weeks and then when everyone leaves, when this international event has ended, we'll come back to there's litter everywhere |
| 3 | `803442` | Kate Frobes | `803440`: Glasgow has a proud heritage of hosting international events and showcasing the best of Glaswegian hospitality and welcoming people | Glasgow has a proud heritage of hosting international events and showcasing the best of Glaswegian hospitality and welcoming people |
| 4 | `803449` | Kate Frobes | `803447`: that's what we should be celebrating | Glasgow's proud heritage as a host city for international events is what we should be celebrating |
| 5 | `803455` | Andrew Bowie | `768058`: Pay the workers so they don't go on strike | pay the workers so they don't go on strike |

Relations:

- `854166` RA: original `803431 -> 803435` (`earlier_to_later`);
normalized `803435 -SUPPORT-> 803431`.
- `854170` CA: `803442 -ATTACK-> 803435`.
- `854174` MA: `803449 -REPHRASE-> 803442`.
- `854178` MA: `803455 -REPHRASE-> 803435`.

## Smoke dataset

The deterministic build produces:

| Artifact | Updates/scenarios | Block examples | Positive blocks | All-negative blocks |
|---|---:|---:|---:|---:|
| Train | 108 updates | 160 | 58 | 102 |
| Held-out eval | 30 scenarios | 46 | 18 | 28 |

Train uses 24 original QT30 parent episodes. Evaluation uses six entirely
disjoint original parent episodes, five scenarios per episode. Individual map
IDs are never the split unit. The frozen held-out episodes are
`qt30092021wtxt`, `qt16092021wtxt`, `qt02092021wtxt`, `qt19082021wtxt`,
`qt05082021wtxt`, and `qt22072021wtxt`. The smoke relation distribution is:

| Split | SUPPORT | ATTACK | REPHRASE |
|---|---:|---:|---:|
| Train | 21 | 6 | 32 |
| Eval | 8 | 3 | 7 |

Earlier propositions are partitioned chronologically into fixed blocks of eight;
the last block may be shorter. Across all blocks for an update, every strictly
earlier proposition in the source map appears exactly once. There is no semantic
retrieval or candidate ranking.

All requested validators pass: chronology, IDs, normalized and original edge
direction, locution grounding, block coverage, gold-target presence,
direct-edge negatives, and dialogue-level leakage.

Across every behavior-compatible update in those same held-out episodes, the
natural distribution contains 3,149 scenarios: 1,661 positive, 1,488 negative,
788 SUPPORT edges, 178 ATTACK edges, and 779 REPHRASE edges. This is the
descriptive natural-distribution view; it is not downsampled.

For the prompt-ceiling diagnostic, a separate deterministic 30-scenario view
uses five one-block scenarios from each frozen episode: 8 single-edge SUPPORT,
8 single-edge ATTACK, 8 single-edge REPHRASE, and 6 all-negative scenarios.
Every scenario has at most eight earlier propositions, so each is exactly one
complete behavior-spec input and one model call per prompt/model combination.
The summaries are in [`data/dialam/eval_summary.json`](data/dialam/eval_summary.json).

## Known source risks and decisions

1. The official site publishes 19,842 locutions and 10,818 propositional
   relations, while direct inspection of this pinned shared-task archive finds
   21,136 unique locution metadata IDs and 12,767 raw/unique RA/CA/MA relation
   nodes. The post-release relation audit corroborates 12,767; the 1,949 gap
   remains unresolved and is not a deduplication effect.
2. Grounding defects: among 11,961 binary relations, 1,715 have at least one
   endpoint with no direct grounding and 270 have ambiguity without a missing
   endpoint; 9,976 have exactly one grounding at both endpoints.
3. Chronology defects: 30 maps have unresolved TA precedence. Among the 9,976
   singly grounded binary relations, 184 occur in those maps and one more has
   same-turn endpoints, leaving 9,791 chronology-resolvable relations.
4. Metadata defects: 535 maps contain duplicate locution metadata IDs; two maps
   contain five total metadata records pointing to non-`L` nodes. Duplicates are
   deterministically collapsed; non-`L` records are ignored and logged.
5. Exact exclusion policy: exclude unresolved maps, maps with any proposition
   not grounded exactly once, updates touched by non-binary/ambiguous relations,
   and relations whose endpoints do not have distinct resolved turns.
6. N-ary and incomplete relations are preserved by the parser but excluded from
   this first feasibility dataset.
7. Raw source and text-bearing transformed JSONL are local-only and Git-ignored;
   permission for this educational use does not establish redistribution rights.

## Reproduction

```bash
uv sync --frozen
uv run python scripts/build_dialam_gate.py
uv run python scripts/validate_dialam_gate.py
uv run pytest
```

Score model outputs containing `example_id` and `raw_response` fields with:

```bash
uv run python scripts/score_dialam_predictions.py predictions.jsonl
```

The scorer reports JSON validity, schema validity, invalid IDs, exact patch
accuracy, micro edge precision/recall/F1, relation macro-F1, direction accuracy,
and false edges per update.

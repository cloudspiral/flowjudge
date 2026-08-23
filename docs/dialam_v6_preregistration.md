# DialAM v6 Vives→QT30 transfer preregistration

Registered before building the final corpus, training a v6 checkpoint, or
generating any v6 development/frozen predictions. The submission-ready v5.1
checkpoint remains frozen at Git tag
`submission-ready-v5.1-pre-video-20260823`.

## Hypothesis and single intervention

V5.1 established that four-way candidate classification is materially better
than block-level free generation, but its frozen recall is only 41.7% and its
ATTACK F1 is 30.8%. The next experiment follows the previously declared
sequence: transfer relation semantics from VivesDebate, then finish on the
unchanged target-only QT30 pairwise corpus.

V6 changes only the training curriculum:

1. 4,096 VivesDebate rows, arranged as 2,048 adjacent positive/NONE pairs.
2. The exact 8,192 v5 QT30 rows, unchanged in prompt and assistant content,
   placed after the warm-up rows.

The base model, pairwise prompt, four labels, complete-block input, QLoRA
configuration, seed, three epochs, loss, pair ordering, inference scorer, and
deterministic JSON assembly remain unchanged.

## VivesDebate conversion

- Source: pinned VivesDebate version 3, CC BY-NC-SA 4.0.
- Text: original Spanish ADUs; the final QT30 stage returns training to English.
- Mapping: `RA→SUPPORT`, `CA→ATTACK`, and `MA→REPHRASE`.
- Retention: direct, single-target, chronology-compatible relations only.
  Multi-target relation slots, malformed targets, forward RA/MA edges,
  ambiguous same-pair labels, and relations without a true earlier NONE
  candidate are excluded deterministically.
- CA is temporally oriented later-to-earlier because conflict is symmetric;
  RA and MA are never reversed.
- Positive targets are balanced at 682 SUPPORT, 683 ATTACK, and 683 REPHRASE.
  Every positive has a same-block lexically hard but unannotated NONE partner.
- No Vives text or label is derived from the QT30 development or frozen sets.

## Development-only selection

Generate raw four-label scores on the unchanged 30-scenario episode-disjoint
QT30 development set. Reuse the existing fixed NONE-margin grid from 0.00 to
3.00 in 0.05 increments and the same deterministic ranking rule. V6 advances
to frozen evaluation only if the selected development margin satisfies every
condition:

- exact patch accuracy ≥ 53.33%;
- edge F1 ≥ 56.00%;
- relation macro-F1 ≥ 54.27%;
- false edges/update ≤ 0.300;
- NONE scenarios with a false edge ≤ 2/6;
- JSON validity and schema validity = 100%.

If no margin passes, stop and retain v5.1. Do not inspect v6 frozen predictions.

## One frozen evaluation and promotion rule

If development passes, lock its selected margin and apply it once to the same
30-scenario frozen evaluation. Promote v6 only if all conditions hold relative
to v5.1:

- exact patch accuracy ≥ 53.33%;
- edge F1 ≥ 50.62% (at least +3.00 percentage points);
- relation macro-F1 ≥ 47.44%;
- ATTACK F1 ≥ 30.77%;
- false edges/update ≤ 0.267;
- NONE scenarios with a false edge = 0/6;
- JSON validity and schema validity = 100%.

The frozen judge is run only after deterministic promotion passes. A failed v6
remains an append-only experiment-ledger point and cannot replace the public
v5.1 submission artifact.

## Reproduction sequence

```bash
.venv/bin/python scripts/build_dialam_training_v6.py
modal run scripts/modal_dialam_qlora.py \
  --action train --size 12288 --dataset-version v6
modal run scripts/modal_dialam_qlora.py \
  --action evaluate --target tuned --size 12288 --dataset-version v6 \
  --eval-split v3_dev \
  --output-path results/dialam_model_generation/v6_n12288_dev_raw/predictions.jsonl
.venv/bin/python scripts/calibrate_dialam_v6.py
```

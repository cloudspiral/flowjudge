# DialAM v8 prior-aware listwise classification preregistration

Registered after the completed v7.2 development report and before building the
v8 corpus, training either v8 checkpoint, or generating v8 development or
frozen predictions. The submission-selected v5.1 checkpoint remains frozen at
Git tag `submission-ready-v5.1-pre-video-20260823`.

## Findings carried forward

The training-episode candidate pool contains 70,719 NONE decisions and 5,634
direct-relation decisions: a 7.38% positive rate. V5 instead trained on a
balanced 50% relation / 50% NONE corpus. Its calibrated development result was
the best so far, but seven of nine false-positive edges were SUPPORT.

V7 continued from v5 with reciprocal relation-vs-NONE preferences. Its original
grid result regressed to 40.0% exact patch accuracy and 44.4% edge F1. The
separately preregistered v7.2 transition-complete calibration recovered 53.3%
exact accuracy and 52.2% edge F1 while reducing all-NONE failures from 5/6 to
2/6, proving that much of v7's regression was a score-scale shift. It still
missed the development gate: SUPPORT caused seven of ten false positives and
REPHRASE recall was only 25%.

The earlier VivesDebate transfer experiment also failed. V8 therefore adds no
external dataset. It targets the two remaining QT30-specific defects together:
an overly positive training prior and a loss that does not compare the gold
label against all four allowed inference labels.

## Single intervention

V8 is prior-aware, restricted-label classification over same-block triples.
Build exactly 4,096 deterministic training groups from the unchanged QT30
training parent episodes. Every group contains:

- one positive candidate, balanced as 1,366 ATTACK, 1,365 REPHRASE, and 1,365
  SUPPORT rows;
- two different direct-edge-negative candidates from the same complete block,
  for 8,192 NONE rows;
- no development or frozen episode, semantic retriever, synthetic text, or
  frozen-derived signal.

Only positive candidates whose complete block has at least two eligible NONE
candidates may be selected. Positives are selected uniquely before deterministic
repetition; the two lexical-overlap-hard negatives are distinct within a group
and rotate across repeated positive occurrences. The full corpus therefore has
12,288 rows with a fixed 2:1 NONE-to-relation ratio. This is deliberately more
conservative than v5's 1:1 ratio without reproducing the raw 12.5:1 corpus prior
that could erase relation recall.

For each row, score the exact allowed labels `NONE`, `SUPPORT`, `ATTACK`, and
`REPHRASE` using the same mean label-token log probability used at inference.
Apply one cross-entropy loss over those four scores with the row's gold label as
the target. This directly teaches both relation-vs-NONE and relation-vs-relation
ranking. Do not add a generative-token NLL, pairwise preference term, class
weight, or label smoothing.

## Fixed training configuration

Continue from the exact v5 n=8192 adapter with tree SHA-256
`cbd9a2a6cae8ab988d78b7694ac9f9829bf9262bbb7a6fab838894a5080f122a`.
Keep Qwen3-0.6B, 4-bit loading, rank-16 LoRA, alpha 32, the same target modules,
maximum sequence length 2,048, batch size 2, gradient accumulation 4, cosine
scheduling, 5% warmup, AdamW 8-bit, and seed 20260823. Train one epoch at a
fixed learning rate of `2e-5`. Do not run a hyperparameter search.

## Resumability gate

Use the same persistent full-state protocol proven by v7. Every saved checkpoint
must include model/adapter state, optimizer, scheduler, RNG, and Trainer state,
then be committed to the Modal volume with an immutable run identity covering
data hash, source-adapter hash, configuration, objective version, and seed.

Before the full run, a nested 252-row prefix must pass this smoke:

1. intentionally stop after the committed step-8 checkpoint;
2. resume with `resume_mode=required` at step 8, not step 0;
3. complete training and reload the saved adapter;
4. invoke the completed run again in `auto` mode and receive the identical
   manifest and adapter hash without training.

The full n=12,288 run checkpoints every 100 optimizer steps, retains the newest
two full states, uses `resume_mode=auto`, and must resume the latest valid
checkpoint after interruption. Partial newer checkpoints are ignored; identity
mismatches are rejected; completed reruns are idempotent.

## Development-only selection

Generate four-label scores on the unchanged 30-scenario episode-disjoint QT30
development set. Construct a transition-complete nonnegative NONE-margin set
from 0.0 plus every distinct positive
`best_positive_score - NONE_score` boundary in those fixed scores. The candidate
set is score-derived without consulting gold labels. Use the existing ranking:
all gates, edge F1, exact patch, macro-F1, lower false edges, fewer all-NONE
false-edge cases, then larger margin.

Advance only if every condition holds:

- exact patch accuracy at least 53.33%;
- edge F1 at least 56.00%;
- relation macro-F1 at least 54.27%;
- false edges/update at most 0.300;
- all-NONE scenarios with a false edge at most 2/6;
- JSON validity and schema validity exactly 100%.

If development fails, make zero v8 frozen candidate or judge calls and retain
v5.1.

## One frozen evaluation and promotion rule

If development passes, lock its margin and generate the unchanged 30-scenario
frozen scores exactly once. Promote v8 only if every condition holds:

- exact patch accuracy at least 53.33%;
- edge F1 at least 50.62%;
- relation macro-F1 at least 47.44%;
- ATTACK F1 at least 30.77%;
- false edges/update at most 0.267;
- all-NONE scenarios with a false edge exactly 0/6;
- JSON validity and schema validity exactly 100%.

Run the unchanged blinded judge only after every deterministic frozen promotion
condition passes. A failed v8 remains an append-only development-ledger point.

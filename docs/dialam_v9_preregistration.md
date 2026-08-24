# DialAM v9 model-error-mined corrective continuation preregistration

Registered after the completed v8.1 development report and before constructing
v9 mining inputs, generating any v9 mining score, building either v9 corpus, or
training/evaluating a v9 checkpoint. The submission-selected v5.1 checkpoint
remains frozen at Git tag `submission-ready-v5.1-pre-video-20260823`.

## Findings carried forward

V8 replaced the candidate-token objective with four-label score
cross-entropy and increased NONE exposure from 50% to 66.7%. After its
transition-complete development calibration, it reached 50.0% exact patch
accuracy, 52.0% edge F1, 53.2% relation macro-F1, and 0.433 false edges per
update. It failed five of seven development checks. SUPPORT still caused 10 of
13 false-positive edges, including false edges on 3/6 all-NONE scenarios.

A post-result diagnostic compared v5.1 and v8.1 on the already-used development
set. V8 gained two exact scenarios but lost three; score mixtures did not beat
v5.1. This rules out checkpoint/score interpolation as the next experiment.
The previous v2, v4, and v8 results also show that generic lexical negatives or
more rows alone are insufficient. V9 therefore changes how negative examples
are selected, not the source corpus, base model, or task formulation.

## Single intervention: selected-model error mining

Use only retained QT30 training parent episodes. Enumerate every unambiguous
candidate-level NONE decision, excluding any exact NONE decision already used
by the v5 n=8192 training corpus. Rank the remaining decisions using the frozen
v5 lexical-hardness ordering and retain exactly 12,288 unique mining candidates.
This deterministic prefilter is fixed before model scoring and contains no
development or frozen episode.

Score every prefiltered candidate with the exact selected v5 n=8192 adapter and
the unchanged mean allowed-label token log-probability scorer. Define model
hardness as:

`max(SUPPORT, ATTACK, REPHRASE score) - NONE score`

Select exactly 4,096 unique NONE candidates by descending model hardness, then
descending SUPPORT-minus-NONE evidence, lexical hardness, and a seeded stable
hash. Do not impose a relation quota, consult development/frozen labels, use a
semantic retriever, or synthesize text. The observed score and winning-relation
distributions are reported after selection but cannot change it.

Build exactly 8,192 v9 training rows:

- the exact 4,096 positive rows from the frozen v5 corpus, preserving its
  1,366 ATTACK / 1,365 REPHRASE / 1,365 SUPPORT mix;
- the 4,096 selected model-hard NONE rows;
- no v5 NONE row, v8 row, external dataset, development episode, or frozen
  episode.

Every row uses the unchanged candidate-level prompt and exact one-label target.
Training applies cross-entropy over the same four exact inference-label scores
used by v8. The single experimental variable is replacement of lexical/sibling
NONE sampling with selected-model error mining.

## Fixed training configuration

Continue from the exact v5 n=8192 adapter with tree SHA-256
`cbd9a2a6cae8ab988d78b7694ac9f9829bf9262bbb7a6fab838894a5080f122a`.
Keep Qwen3-0.6B, 4-bit loading, rank-16 LoRA, alpha 32, the same target modules,
maximum sequence length 2,048, batch size 2, gradient accumulation 4, cosine
scheduling, 5% warmup, AdamW 8-bit, and seed 20260823. Train one epoch at
learning rate `5e-6`. Do not run a learning-rate, epoch, model, or loss search.

The lower rate is fixed because v8's `2e-5` continuation gained recall but
overrode useful v5 calibration. V9 is a corrective rehearsal pass, not a fresh
task fit.

## Interruption and idempotence gates

Mining uses an immutable identity covering the v5 adapter hash, mining-input
hash, candidate order, label scorer, and implementation version. Persist each
completed chunk of at most 128 candidates to the Modal volume and commit before
starting the next chunk. A rerun must reject identity drift, reuse valid chunks,
score only missing candidates, and return the identical completed result
without loading the model.

Training uses the full-state protocol proven by v8: every checkpoint includes
adapter/model state, optimizer, scheduler, RNG, and Trainer state and is
committed to persistent storage. Before the full run, a nested 256-row prefix
must intentionally stop after a committed step-8 checkpoint, resume at step 8,
complete, reload the adapter, and return idempotently on a third invocation.
The full n=8192 run checkpoints every 100 optimizer steps, retains the newest
two states, and uses `resume_mode=auto`.

Development and any allowed frozen evaluation persist each completed scenario
independently under an immutable identity. Reruns score only missing scenarios
or return a no-op result when complete.

## Development-only selection

Generate four-label scores on the unchanged 30-scenario episode-disjoint QT30
development set. Construct a transition-complete nonnegative NONE-margin set
from `0.0` plus every distinct positive
`best_positive_score - NONE_score` boundary, without consulting gold labels.
Use the existing ranking: all gates, edge F1, exact patch, macro-F1, lower false
edges, fewer all-NONE false-edge cases, then larger margin.

Advance only if every condition holds:

- exact patch accuracy at least 53.33%;
- edge F1 at least 56.00%;
- relation macro-F1 at least 54.27%;
- false edges/update at most 0.300;
- all-NONE scenarios with a false edge at most 2/6;
- JSON validity and schema validity exactly 100%.

If development fails, make zero v9 frozen candidate or judge calls and retain
v5.1.

## One frozen evaluation and promotion rule

If development passes, lock its margin and generate the unchanged 30-scenario
frozen scores exactly once. Promote v9 only if every condition holds:

- exact patch accuracy at least 53.33%;
- edge F1 at least 50.62%;
- relation macro-F1 at least 47.44%;
- ATTACK F1 at least 30.77%;
- false edges/update at most 0.267;
- all-NONE scenarios with a false edge exactly 0/6;
- JSON validity and schema validity exactly 100%.

Run the unchanged blinded judge only after every deterministic frozen promotion
condition passes. A failed v9 remains an append-only development-ledger point.

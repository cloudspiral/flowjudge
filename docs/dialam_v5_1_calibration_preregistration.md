# DialAM v5.1 NONE-margin calibration preregistration

This fallback is registered before any tuned v5 development prediction exists.
It does not change the v5 checkpoint, training data, QLoRA configuration, label
scores, external Behavior Spec, development set, frozen set, judge, or metrics.

## Motivation

The 20 training episodes contain 76,353 unambiguous candidate decisions:
70,719 NONE, 2,618 SUPPORT, 517 ATTACK, and 2,499 REPHRASE. Only 7.379% are
positive. V5 deliberately trains exact one-positive/one-NONE contrast batches,
so its training prior is 50% positive. Raw v5 uses the already-preregistered
four-label argmax and remains the primary result. V5.1 exists only to test a
fixed validation-time correction for this known prior shift if raw v5 fails.

## Fixed calibration rule

For each candidate, retain the highest-scoring positive label using the fixed
tie order SUPPORT, ATTACK, REPHRASE. Emit it only when
`best_positive_mean_logprob - NONE_mean_logprob > margin`; otherwise emit NONE.
Candidate IDs and block patches are still assembled deterministically.

Evaluate margins `0.00, 0.05, ..., 3.00` on the existing 30-scenario,
four-parent-episode development set. Select lexicographically by:

1. all seven existing v5 development gate checks passing;
2. edge F1;
3. exact patch accuracy;
4. relation macro-F1;
5. lower false edges per update;
6. lower NONE scenarios with a false edge;
7. larger margin.

If no margin passes all seven unchanged development checks, v5.1 fails and the
frozen benchmark is not run. If one passes, freeze that one margin, obtain one
set of frozen candidate label scores, apply the margin without further tuning,
run the unchanged judge once, and use the unchanged v5 frozen promotion gate.

Raw v5 and calibrated v5.1 are separate points in the experiment ledger. A
v5.1 PASS may promote the pairwise pipeline; a failure leaves public v3 selected.

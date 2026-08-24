# DialAM v7.2 score-scale calibration preregistration

Registered after the completed v7.1 report showed that the selected development
margin saturated at the original grid maximum of 3.0, and before computing any
candidate, metric, or selected prediction at a margin above 3.0. This is a
separate, explicitly post-v7.1 calibration experiment—not a change to the v7.1
preregistration or result.

## Fixed inputs and scope

V7.2 reuses the exact v7 checkpoint, prompts, pairwise decisions, and 30-scenario
episode-disjoint development set. The raw development prediction artifact is
fixed at SHA-256
`ac96189df65ae020bb1d590428cd4d330de5e90802e96c37ae2031e5037b4ed7`;
the adapter tree is fixed at
`868e00d526c91555d6c03d55d017b23fef3996b5006e0243e6ce65403c08bfd6`.
There is no retraining, new dataset, new example, new model call, or changed
label-scoring rule. The frozen evaluation remains sealed unless the unchanged
development gate passes.

## Transition-complete candidate set

V7.1 used margins 0.00 through 3.00 in increments of 0.05. V7.2 evaluates 3.0
again as an audit anchor plus every distinct decision transition above 3.0 in
the persisted raw development scores. A transition is exactly
`best_positive_score - NONE_score` for one supplied candidate. Candidate
margins are sorted and deduplicated.

This construction uses model scores only and does not consult gold labels or
metrics. Since a positive label is emitted exactly when its score gap is
strictly greater than the margin, the resulting finite set covers every
possible prediction configuration at or above 3.0 without an arbitrary upper
cap.

## Locked selection and development gate

Reuse the existing deterministic ranking in this exact order:

1. all development gates passed;
2. higher edge F1;
3. higher exact patch accuracy;
4. higher relation macro-F1;
5. fewer false edges per update;
6. fewer all-NONE scenarios with false edges;
7. larger margin.

The selected margin advances only if every original v7 development condition
holds:

- exact patch accuracy at least 53.33%;
- edge F1 at least 56.00%;
- relation macro-F1 at least 54.27%;
- false edges/update at most 0.300;
- all-NONE scenarios with a false edge at most 2/6;
- JSON validity and schema validity exactly 100%.

If any condition fails, record v7.2 as failed, make zero frozen candidate or
judge calls, and retain v5.1.

## Frozen evaluation and promotion rule

If development passes, lock the selected margin and generate the unchanged
30-scenario frozen scores exactly once. Apply that margin without further
selection. Promote only if every original v7 frozen condition holds:

- exact patch accuracy at least 53.33%;
- edge F1 at least 50.62%;
- relation macro-F1 at least 47.44%;
- ATTACK F1 at least 30.77%;
- false edges/update at most 0.267;
- all-NONE scenarios with a false edge exactly 0/6;
- JSON validity and schema validity exactly 100%.

Run the unchanged blinded judge only after every deterministic frozen condition
passes. V5.1 otherwise remains the submission model.

## Reproduction sequence

```bash
PYTHONPATH=src .venv/bin/python scripts/calibrate_dialam_v7_2.py

# Run only if reports/dialam_v7_2_calibration.json says PASS_RUN_FROZEN_ONCE.
modal run scripts/modal_dialam_qlora.py \
  --action evaluate --target tuned --size 8192 --dataset-version v7 \
  --eval-split frozen \
  --output-path results/dialam_model_generation/v7_n8192_frozen_raw/predictions.jsonl
```

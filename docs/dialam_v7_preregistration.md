# DialAM v7 QT30 reciprocal-preference preregistration

Registered before building the v7 corpus, training either v7 checkpoint, or
generating any v7 development or frozen predictions. The submission-selected
v5.1 checkpoint remains frozen at Git tag
`submission-ready-v5.1-pre-video-20260823`.

## Finding carried forward

The v6 VivesDebate transfer experiment failed its episode-disjoint development
gate: exact patch accuracy was 33.3%, edge F1 was 40.0%, false edges/update was
0.667, and five of six all-NONE scenarios received false edges. Thirteen of its
twenty false-positive edges were SUPPORT. No v6 frozen prediction or judge call
was made.

V5.1 therefore remains the baseline. Its four-label candidate scorer was the
best tested architecture, but its training loss optimized only the gold label's
token likelihood while inference compares the mean token log likelihood of all
four allowed labels. V7 tests one controlled, QT30-only correction to that
training/inference mismatch.

## Single intervention

V7 continues from the exact v5 n=8192 LoRA adapter (tree SHA-256
`cbd9a2a6cae8ab988d78b7694ac9f9829bf9262bbb7a6fab838894a5080f122a`)
and converts the exact frozen v5 corpus into reciprocal preferences:

- for each direct-relation candidate, `gold relation > NONE`;
- for its same-block lexically hard negative, `NONE > that relation`.

The preference score is the same mean label-token log probability used at
inference. The fixed loss is
`softplus(-(chosen_score - rejected_score)) + 0.1 * chosen_label_NLL`.
This directly penalizes the known false-relation boundary while the small NLL
anchor preserves the chosen label. There is no external corpus, new episode,
semantic retrieval, or frozen-set-derived training signal.

One continuation epoch is used with learning rate `5e-5`, batch size 2,
gradient accumulation 4, cosine scheduling, 5% warmup, AdamW 8-bit, and seed
20260823. No hyperparameter search is permitted. The original Qwen3-0.6B base,
v5 rank-16 adapter architecture, prompt, labels, deterministic scorer, and JSON
assembly remain fixed.

## Resumability gate

Before the full run, the nested first 256 preference rows must pass an
interruption/recovery smoke:

1. train until the first step-8 full Trainer checkpoint;
2. commit model/adapter, optimizer, scheduler, RNG, and Trainer state plus a
   run-identity manifest to the persistent Modal volume;
3. intentionally terminate the run after that committed checkpoint;
4. resume with `resume_mode=required` and finish from step 8 rather than step 0;
5. save and reload the final smoke adapter.

The full n=8192 run checkpoints every 100 optimizer steps and retains the two
newest full states. `resume_mode=auto` must resume the latest valid state, return
an already-complete run without retraining, reject incomplete state without a
valid checkpoint, and reject any changed data/source-adapter/config/seed/code
identity. Trainer data skipping remains enabled so optimizer, scheduler, RNG,
and sample position continue from the saved step.

## Development-only selection

Generate four-label scores on the unchanged 30-scenario episode-disjoint QT30
development set. Reuse the fixed NONE-margin grid from 0.00 to 3.00 in 0.05
increments and its deterministic ranking rule. V7 advances to frozen evaluation
only if the selected development margin satisfies every condition:

- exact patch accuracy at least 53.33%;
- edge F1 at least 56.00%;
- relation macro-F1 at least 54.27%;
- false edges/update at most 0.300;
- all-NONE scenarios with a false edge at most 2/6;
- JSON validity and schema validity exactly 100%.

If any condition fails, stop, record the result, and retain v5.1 without
generating v7 frozen predictions.

## One frozen evaluation and promotion rule

If development passes, lock its selected margin and apply it once to the same
30-scenario frozen evaluation. Promote v7 only if all conditions hold:

- exact patch accuracy at least 53.33%;
- edge F1 at least 50.62%, which is at least 3.00 percentage points over v5.1;
- relation macro-F1 at least 47.44%;
- ATTACK F1 at least 30.77%;
- false edges/update at most 0.267;
- all-NONE scenarios with a false edge exactly 0/6;
- JSON validity and schema validity exactly 100%.

Run the unchanged blinded judge only after every deterministic frozen promotion
condition passes. A failed v7 remains an append-only experiment-ledger point.

## Reproduction sequence

```bash
.venv/bin/python scripts/build_dialam_training_v7.py

# Expected intentional failure after a committed step-8 checkpoint.
modal run scripts/modal_dialam_qlora.py \
  --action train --size 256 --dataset-version v7-smoke \
  --resume-mode auto --interrupt-after-steps 8

# Must report resumed_from_step=8 and complete without restarting.
modal run --detach scripts/modal_dialam_qlora.py \
  --action train --size 256 --dataset-version v7-smoke \
  --resume-mode required

# Full run is itself safe to rerun after interruption.
modal run --detach scripts/modal_dialam_qlora.py \
  --action train --size 8192 --dataset-version v7 --resume-mode auto

modal run scripts/modal_dialam_qlora.py \
  --action evaluate --target tuned --size 8192 --dataset-version v7 \
  --eval-split v3_dev \
  --output-path results/dialam_model_generation/v7_n8192_dev_raw/predictions.jsonl
.venv/bin/python scripts/calibrate_dialam_v7.py
```

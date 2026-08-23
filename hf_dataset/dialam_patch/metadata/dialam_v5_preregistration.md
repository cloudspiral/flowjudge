# DialAM v5 pairwise-classification preregistration

Status: **FROZEN BEFORE DATA BUILDING OR TRAINING**.

V3 proved that same-update positive/NONE pairing and equal per-example loss
materially improve the task. V4 then improved development edge F1 from 21.1%
to 34.1% and ATTACK F1 from 0% to 54.5%, but doubled NONE scenarios with a
false edge from 2/6 to 4/6 and reduced REPHRASE F1 to 0%. This establishes a
learnable signal and a formulation bottleneck: free generation of a sparse,
variable-length multi-edge patch trades relation recall against NONE
calibration.

## Intervention

Keep the public behavior unchanged: one complete comparison block enters and
one bare relation-patch JSON object exits. Change only the learned internal
primitive:

1. For each earlier proposition in a supplied block, present the model with
   the complete block plus that proposition's supplied candidate ID.
2. QLoRA-train the model to return exactly one of `NONE`, `SUPPORT`, `ATTACK`,
   or `REPHRASE`.
3. At inference, score the four allowed label continuations without sampling,
   choose the highest mean token log-probability with a fixed NONE-first tie
   break, and deterministically assemble every non-NONE decision into the
   required patch JSON using only supplied IDs.

The n=8,192 training corpus contains 4,096 exact within-block contrast pairs.
Every pair contains one positive candidate and one hard NONE candidate from
the same new proposition and complete comparison block. The positive half is
fixed at 1,366 ATTACK, 1,365 SUPPORT, and 1,365 REPHRASE decisions. Unique
examples are exhausted before deterministic repetition; repeated positives
cycle through distinct in-block NONE candidates before reusing one. Each
per-device batch contains exactly one positive/NONE pair, and prompt-masked
loss is averaged per example before the batch mean.

The base remains `Qwen/Qwen3-0.6B`; the LoRA rank, modules, optimizer, learning
rate, seed, effective batch size, three epochs, maximum sequence length, and
four-bit Unsloth QLoRA configuration remain unchanged. No VivesDebate rows,
new source corpus, generated labels, semantic retrieval, probability
threshold, decoding sample, or hyperparameter search is introduced.

Pairs with multiple direct labels between the same ordered proposition pair
are excluded as ambiguous. Parent-episode splits remain the v3 split: 20 train,
four development, and six reused-frozen episodes, with no overlap.

## Development gate

The existing four-episode, 30-case v3 development set remains the only model
selection gate. V5 reaches the reused frozen benchmark only if all conditions
hold:

- exact patch accuracy at least `0.30`;
- edge F1 at least `0.39146341463414636` (five points above v4 development);
- relation macro-F1 at least `0.3151515151515151`;
- false edges/update at most `0.3333333333333333`;
- at most two of six NONE cases have a false edge; and
- JSON and schema validity remain 100%.

Base and tuned pairwise systems will both be persisted on development. The
gate is based on the tuned result and the fixed thresholds above, not on a
post-hoc comparison.

## Reused frozen-benchmark promotion rule

V3 and v4 errors informed this intervention, so the 30-case frozen set is a
reused comparison benchmark; the staff-heldout set remains the final unbiased
evaluation. If the development gate passes, v5 replaces v3 only if every
reused-benchmark condition holds:

- exact patch accuracy at least `0.43333333333333335`;
- edge F1 at least `0.40`;
- relation macro-F1 at least `0.39`;
- ATTACK F1 at least `0.30`;
- false edges/update at most `0.30`;
- at most one of six NONE cases has a false edge;
- mean frozen-judge Robustness at least `2.966666666666667`; and
- JSON and schema validity remain 100%.

The original frozen reliability bar remains unchanged and will be reported
separately. V3 remains selected if either complete v5 gate fails. Every result
will be appended to the versioned experiment ledger; development-only points
will never be plotted as frozen-evaluation points.

## Reproduction

```bash
.venv/bin/python scripts/build_dialam_training_v5.py
modal run scripts/modal_dialam_qlora.py --action train --size 8192 --dataset-version v5
modal run scripts/modal_dialam_qlora.py --action evaluate --target base --size 8192 --dataset-version v5 --eval-split v3_dev --output-path results/dialam_model_generation/v5_base_dev/predictions.jsonl
modal run scripts/modal_dialam_qlora.py --action evaluate --target tuned --size 8192 --dataset-version v5 --eval-split v3_dev --output-path results/dialam_model_generation/v5_n8192_dev/predictions.jsonl
.venv/bin/python scripts/score_dialam_predictions.py results/dialam_model_generation/v5_n8192_dev/predictions.jsonl --examples data/dialam/training/v3_dev_eval_examples.jsonl
```

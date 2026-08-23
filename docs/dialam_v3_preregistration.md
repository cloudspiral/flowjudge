# DialAM v3 paired-loss experiment preregistration

Status: **FROZEN BEFORE TRAINING**.

The v3 experiment addresses the diagnosed v2 failure without changing the base
model, LoRA rank, optimizer, learning rate, epochs, prompt, generation settings,
or frozen judge. It makes two declared interventions:

1. select 2,048 positive/NONE pairs, where both blocks come from the exact same
   update and each update is used once; and
2. average assistant-token loss within each example before averaging examples,
   so the short `{"relations":[]}` target receives the same configured weight as
   its longer positive sibling.

The resulting private corpus has 4,096 rows. Four deterministic parent episodes
are excluded as a new 30-case development set. The original six-episode,
30-scenario evaluation and frozen judge rubric are unchanged and will be used
once, after the development check.

## Fixed variables

- Base: `Qwen/Qwen3-0.6B`
- Method: four-bit Unsloth QLoRA
- Epochs: 3
- Learning rate: `2e-4`
- Effective batch size: 8
- LoRA: rank 16, alpha 32, dropout 0
- Seed: `20260823`
- Prompt: unchanged DialAM zero-shot behavior prompt
- Decoding: deterministic greedy generation

## Primary success rule

V3 is materially better only if all of these hold on the frozen evaluation:

- edge F1 is at least `0.2642857142857143`, which is five absolute points above
  the best prior semantic result (v2/n=2048);
- false edges per update are at most `0.6666666666666666`, no worse than the
  selected v1/n=2048 baseline;
- at most two of the six frozen NONE cases contain a false edge;
- JSON validity and schema validity remain 100%.

All metrics and comparisons will be reported even if the rule fails. The v1
public model remains selected unless v3 clears the complete rule.

## Reproduction

```bash
.venv/bin/python scripts/build_dialam_training_v3.py
modal run scripts/modal_dialam_qlora.py --action train --size 4096 --dataset-version v3
modal run scripts/modal_dialam_qlora.py --action evaluate --target tuned --size 4096 --dataset-version v3 --eval-split v3_dev --output-path results/dialam_model_generation/v3_n4096_dev/predictions.jsonl
.venv/bin/python scripts/score_dialam_predictions.py results/dialam_model_generation/v3_n4096_dev/predictions.jsonl --examples data/dialam/training/v3_dev_eval_examples.jsonl
modal run scripts/modal_dialam_qlora.py --action evaluate --target tuned --size 4096 --dataset-version v3 --eval-split frozen --output-path results/dialam_model_generation/v3_n4096/predictions.jsonl
.venv/bin/python scripts/evaluate_dialam_model.py --predictions results/dialam_model_generation/v3_n4096/predictions.jsonl --output-dir results/dialam_model_eval/v3_n4096 --approval APPROVE_DIALAM_MODEL_JUDGING
```

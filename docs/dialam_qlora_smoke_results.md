# DialAM QLoRA v1 / n=256 permanent baseline

## Verdict

**End-to-end smoke PASS; task reliability not yet achieved.** The fixed
Qwen3-0.6B QLoRA job trained all 96 optimizer steps on an NVIDIA L4, saved the
adapter, reloaded it from the Modal Volume, generated all 30 frozen predictions,
and completed all 30 blinded frozen-judge calls. The checkpoint learned the
required output contract, but semantic relation selection remains weak at
`n=256`. The pre-registered reliability bar is not being reinterpreted as a
training-smoke threshold.

This is the requested reporting point before the `n=512`, `n=1024`, and
`n=2048` runs. There is no compatibility or infrastructure blocker to those
runs.

## Frozen evaluation result

Both rows use the exact same 30 scenarios, evaluation SHA-256
`76badbae23ace13b22822335fa794f7cc796ce0d02753666ebd83c4d7260305b`,
judge model `gpt-5.4-mini-2026-03-17`, and rubric SHA-256
`b27fe77a9adce4fe586327fa556a51352efb76d402b93676df2173225be35f66`.

| Metric | Untouched base | QLoRA n=256 |
|---|---:|---:|
| JSON validity | 13.3% | **100.0%** |
| Schema validity | 0.0% | **100.0%** |
| Exact patch accuracy | 0.0% (0/30) | **16.7% (5/30)** |
| Edge precision | 0.0% | 11.1% |
| Edge recall | 0.0% | 8.3% |
| Edge F1 | 0.0% | **9.5%** |
| Relation macro-F1 | 0.0% | **5.6%** |
| Direction accuracy | not evaluable | 100.0% (2 edges) |
| Invalid IDs | 0 | 0 |
| False edges/update | 0.00 (no valid edges) | **0.53** |
| Judge Spec adherence | 0.23/4 | **4.00/4** |
| Judge Robustness | 1.40/4 | **2.47/4** |

The n=256 adapter recovered two SUPPORT edges, but ATTACK and REPHRASE F1 both
remain zero. It produced 16 false-positive edges, including 14 labeled SUPPORT,
so false-edge SUPPORT overprediction is the primary failure. Three of the six
gold-NONE scenarios received a spurious SUPPORT edge, a 50% NONE-case false
positive rate. This is evidence for continuing the fixed data-efficiency curve,
not evidence that `n=256` reliably holds the complete behavior.

## Fixed training run

- Base: `Qwen/Qwen3-0.6B`
- Method: Unsloth 4-bit QLoRA; no base-model comparison or hyperparameter search
- Data: 256-example exact prefix of the frozen nested corpus; 24 training parent
  episodes and zero overlap with the six held-out parent episodes
- Mix: 103 NONE, 51 SUPPORT, 38 ATTACK, 51 REPHRASE, and 13 MIXED examples
- LoRA: rank 16, alpha 32, dropout 0, all attention and MLP projection modules
- Optimization: 3 epochs, effective batch size 8, learning rate `2e-4`, cosine
  schedule, 5% warmup, `adamw_8bit`, seed `20260823`, prompt-only masking
- Runtime: 116.6 seconds of training, 96 optimizer steps, final aggregate train
  loss 0.1335 on an NVIDIA L4
- Reload probe: `{"relations":[]}`

The adapter remains on Modal Volume `flowjudge-dialam-qwen3` at
`checkpoints/n256/adapter` and is also downloaded to ignored local artifacts.
The saved adapter weights SHA-256 is
`c29156f0eeec3e609319dc05fd3cd55b4baa389bdef03d55696f3bdf9bd44613`.
Candidate responses, deterministic diagnostics, judge responses, SDK envelopes,
and the complete training history remain local and ignored.

The permanent, text-free baseline manifest is
[`baselines/dialam_v1_n256/manifest.json`](../baselines/dialam_v1_n256/manifest.json).
It also pins the canonical Qwen Hub revision, the Unsloth 4-bit runtime revision,
their model/config hashes, and every local private artifact below.

| Private local artifact | SHA-256 |
|---|---|
| `data/dialam/training/dialam_n256.jsonl` | `53ac800e7272c06f8389af55b75061bca883e4dceb3c897c16486e1443579749` |
| `data/dialam/training/frozen_eval_inputs.jsonl` | `8e959ff0a1b58fc4ed9929f5c9259d70598a50d5f928e67b408cd1e35cf154a5` |
| `results/dialam_model_eval/base/predictions.jsonl` | `3501dc29c8da8517ef3c9ba505546ad4e8da927d690ccdade4b766a80d02ec13` |
| `results/dialam_model_eval/base/records.jsonl` | `f73cee950a0527394d650f6ebbd3f1ec3c496658b6e5ada4902b0f76f7fda98a` |
| `results/dialam_model_eval/base/judge_transcripts.jsonl` | `40a6971f8a2a1e4e763633d7dc384d946f7144ef6feb6096c55d2bda190ee98b` |
| `results/dialam_model_eval/base/summary.json` | `85dd97bf9edb914457afc7a2b4f1507d424661ab89545956629797cf4712383a` |
| `results/dialam_model_eval/n256/predictions.jsonl` | `9245a34bfdc5942ce8c76aa40ad98f824577c956225398f4fa096b91d939ece1` |
| `results/dialam_model_eval/n256/records.jsonl` | `f88858ad00d790de1161af7a8332a46f6cb9c4718832d89d6e49cd29d0da4b80` |
| `results/dialam_model_eval/n256/judge_transcripts.jsonl` | `81419ea5ebb4a7f8c4a8e9aff5044b6367ab3cffe8e250564f3fbfec79b3269f` |
| `results/dialam_model_eval/n256/summary.json` | `605a38cb217d4e399899d0a098df483d6c370640fd209a5ca8a4d8d4af91793b` |
| `artifacts/dialam_qlora/n256/training_manifest.json` | `388d91b294d6d497290766a15887a5275d45406ff1401c712403c4164d0327d4` |
| `artifacts/dialam_qlora/n256/adapter/adapter_config.json` | `199241f92e02f15f15dce287b7447dfc760060ef6d1230c8b008bb600482c4c1` |
| `artifacts/dialam_qlora/n256/adapter/adapter_model.safetensors` | `c29156f0eeec3e609319dc05fd3cd55b4baa389bdef03d55696f3bdf9bd44613` |

## Reproduction commands

```bash
uv run python scripts/freeze_dialam_gate.py
uv run python scripts/build_dialam_training.py
uv run python scripts/prepare_dialam_hf_dataset.py

modal run scripts/modal_dialam_qlora.py \
  --action evaluate --target base --size 256 \
  --output-path results/dialam_model_generation/base/predictions.jsonl

uv run python scripts/evaluate_dialam_model.py \
  --predictions results/dialam_model_generation/base/predictions.jsonl \
  --output-dir results/dialam_model_eval/base \
  --approval APPROVE_DIALAM_MODEL_JUDGING

modal run scripts/modal_dialam_qlora.py \
  --action train --size 256 \
  --output-path artifacts/dialam_qlora/n256/remote_training_result.json

modal run scripts/modal_dialam_qlora.py \
  --action evaluate --target tuned --size 256 \
  --output-path results/dialam_model_generation/n256/predictions.jsonl

uv run python scripts/evaluate_dialam_model.py \
  --predictions results/dialam_model_generation/n256/predictions.jsonl \
  --output-dir results/dialam_model_eval/n256 \
  --approval APPROVE_DIALAM_MODEL_JUDGING
```

Text-bearing QT30 training and evaluation artifacts remain private. The
publishable metadata-only dataset repository structure is under
`hf_dataset/dialam_patch/` and contains deterministic reconstruction scripts,
schemas, statistics, manifests, and provenance without corpus text.

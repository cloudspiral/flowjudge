# Training-log evidence

These are the complete stdout logs and machine-readable manifests from the
four required Qwen3 0.6B v1 data-efficiency checkpoints and the subsequent
error-driven checks. Checkpoint weights remain ignored locally and the best
n=96 adapter is fused into `artifacts/publish/flowjudge-qwen3-0.6b/` for public
Hugging Face upload.

| Prefix | Purpose |
| --- | --- |
| `n12`, `n24`, `n48`, `n96` | Required nested v1 efficiency curve |
| `v2_attachment_n120` | Rejected ID-remap data revision |
| `v2_curriculum_n120` | Multi-edge curriculum data revision |
| `qwen3_1.7b_v2` | Rejected model-capacity check |

All runs use the hyperparameters recorded in the corresponding manifest; no
API keys, prompts from held-out evaluation, or model responses appear in these
training logs.

# FlowJudge publication readiness

## Locally verified model artifact

- Best checkpoint: Qwen3 0.6B, v1 n=96 QLoRA adapter.
- Fused directory: `artifacts/publish/flowjudge-qwen3-0.6b/`.
- Publication format: dequantized standard safetensors plus Qwen3 config,
  tokenizer, model card, training manifest, and verification JSON.
- Size: 1.1 GB.
- Independent runtime check: Transformers loaded `Qwen3ForCausalLM` on CPU,
  counted 596,049,920 parameters, and produced nonempty generation.

The local artifact is ignored because the same bytes belong in the public Hub
repository, not Git. Recreate and verify it with:

```bash
uv run --group mlx-train python scripts/fuse_mlx_for_hf.py \
  --model mlx-community/Qwen3-0.6B-4bit \
  --adapter artifacts/efficiency/n96/adapter \
  --output-dir artifacts/publish/flowjudge-qwen3-0.6b

uv run --group train python scripts/verify_hf_checkpoint.py \
  artifacts/publish/flowjudge-qwen3-0.6b
```

## Remaining user-owned publication step

No Hugging Face credential is configured (`hf auth whoami` reports not logged
in), and repository ownership/names cannot be inferred safely. After logging in,
choose public repository IDs and run:

```bash
uv run --group train python scripts/publish_hf.py \
  --model-dir artifacts/publish/flowjudge-qwen3-0.6b \
  --model-repo <your-hf-name>/flowjudge-qwen3-0.6b \
  --dataset-repo <your-hf-name>/flowjudge-v1
```

The script creates public repos, uploads the complete model plus the filtered
dataset and efficiency slices, and writes exact final model and dataset commit
hashes to `docs/publication_manifest.json`. It never prints the token.

## Other external blockers

- The fixed GPT-5.6 Sol judge cannot finish the remaining rubric rows until the
  OpenAI account has credit again.
- The staff-held-out set has not been supplied, so its required results cannot
  be fabricated or precomputed.
- The final three-to-five-minute video and live grader prompt are necessarily
  user/grader actions after publication.

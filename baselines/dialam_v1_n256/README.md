# DialAM v1 / n=256 baseline

This directory permanently identifies the first fixed-config DialAM QLoRA
baseline. It is the before-state for any later v2 dataset comparison.

- Baseline ID: `dialam-v1-n256`
- Canonical model: `Qwen/Qwen3-0.6B`
- Training runtime model: `unsloth/qwen3-0.6b-unsloth-bnb-4bit`
- Dataset: v1, 256-example exact prefix
- Frozen evaluation: 30 scenarios / 30 blocks
- Result: engineering smoke PASS; task reliability not achieved

`manifest.json` contains the aggregate metrics, fixed training configuration,
dataset mix, exact Hub revisions, model/config/checkpoint hashes, frozen
evaluation and rubric hashes, and the paths plus SHA-256 values of every private
artifact required to audit the run.

Private artifacts are intentionally not copied here. QT30-derived text,
candidate responses, judge transcripts, and adapter weights remain ignored
locally, with a second copy of the adapter on the named Modal Volume.

Verify the permanent metadata and, when the private files are present, every
private artifact hash:

```bash
uv run python scripts/verify_dialam_v1_n256_baseline.py
```

For a public checkout without private artifacts:

```bash
uv run python scripts/verify_dialam_v1_n256_baseline.py \
  --allow-missing-private
```

The human-readable analysis and full reproduction commands are in
[`docs/dialam_qlora_smoke_results.md`](../../docs/dialam_qlora_smoke_results.md).

# DialAM V9 mining interruption and resume proof

Decision: **PASS_BUILD_V9_CORPUS**.

The first invocation intentionally stopped only after atomically writing and
committing chunk 1 (128 candidates) to the persistent Modal volume. The next
invocation reused those 128 scores, scored only the remaining 95 chunks, and
completed all 12,288 fixed candidates.

A completed-state replay then reused all 96 chunks, created zero chunks, and
returned the identical semantic score hash. Canonical sorted-key JSONL export
was added after detecting that mixed in-memory and reloaded dictionaries could
serialize identical scores with different key order. The regenerated original
and replay files are now byte-for-byte identical with SHA-256
`406a704f8b58f2e54dd0cb02bd1a4926f8fd7d290bc4e4f240b842165fa4b479`.

The immutable run identity covers the input hash, exact V5.1 adapter hash,
candidate order, allowed labels, scoring rule, implementation version, chunk
size, batch size, and maximum sequence length. Identity drift refuses resume.

Candidate prompts and per-candidate scores remain local and ignored. Aggregate
counts, artifact paths, hashes, and Modal run IDs are recorded in
`reports/dialam_v9_mining_resume.json`.

## Reproduce

```bash
# Prove one durable chunk can survive termination.
modal run scripts/modal_dialam_qlora.py --action mine-v9 --stop-after-chunks 1

# Resume only missing chunks. Detach keeps the remote function alive if the
# local terminal disconnects.
modal run --detach scripts/modal_dialam_qlora.py --action mine-v9 \
  --output-path data/dialam/training/v9_mining_scores_n12288.jsonl

# A completed rerun must reuse all 96 chunks and write identical bytes.
modal run scripts/modal_dialam_qlora.py --action mine-v9 \
  --output-path data/dialam/training/v9_mining_scores_idempotent_n12288.jsonl

PYTHONPATH=src .venv/bin/python scripts/build_dialam_training_v9.py
```

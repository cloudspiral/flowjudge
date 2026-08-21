# FlowJudge training data

The training pool is derived only from VivesDebate Debates 1–7. Debates 8–10 are reserved for the project's leakage-safe evaluation set and must never be used for generation, filtering, prompt examples, or training.

`candidates.jsonl` contains deterministic source-derived windows anchored on opposite-stance VivesDebate conflict annotations. The raw `ADU_EN` text is intentionally not a training artifact yet: the frontier teacher rewrites it into fluent English while preserving IDs, sides, atomic boundaries, meaning, and fixed gold edges. A separate strict filter accepts only rewrites that preserve meaning and edge support, add no label-revealing response cues, and remain fluent and atomic.

Accepted chat-format examples are materialized as four nested slices:

- `v1_n12.jsonl`
- `v1_n24.jsonl`
- `v1_n48.jsonl`
- `v1_n96.jsonl`

The approximately log2 spacing tests the low-data region efficiently. Every row in a smaller slice is also present in every larger slice, so dataset size is the only intended variable.

Generate the offline pool and leakage-safe own-eval set:

```bash
uv run flowjudge build-training-data --limit 115
```

Run frontier rewriting and filtering with the fixed strong model from `JUDGE_MODEL`:

```bash
uv run flowjudge distill-training-data --max-workers 4 --required-examples 96
```

The derived dataset remains subject to VivesDebate's CC BY-NC-SA 4.0 license. Raw generation/filter responses and complete SDK envelopes are retained under ignored `results/data_generation/` directories; `v1_manifest.json` identifies the exact retained dataset artifacts.

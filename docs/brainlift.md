# FlowJudge Brainlift

## Thesis

FlowJudge tests whether data can instill one narrow behavior in a small open model: distinguishing a direct cross-side response from mere topical proximity, same-side continuation, repetition, or an independent counterargument. The intended gain is not broad debate understanding; it is reliable sparse graph reconstruction in a strict JSON contract.

## Behavior specification

Given a chronological 6–12-unit debate excerpt, the model must return one bare JSON object whose `relations` array contains exactly every edge from a later opposing-side unit to an earlier unit that it directly answers, attacks, mitigates, or turns. It must return no prose, duplicate/unknown/forward edges, or edges based only on topical similarity, same-side extension, repetition/rephrase, or an independent counterargument.

## Why training rather than prompting

The best of six prompt-ceiling cells reached only 58.6% mean Spec adherence, 69.5% mean Robustness, and 15.6% exact graph match. It failed mainly by adding spurious response edges around same-side extensions, far below the pre-registered 95%/90% gate.

## Dataset design

- Source: VivesDebate v3, CC BY-NC-SA 4.0.
- Training debates: 1–7.
- Own evaluation debates: 8–10.
- Source labels: opposite-stance `CA` conflicts oriented from later ADU to earlier ADU.
- Teacher: fixed strong OpenAI model rewrites noisy source English without changing units or labels.
- Filter: a separate call checks semantic preservation, edge support, absence of label cues, fluency, and atomicity; deterministic validation independently enforces IDs, chronology, stance, schema, and graph direction.
- Efficiency sizes: 12, 24, 48, and 96 nested examples. This approximately log2-spaced sweep emphasizes the low-data regime while remaining feasible for four real QLoRA checkpoints.

## Model and training

The canonical base is `Qwen/Qwen3-0.6B`. The first local sweep uses the corresponding `mlx-community/Qwen3-0.6B-4bit` checkpoint; MLX-LM trains LoRA layers over that quantized model, which is QLoRA. Every checkpoint uses the same training hyperparameters and only dataset size changes. Hyperparameters and actual training logs are written beside each checkpoint.

## Evaluation

`eval.py --model <hf-repo-id> --eval-set <path>` automatically detects a PEFT or MLX-LM adapter, evaluates both its declared base and tuned adapter, and writes raw generations, full judge JSONL, deterministic graph metrics, rubric means, and a Markdown comparison table. Staff data uses the same one-row-per-`BenchmarkCase` JSONL format as `data/eval/own_eval.jsonl`.

## Results

On the nine-case own evaluation set, the untuned Qwen3 0.6B base produced 66.7% valid JSON and 0% edge F1. The best n=96 checkpoint produced 100% valid JSON and 16% edge F1, but 0% exact graph match. Smaller 12/24/48 checkpoints had 0% edge F1. Two error-driven v2 data revisions did not improve beyond 16%, and a Qwen3 1.7B capacity check was worse than its base.

No tested size reliably holds the full behavior, so the minimum viable dataset size is **not established at N ≤ 120**. This is a negative but falsifiable result: the data clearly instilled schema compliance and a small amount of response reconstruction, not reliable graph recovery.

## Limitations

The target graph follows the documented VivesDebate-to-FlowJudge mapping and does not claim to capture every philosophically plausible response. The corpus covers one debate topic and one source language context, and the own evaluation set is necessarily small. The final claim must therefore stay narrow: data taught the selected small model to reproduce this response-edge behavior more reliably on held-out debates.

## Reproduction

```bash
uv sync --frozen
uv run flowjudge build-training-data --limit 115
uv run flowjudge distill-training-data --max-workers 4 --required-examples 96
uv sync --frozen --group mlx-train
uv run --group mlx-train python scripts/run_efficiency_curve.py --backend mlx
uv run python eval.py --model <hf-repo-id> --eval-set data/eval/own_eval.jsonl
```

Publication is locally prepared but still needs Hugging Face authentication and repository IDs. Fixed-judge rows beyond n=24 need replenished OpenAI credits, and staff-held-out results need the grader-supplied JSONL.

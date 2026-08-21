# FlowJudge Brainlift

## Thesis

FlowJudge tests whether data can instill one narrow behavior in a small open model: distinguishing a direct cross-side response from mere topical proximity, same-side continuation, repetition, or an independent counterargument. The intended gain is not broad debate understanding; it is reliable sparse graph reconstruction in a strict JSON contract.

## Behavior specification

Given a chronological 6–12-unit debate excerpt, the model must return one bare JSON object whose `relations` array contains exactly every edge from a later opposing-side unit to an earlier unit that it directly answers, attacks, mitigates, or turns. It must return no prose, duplicate/unknown/forward edges, or edges based only on topical similarity, same-side extension, repetition/rephrase, or an independent counterargument.

## Why training rather than prompting

The official prompt-ceiling report will record the best of six model/prompt cells and the recurring graph error that survives it. The project proceeds only if no cell reaches the pre-registered 95% mean Spec-adherence and 90% mean Robustness gate.

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

Prompt-ceiling, base-versus-tuned, and data-efficiency numbers will be inserted from their machine-generated reports. The minimum viable dataset size is the smallest checkpoint that holds the improvement on both the own and staff-held-out sets without a recurring core failure.

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

The final submission must add the exact evaluation-code commit, public dataset revision, public Hugging Face model commit, complete efficiency table, and demo URL.

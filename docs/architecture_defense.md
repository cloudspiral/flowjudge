# FlowJudge architecture defense

## Behavior thesis

General-purpose models often recognize debate topics but do not reliably reconstruct the sparse, directed graph of which later opposing argument directly responds to which earlier argument. A small model can learn this narrow distinction from carefully filtered response and hard-negative examples.

## Falsifiable behavior

Given a chronological 6–12-unit debate excerpt, the model must return one bare JSON object whose `relations` array contains exactly every edge from a later opposing-side unit to an earlier unit that it directly answers, attacks, mitigates, or turns. It must return no prose, duplicate/unknown/forward edges, or edges based only on topical similarity, same-side extension, repetition/rephrase, or an independent counterargument.

## Evidence plan

1. Prompt ceiling: GPT-5.4 Mini and Claude Haiku 4.5, three prompts each, 32 scenarios per cell, fixed GPT-5.6 Sol rubric judge, and deterministic graph scoring.
2. Data: VivesDebate Debates 1–7 only, rewritten by a strong teacher and independently filtered without changing source-derived gold graphs.
3. Own evaluation: nine untouched excerpts from Debates 8–10, plus compatibility with grader-supplied `BenchmarkCase` JSONL.
4. Model: Qwen3 0.6B with 4-bit NF4 QLoRA and identical hyperparameters across 12/24/48/96-example checkpoints.
5. Proof: one-command base-versus-tuned evaluation, raw responses, raw judge JSONL with reasoning, deterministic behavioral checks, and a data-efficiency curve.

## Main risk

VivesDebate conflict annotations are a conservative proxy for direct response. They can omit plausible unannotated responses, so claims are limited to reproducing this documented mapping plus the two synthetic FlowJudge-specific cases. The leakage-safe training/evaluation split is by complete debate, unlike the earlier prompt-ablation split.

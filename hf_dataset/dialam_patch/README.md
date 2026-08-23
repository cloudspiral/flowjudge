---
pretty_name: FlowJudge DialAM Patch Reconstruction Artifact
license: other
task_categories:
  - text-generation
tags:
  - argument-mining
  - qlora
  - reproducibility
---

# FlowJudge DialAM patch reconstruction artifact

This publishable artifact documents a private educational transformation of the
English DialAM/QT30 corpus for incremental argument-graph patch prediction. It
contains **no raw or transformed QT30 dialogue text and no original QT30
episode, map, proposition, or example IDs**. Because IDs/labels-only
redistribution remains unclear, identifier-bearing fields and per-example
failure inventories are replaced by counts and source-file hashes. The artifact
contains only allowed source/manifests, aggregate statistics, JSON schemas,
frozen evaluation hashes, prompt-ceiling metrics, and deterministic
reconstruction code.

The private training slices contain 256, 512, 1024, and 2048 nested examples.
Every slice is an exact prefix of the next and uses an original-parent-episode
split: 24 episodes for training and six untouched episodes for evaluation.
The n=2048 slice contains 819 NONE, 410 SUPPORT, 307 ATTACK, 410 REPHRASE, and
102 mixed-label blocks.

All four v1 sizes and one hard-negative v2 n=2048 checkpoint were evaluated on
the same frozen 30-scenario set. V1 n=2048 was the best fixed-curve point. The
v2 data change added 1,024 difficult NONE blocks, but increased false edges and
failed its preregistered material-improvement criterion.

The preregistered v3 experiment materially improved the behavior and is now the
selected direction. Its private 4,096-row corpus contains 2,048 exact
same-update positive/NONE pairs. It reserves four of the former training
episodes for development, trains on the remaining 20, and keeps the original
six frozen evaluation episodes untouched. Per-example assistant-token loss
eliminates the output-length weighting mismatch documented after v2. V3 reached
43.3% exact patch accuracy, 35.0% edge F1, 0.300 false edges/update, and 0/6
NONE false-positive cases on the frozen set. It still does not clear the
original reliability bar. The selected public adapter is
[`mr-mc/flowjudge-dialam-qwen3-0.6b-v3-n4096`](https://huggingface.co/mr-mc/flowjudge-dialam-qwen3-0.6b-v3-n4096).
The text-free manifest, schema, aggregate result, and reconstruction
implementation are included here; the negative reliability finding is
preserved rather than presenting the model as production-ready.

## Behavior

Given one new proposition and a complete fixed-size comparison block of earlier
propositions from the same dialogue, return one bare JSON object containing all
direct SUPPORT, ATTACK, or REPHRASE relations to supplied IDs. Return an empty
list when none exists; do not emit indirect relations, invented IDs, or prose.

## Reconstruction

The `reconstruction/` directory contains the deterministic parser, filtering,
split, prompt, freeze, and nested-corpus build code. Reconstruction requires a
separately obtained official QT30 archive and permission appropriate to the
user's intended use. The archive itself and generated text-bearing JSONL files
are deliberately omitted.

The metadata directory records the exact source checksum, retention funnel,
episode split, class counts, nested-slice hashes, frozen 30-scenario evaluation
hashes, prompt-ceiling metrics, and judge-rubric fingerprint.

## Redistribution boundary

Project-use approval does not establish a general public redistribution license
for the underlying QT30 text. Consumers should obtain the official source and
review its terms themselves. This repository is a reproducibility and evidence
artifact, not a mirror of the corpus.

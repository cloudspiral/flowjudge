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
contains **no raw or transformed QT30 dialogue text**. It contains only allowed
source/manifests, IDs and aggregate statistics, JSON schemas, frozen evaluation
hashes, prompt-ceiling metrics, and deterministic reconstruction code.

The private training slices contain 256, 512, 1024, and 2048 nested examples.
Every slice is an exact prefix of the next and uses an original-parent-episode
split: 24 episodes for training and six untouched episodes for evaluation.
The n=2048 slice contains 819 NONE, 410 SUPPORT, 307 ATTACK, 410 REPHRASE, and
102 mixed-label blocks.

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

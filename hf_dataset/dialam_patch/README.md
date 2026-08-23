---
pretty_name: FlowJudge DialAM Incremental Argument Patches
license: other
language:
  - en
task_categories:
  - text-generation
  - text-classification
tags:
  - argument-mining
  - qlora
  - qt30
  - reproducibility
configs:
  - config_name: v1_curve
    data_files:
      - split: train_256
        path: data/train_v1_n256.jsonl
      - split: train_512
        path: data/train_v1_n512.jsonl
      - split: train_1024
        path: data/train_v1_n1024.jsonl
      - split: train_2048
        path: data/train_v1_n2048.jsonl
  - config_name: v2_hard_negative
    data_files:
      - split: train
        path: data/train_v2_n2048.jsonl
  - config_name: v3_paired_patch
    data_files:
      - split: train
        path: data/train_v3_n4096.jsonl
  - config_name: v5_pairwise
    data_files:
      - split: train
        path: data/train_v5_n8192.jsonl
  - config_name: evaluation
    data_files:
      - split: development
        path: data/development_eval_30.jsonl
      - split: test
        path: data/frozen_own_eval_30.jsonl
---

# FlowJudge DialAM incremental argument patches

This is the actual transformed dataset used to test whether a 0.6B open model
can learn one falsifiable behavior: given one new proposition and one complete
fixed-size block of earlier propositions from the same dialogue, emit every and
only direct `SUPPORT`, `ATTACK`, or `REPHRASE` edge as one bare JSON object.
An empty relation list is required when no direct edge exists.

The package includes the nested v1 data-efficiency curve, the v2 hard-negative
intervention, the v3 paired patch corpus, the selected v5/v5.1 pairwise pipeline,
the project-owned development and frozen evaluation sets, schemas, aggregate
reports, a chronological improvement chart, and deterministic reconstruction
code. The `evidence/` directory preserves every prompt-ceiling candidate/judge
record plus raw candidate and blinded-judge JSONL for the base, all four v1
checkpoints, v2, v3, and the promoted v5.1 result.

![FlowJudge DialAM experiment history](metadata/dialam_experiment_history.svg)

## Splits and leakage controls

All splits operate on original QT30 parent episodes, never individual DialAM
map IDs. V1/v2 use 24 training episodes and six frozen evaluation episodes.
V3/v5 reserve four of those former training episodes as a separate development
set, train on the remaining 20, and leave the same six frozen episodes
untouched. The 30-case development and 30-case frozen files are project-owned
evaluation scenarios; the grader's staff-held-out set is not included.

The v1 sizes are deterministic nested prefixes. V3 has 4,096 block-level rows
arranged as 2,048 exact same-update positive/NONE contrast pairs. V5 has 8,192
candidate-level rows arranged as 4,096 positive/NONE pairs: 1,366 ATTACK, 1,365
REPHRASE, 1,365 SUPPORT, and 4,096 NONE rows. V5 still supplies the complete
comparison block in every input; its fixed four-label predictions are assembled
deterministically into the external block-level JSON contract. V5.1 keeps the
same checkpoint and applies the preregistered fixed 3.0 NONE margin selected on
the separate development set.

## Source and permission

The source is English QT30 as distributed for DialAM-2024 by ARG-tech at the
University of Dundee. Exact archive URL, retrieval metadata, SHA-256 hashes,
format references, and known archive/API discrepancies are in
`metadata/source_manifest.json` and `metadata/DATA_AUDIT.md`.

The project owner directly attested on 2026-08-23 that their project-specific
permission includes redistribution. This repository therefore publishes the
curated transformed training/evaluation JSONL and original identifiers needed
for provenance. It does not claim a general QT30 license or grant rights beyond
this project. The official raw archive and extracted maps are deliberately not
mirrored; obtain those from ARG-tech.

## Reproduction

From the FlowJudge source repository, place the separately obtained official
archive at the documented input path and run:

```bash
uv sync --frozen
uv run python scripts/build_dialam_gate.py
uv run python scripts/validate_dialam_gate.py
uv run python scripts/build_dialam_training.py
uv run python scripts/build_dialam_training_v2.py
uv run python scripts/build_dialam_training_v3.py
uv run python scripts/build_dialam_training_v5.py
uv run python scripts/prepare_dialam_hf_dataset.py
```

Every published file is inventoried with its byte count and SHA-256 hash in
`publish_manifest.json`. The manifests record the seed, exact episode split,
class mix, filter funnel, frozen eval/rubric hashes, training configuration,
checkpoint evidence, and all preregistered experiment decisions.

## Limitations

The selected v5.1 result reaches 53.3% exact patches, 47.6% edge F1, and 47.4%
relation macro-F1 on the reused 30-scenario frozen benchmark. It passes the
project's preregistered v3-to-v5.1 promotion gate but not the original high
reliability bar; it is not production-ready.

The relation labels inherit QT30's annotations and the documented grounding,
chronology, and binary-relation filters. The frozen set has only 30 scenarios,
so small absolute changes produce visibly large percentage changes. Failed
experiments remain in the history to prevent selective reporting. Consult the
latest result and promotion decision in `metadata/dialam_v5_1_results.md`; do not
interpret publication as a claim that the selected model is production-ready.

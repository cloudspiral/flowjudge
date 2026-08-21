---
license: other
license_name: cc-by-nc-sa-4.0
task_categories:
  - text-generation
language:
  - en
tags:
  - argument-mining
  - debate
  - response-graph
---

# FlowJudge distilled response-graph dataset

This dataset teaches one narrow behavior: return every direct response edge
from a later opposing-side argumentative discourse unit to the earlier unit it
answers, attacks, mitigates, or turns, while excluding topical similarity,
same-side extension, repetition, rephrase, and independent counterarguments.

The examples derive from VivesDebate v3 Debates 1–7. Gold edges are mapped from
opposite-stance conflict annotations, then a fixed strong teacher rewrites the
source excerpts into concise English and a separate strict pass filters meaning,
edge support, label leakage, and contextual clarity. Every published example
passed deterministic schema, chronology, stance, and graph-direction checks.

`train.jsonl` contains the complete 96-example v1 set. The `efficiency/`
directory contains nested 12-, 24-, and 48-example prefixes used with the same
training configuration. Debates 8–10 are excluded for evaluation.

Because this is a derivative of VivesDebate, it is distributed under
CC BY-NC-SA 4.0. See the repository mapping documentation for full provenance
and limitations.

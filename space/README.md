---
title: FlowJudge DialAM Patch Demo
emoji: ⚖️
colorFrom: blue
colorTo: indigo
sdk: static
app_file: index.html
pinned: false
---

# FlowJudge DialAM patch demo

This Space compares the untouched Qwen3-0.6B base with the selected public
FlowJudge DialAM v5.1/n=8192 pairwise QLoRA adapter on exactly the same
user-supplied incremental patch input. Both systems use fixed four-label
candidate scoring and the preregistered 3.0 NONE margin. V5.1 passed every
reused-benchmark promotion check, improving frozen exact-patch accuracy to
53.3% and edge F1 to 47.6%, but it remains a research artifact because it did
not clear the project's original high reliability threshold.

The static interface calls a scale-to-zero Modal L4 endpoint that loads the
public adapter. Cold starts can take about a minute. No QT30-derived dialogue
text is bundled with the demo.

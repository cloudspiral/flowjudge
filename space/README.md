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
FlowJudge DialAM v3/n=4096 paired-loss QLoRA adapter on exactly the same
user-supplied incremental patch input. V3 materially improved held-out edge F1,
exact-patch accuracy, and false-edge calibration over v1 and v2, but it remains
a research artifact because it did not clear the project's frozen reliability
threshold.

The static interface calls a scale-to-zero Modal CPU endpoint that loads the
public adapter. Cold starts can take about a minute. No QT30-derived dialogue
text is bundled with the demo.

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
FlowJudge DialAM v1/n=2048 QLoRA adapter on exactly the same user-supplied
incremental patch input. The checkpoint is a research artifact: it improved the
strict output contract and held-out edge F1 over the base, but did not clear the
project's frozen reliability threshold.

The static interface calls a scale-to-zero Modal CPU endpoint that loads the
public adapter. Cold starts can take about a minute. No QT30-derived dialogue
text is bundled with the demo.

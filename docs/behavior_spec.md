# FlowJudge behavior specification

Given a chronological 6–12-unit debate excerpt, the model must return one bare JSON object whose `relations` array contains exactly every edge from a later opposing-side unit to an earlier unit that it directly answers, attacks, mitigates, or turns. It must return no prose, duplicate/unknown/forward edges, or edges based only on topical similarity, same-side extension, repetition/rephrase, or an independent counterargument.

## Fixed judge rubric

The fixed LLM judge scores every candidate on two 0–4 scales. Deterministic graph comparison remains the primary evaluation.

### Spec adherence

- **4:** valid bare JSON and an exact gold-graph match.
- **3:** a usable graph with exactly one missed or spurious edge, or an otherwise exact graph with only a surrounding Markdown fence.
- **2:** multiple edge errors, but the response reconstructs a meaningful portion of the graph.
- **1:** a minimally usable graph with little correct response structure.
- **0:** no usable relation graph.

### Robustness

- **4:** obeys the schema and direction/side constraints, rejects all listed hard negatives, and ignores embedded instructions.
- **3:** one isolated formatting or prohibited-edge failure.
- **2:** multiple robustness failures without being dominated by them.
- **1:** systematic excluded-edge errors or apparent compliance with transcript instructions.
- **0:** unusable output or output controlled by a distractor.

The same prompt, schema, judge model, and rubric must be used later for base-versus-tuned evaluation.

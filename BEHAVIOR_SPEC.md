# Incremental argument-graph patching behavior specification

## Required behavior

Given one new proposition and a complete comparison block of earlier
propositions from the same dialogue, return exactly one JSON object containing
all direct `SUPPORT`, `ATTACK`, or `REPHRASE` relations from the new proposition
to supplied earlier proposition node IDs.

Return an empty `relations` list when the new proposition has no direct relation
to any proposition in the supplied block. Never emit indirect or transitive
relations, invented node IDs, relations to propositions outside the supplied
block, duplicate relations, Markdown, explanations, or any other prose.

## Output contract

The response must be one bare JSON object of this form:

```json
{
  "relations": [
    {
      "source": "<new proposition ID>",
      "target": "<supplied earlier proposition ID>",
      "type": "SUPPORT"
    }
  ]
}
```

Constraints:

- `relations` is always present and is a JSON array.
- Every relation's `source` is exactly the supplied new proposition ID.
- Every relation's `target` is an ID in the supplied comparison block.
- `type` is exactly one of `SUPPORT`, `ATTACK`, or `REPHRASE`.
- Each direct gold relation to a proposition in the block appears exactly once.
- No relation is inferred merely because two propositions are topically
  related or because a path between them exists elsewhere in the graph.
- With no qualifying direct relation, the exact semantic result is
  `{"relations": []}`.

## Incremental and block semantics

Each evaluation scenario represents one graph update. The new proposition is
compared only with earlier propositions from its own dialogue. A dialogue's
earlier propositions may be partitioned into deterministic fixed-size blocks;
the union of those blocks must cover every earlier proposition exactly once.
Each block is scored independently, and the complete patch for the update is
the union of its block-level relation lists.

The canonical direction is always from the new proposition to an earlier
proposition. Source-corpus graph direction must be preserved separately so any
normalization into this incremental direction remains auditable.

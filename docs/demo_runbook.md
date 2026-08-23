# FlowJudge DialAM three-to-five-minute demo runbook

1. Show `BEHAVIOR_SPEC.md`: one new proposition plus earlier candidates in,
   one direct SUPPORT/ATTACK/REPHRASE JSON patch out.
2. Show the Mini/Haiku 2x3 prompt-ceiling table and name the recurring failure:
   plausible topical similarity was mistaken for a direct edge.
3. Show the parent-episode split and nested v1 256/512/1024/2048 data curve.
4. Run the prescribed `eval.py --model ... --eval-set ...` command or show its
   completed base-versus-tuned table and per-example judge JSONL.
5. Open the public DialAM demo. Paste the grader-supplied live update, run the
   selected v3/n=4096 adapter, and compare its patch with the untouched base on
   exactly the same input:
   `https://huggingface.co/spaces/mr-mc/flowjudge-dialam-demo`.
6. Show the failed v2 comparison: edge F1 rose slightly, but false edges and
   robustness worsened because hard negatives were usually not paired with
   their positive sibling in the selected slice.
7. Show how v3 tested that diagnosis with 2,048 exact positive/NONE pairs and
   equal per-example loss. On the unchanged frozen set it reached 43.3% exact,
   35.0% edge F1, 0.300 false edges/update, and 0/6 false-positive NONE cases.
8. Close with the honest result: v3 materially improved the learned behavior
   and fixed much of the false-positive problem, but it still misses too many
   direct edges to establish reliability or a minimum viable N.

Before recording, verify that the model, dataset, and demo are public in a
logged-out browser and that the demo finishes loading the adapter.

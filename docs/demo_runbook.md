# FlowJudge DialAM three-to-five-minute demo runbook

1. Show `BEHAVIOR_SPEC.md`: one new proposition plus earlier candidates in,
   one direct SUPPORT/ATTACK/REPHRASE JSON patch out.
2. Show the Mini/Haiku 2x3 prompt-ceiling table and name the recurring failure:
   plausible topical similarity was mistaken for a direct edge.
3. Show the parent-episode split and nested 256/512/1024/2048 data curve.
4. Run the prescribed `eval.py --model ... --eval-set ...` command or show its
   completed base-versus-tuned table and per-example judge JSONL.
5. Open the public DialAM demo. Paste the grader-supplied live update, run the
   selected v1/n=2048 adapter, and compare its patch with the untouched base on
   exactly the same input:
   `https://huggingface.co/spaces/mr-mc/flowjudge-dialam-demo`.
6. Show the failed v2 comparison: edge F1 rose slightly, but false edges and
   robustness worsened because hard negatives were usually not paired with
   their positive sibling in the selected slice.
7. Close with the honest result: training taught strict JSON and improved edge
   F1 over the base, but did not establish reliable behavior or a minimum
   viable N at or below 2048.

Before recording, verify that the model, dataset, and demo are public in a
logged-out browser and that the demo finishes loading the adapter.

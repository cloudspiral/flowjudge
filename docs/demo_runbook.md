# FlowJudge DialAM three-to-five-minute demo runbook

1. Show `BEHAVIOR_SPEC.md`: one new proposition plus earlier candidates in,
   one direct SUPPORT/ATTACK/REPHRASE JSON patch out.
2. Show the Mini/Haiku 2x3 prompt-ceiling table and name the recurring failure:
   plausible topical similarity was mistaken for a direct edge.
3. Show the parent-episode split and nested v1 256/512/1024/2048 data curve.
4. Run the prescribed `eval.py --model ... --eval-set ...` command or show its
   completed base-versus-tuned table and per-example judge JSONL.
5. Open the public DialAM demo. Paste the grader-supplied live update, run the
   selected v5.1/n=8192 adapter, and compare its patch with the untouched base on
   exactly the same input:
   `https://huggingface.co/spaces/mr-mc/flowjudge-dialam-demo`.
6. Show the failed v2 comparison: edge F1 rose slightly, but false edges and
   robustness worsened because hard negatives were usually not paired with
   their positive sibling in the selected slice.
7. Show how v3 tested that diagnosis with 2,048 exact positive/NONE pairs and
   equal per-example loss. On the unchanged frozen set it reached 43.3% exact,
   35.0% edge F1, 0.300 false edges/update, and 0/6 false-positive NONE cases.
8. Show the v5 formulation change: one fixed four-label decision per supplied
   candidate, followed by deterministic patch assembly. Raw v5 improved recall
   but overpredicted edges because the paired corpus was 50% positive while the
   natural candidate pool was 7.379% positive.
9. Show the preregistered v5.1 3.0 NONE-margin correction and the improvement
   chart. On the unchanged frozen set it reached 53.3% exact, 47.6% edge F1,
   47.4% macro-F1, 0.267 false edges/update, and 0/6 false-positive NONE cases.
10. Close with the honest result: v5.1 passed every promotion check and is the
    selected artifact, but it still misses the original high reliability bar;
    no tested N can be called a reliable minimum.

Before recording, verify that the model, dataset, and demo are public in a
logged-out browser and that the demo finishes loading the adapter.

# FlowJudge three-to-five-minute demo runbook

1. Show `docs/behavior_spec.md` and state the pass/fail rule in one sentence.
2. Show the prompt-ceiling 2×3 table and name the recurring same-side-extension
   false-positive failure.
3. Show the filtered-data report and the debate-level Debates 1–7 versus 8–10
   split.
4. Run the prescribed `eval.py --model ... --eval-set ...` command or show its
   completed base-versus-tuned table and raw judge JSONL.
5. Open the public Gradio Space. Paste the grader-supplied live transcript, run
   the tuned model, and compare its graph with the base model using the same
   transcript.
6. Close with the honest result: training improved format reliability and gave
   a small nonzero edge-F1 gain, but did not establish reliable behavior or a
   minimum viable N.

Before recording, verify that the model, dataset, and Space are public in a
logged-out browser and that the Space finishes loading the model.

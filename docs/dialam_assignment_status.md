# DialAM assignment status

Status is reconciled against `Train_Your_Own_Small_Learning_Model.pdf`; the PDF
remains authoritative.

## Complete

- Frozen Behavior Spec, prompt-ceiling matrix, reliability gate, and raw local
  evidence.
- Leakage-safe QT30 parser, audit, validators, deterministic metrics, and
  reconstruction scripts.
- Untouched Qwen3-0.6B base evaluation and four fixed QLoRA checkpoints at
  N=256/512/1024/2048 on the same 30-scenario held-out set.
- Complete fixed-v1 performance-versus-N curve and one controlled v2
  hard-negative experiment.
- Preregistered v3 paired-loss experiment at N=4096: a 20-train / 4-development
  / 6-frozen parent-episode split, 2,048 exact same-update positive/NONE pairs,
  per-example assistant-token loss, one development pass, and one unchanged
  frozen-evaluation pass.
- V3 material-improvement PASS: 43.3% exact patch accuracy, 35.0% edge F1,
  33.9% macro-F1, 0.300 false edges/update, and 0/6 NONE cases with a false
  edge. It is the selected direction but does not clear the original frozen
  reliability bar.
- Public selected model checkpoint and exact Hub commit:
  `mr-mc/flowjudge-dialam-qwen3-0.6b-v3-n4096@56371373be622ea997c5723ceebf35af27cb5711`.
- Public text-free dataset/reconstruction artifact and exact Hub commit:
  `mr-mc/flowjudge-dialam-reconstruction-v3@e1dcc6834de431505fc8301a00d950f28506498e`.
  Automated checks exclude both QT30 text and original identifiers. The earlier
  identifier-bearing repository was made private rather than destructively
  deleted, preserving recovery while removing its history from public access.
- Required `eval.py --model <hf-repo-id> --eval-set <path>` interface, including
  DialAM schema auto-detection, base-versus-tuned generation, block unioning,
  deterministic correctness metrics, and blinded frozen-judge transcripts.
- DialAM Brainlift with the behavior thesis, fixed curve, v1-to-v2-to-v3
  evidence, minimum-viable-N finding, failure diagnosis, and exact public
  artifact commits.
- Public base-versus-tuned inference demo at
  `mr-mc/flowjudge-dialam-demo@b34685eab4b02044d62dbfdf4c3ab244281a179c`,
  anonymously verified in `RUNNING` state with a successful synthetic v3
  inference. Its public repository contains only `README.md` and `index.html`.
- Raw QT30-derived training/evaluation text, predictions, records, and judge
  transcripts remain ignored locally; their paths and hashes are preserved in
  the aggregate v3 report.
- Exact evaluation/release-code commit:
  `f881a6b5c0b4fd537b0be34d8543626053e2bdd7`.

## Still required before final submission

- Run the same harness on the staff-held-out JSONL when staff provides it; this
  result cannot be produced in advance.
- Record the required three-to-five-minute demo video after the public demo and
  final evaluator are verified.

## Honest acceptance finding

The selected v3 model beats the untouched base and both earlier data strategies,
satisfies the strict JSON contract, and materially fixes the dominant
false-positive SUPPORT/NONE failure. It still misses too many true relations,
especially ATTACK edges, so no tested run reliably holds semantic edge
selection and minimum viable N is not established. The assignment explicitly
calls for genuine results even when imperfect; the submission should preserve
that conclusion rather than reinterpret the frozen reliability threshold.

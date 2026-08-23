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
  edge. It was the previous selected direction.
- Preregistered v5 pairwise-classification training at N=8192 and separately
  preregistered v5.1 prior calibration. Raw v5 exposed the expected 50.0%
  training-versus-7.379%-natural positive-prior mismatch; the fixed 3.0 NONE
  margin passed every development check before one frozen evaluation was run.
- V5.1 frozen promotion PASS: 53.3% exact patch accuracy, 47.6% edge F1,
  47.4% macro-F1, 30.8% ATTACK F1, 0.267 false edges/update, 0/6 NONE cases
  with a false edge, and 3.03/4 judge Robustness. It is the selected direction
  but does not clear the original high reliability bar.
- Permission-cleared Hugging Face model and transformed-dataset packages are
  complete locally; exact public Hub commits are recorded after upload in
  `docs/dialam_hf_publication_manifest.json`.
- Required `eval.py --model <hf-repo-id> --eval-set <path>` interface, including
  DialAM schema auto-detection, v5.1 fixed-label likelihood scoring, block
  unioning, deterministic correctness metrics, and blinded judge transcripts.
- DialAM Brainlift with the behavior thesis, fixed curve, v1-to-v5.1
  evidence, minimum-viable-N finding, failure diagnosis, and exact public
  artifact commits.
- Public base-versus-tuned inference demo at
  `mr-mc/flowjudge-dialam-demo@b34685eab4b02044d62dbfdf4c3ab244281a179c`,
  anonymously verified in `RUNNING` state with a successful synthetic v3
  inference. Its public repository contains only `README.md` and `index.html`.
- The official QT30 raw archive/maps remain ignored. Under the owner's explicit
  project-specific redistribution permission, the vetted HF dataset package
  includes transformed training/evaluation JSONL and candidate/judge evidence,
  with paths and hashes in its publication manifest.
- Exact evaluation/release-code commit:
  `f881a6b5c0b4fd537b0be34d8543626053e2bdd7`.
- V4 class-balanced rehearsal experiment at N=8192. It
  improved episode-disjoint development edge F1 from 21.1% to 34.1% and ATTACK
  F1 from 0% to 54.5%, but worsened NONE cases with a false edge from 2/6 to
  4/6. It failed its preregistered development gate, so the reused frozen set
  and judge were not run; the failed point remains visible in the ledger.

## Still required before final submission

- Run the same harness on the staff-held-out JSONL when staff provides it; this
  result cannot be produced in advance.
- Record the required three-to-five-minute demo video after the public demo and
  final evaluator are verified.

## Honest acceptance finding

The selected v5.1 pipeline beats the untouched base and every earlier trained
strategy, satisfies the strict JSON contract, improves every primary frozen
semantic metric over v3, and preserves zero false-positive NONE cases. The
formulation and prior correction mattered more than blindly adding rows. It
still misses too many true relations to clear the original reliability bar, so
minimum reliable N is not established. The assignment calls for genuine
results even when imperfect; the submission should report both the real v3 to
v5.1 gain and the remaining reliability shortfall.

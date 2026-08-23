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
- Performance-versus-N curve, honest finding that no tested N is reliable, and
  one controlled error-driven v2 data experiment.
- Public selected model checkpoint and exact Hub commit:
  `mr-mc/flowjudge-dialam-qwen3-0.6b-v1-n2048@18ee7ee16a48159e8a18997c4719c5dd87d54a6f`.
- Public text-free dataset/reconstruction artifact and exact Hub commit:
  `mr-mc/flowjudge-dialam-reconstruction@649950879893bfcc1b5b6fb53ec5feff77ab3e66`.
- Required `eval.py --model <hf-repo-id> --eval-set <path>` interface, including
  DialAM schema auto-detection, base-versus-tuned generation, block unioning,
  deterministic correctness metrics, and blinded frozen-judge transcripts.
- DialAM Brainlift with the behavior thesis, curve, minimum-viable-N finding,
  v1-to-v2 evidence, failure diagnosis, and exact public artifact commits.
- Public base-versus-tuned inference demo at
  `mr-mc/flowjudge-dialam-demo@13edded07c43faf456c08462039c37c56efda26c`,
  anonymously verified in `RUNNING` state with a successful synthetic inference.

## Still required before final submission

- Record and pin the Git commit containing the final evaluation entrypoint.
- Run the same harness on the staff-held-out JSONL when staff provides it; this
  result cannot be produced in advance.
- Record the required three-to-five-minute demo video after the public demo and
  final evaluator are verified.

## Honest acceptance finding

The tuned model beats the untouched base numerically and satisfies the strict
JSON contract, but no tested N reliably holds semantic edge selection. The
assignment explicitly calls for genuine results even when unimpressive, so the
submission should preserve this negative conclusion rather than substitute an
unregistered model or reinterpret the frozen reliability threshold.

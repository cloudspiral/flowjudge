# Pre-video submission checkpoint — 2026-08-23

This is the permanent, publishable checkpoint immediately before recording the
required final demo video. The annotated Git tag
`submission-ready-v5.1-pre-video-20260823` identifies the checkpoint commit.

## Frozen release state

- Selected pipeline: DialAM v5.1 pairwise four-label scoring with the fixed
  `3.0` NONE margin.
- Evaluation/release source used for the reported result:
  `01b68aed7918c919a681ba881c31c9e7d3a9328d`.
- Pre-checkpoint publication commit:
  `92cb2bbdb0f5bef3ddeed175a0be387b6630c71b`.
- Public model:
  `mr-mc/flowjudge-dialam-qwen3-0.6b-v5-1-n8192@888f710037af43d6965c75163b240c785dee3cad`.
- Public dataset:
  `mr-mc/flowjudge-dialam@920617aa6e8a9780e3f5db9399eaa859ae6b16b0`.
- Public demo:
  `mr-mc/flowjudge-dialam-demo@d540bc489db5fc5dc3c51e620222a93400ddddf4`.

## Verification at checkpoint time

- The Git worktree was clean and `origin/main` resolved to the pre-checkpoint
  publication commit above.
- The complete test suite passed: 81 collected tests, 81 passed.
- `eval.py --help` confirmed the required
  `--model <hf-repo-id> --eval-set <path>` interface and automatic DialAM
  schema handling.
- Anonymous requests to the exact GitHub, Hugging Face model, dataset, and
  Space revisions returned HTTP 200.
- The scale-to-zero Modal demo backend returned HTTP 200 from `/health` and
  identified the selected public model.

## Frozen result

The selected v5.1 checkpoint achieved 53.3% exact patch accuracy, 47.6% edge
F1, 47.4% relation macro-F1, 0.267 false edges per update, zero false-positive
NONE cases out of six, and 3.03/4 judge Robustness on the frozen 30-scenario
evaluation. It beats the untouched base and every earlier trained strategy,
but does not clear the preregistered high reliability bar. The minimum reliable
dataset size therefore remains not established rather than being overstated.

## Remaining submission actions

- Record and submit the required three-to-five-minute demo video, including a
  live unseen or grader-supplied example against both base and tuned models.
- If staff supplies its hidden JSONL, run the unchanged evaluation harness on
  that file. Those scores cannot be computed before the file is provided.

Any experiment after this tag is exploratory v6 work and must not rewrite or
silently replace this submission checkpoint.

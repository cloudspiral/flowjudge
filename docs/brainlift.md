# FlowJudge DialAM Brainlift

## Behavior thesis

This project tests whether a 0.6B-parameter model can learn one narrow,
grounded graph-editing behavior from data: given one new proposition and one
complete fixed-size block of earlier propositions from the same dialogue,
return exactly the direct `SUPPORT`, `ATTACK`, and `REPHRASE` edges from the new
proposition to supplied IDs. The output must be one bare JSON object, with an
empty relation list when no direct edge exists. Indirect edges, topical
similarity, invented IDs, and prose are errors.

The behavior matters because incremental graph patching is easier to verify and
compose than asking a model to regenerate an entire argument map after every
turn. Its hardest requirement is calibrated sparsity: most semantically related
proposition pairs are not directly connected.

## Why prompting was not enough

The frozen prompt-ceiling gate evaluated 30 held-out update scenarios for all
six combinations of GPT-5.4 Mini / Claude Haiku 4.5 and zero-shot / few-shot /
strong-structured prompts. The best edge F1 was 34.3%, and no combination
cleared the preregistered reliability thresholds. GPT Mini produced valid JSON
but overpredicted edges, including at least one false edge on all six NONE
scenarios in its best cell. Haiku frequently wrapped the required bare object
in Markdown fences. The prompt-ceiling gate therefore passed: the behavior was
not already reliable under the required hosted-model baselines.

## Data

The private source is the English DialAM-2024/QT30 argument-map corpus. The
canonical parser preserves parent episode, map and proposition identity,
speaker, chronology, locution grounding, original RA/CA/MA labels, normalized
relation labels, and edge direction. Training keeps only direct,
chronology-resolvable, singly grounded, unambiguous binary relations.

Splitting is by original parent episode, not map ID. V1 and v2 use 24 training
episodes while six episodes remain untouched for the 30-scenario frozen
evaluation. The v1 training slices are nested deterministic prefixes at N=256,
512, 1024, and 2048. V3 removes four of those former training episodes for a
separate 30-case development set, trains on the remaining 20, and leaves the
same six frozen episodes untouched. Its 4,096 rows are 2,048 exact same-update
positive/NONE pairs: 2,048 NONE, 914 SUPPORT, 915 REPHRASE, 189 ATTACK, and 30
mixed-label blocks. QT30-derived text stays private; the public dataset artifact
contains only allowed aggregate metadata, IDs, schemas, hashes, and
deterministic reconstruction code.

## Model and fixed training method

The only base model is `Qwen/Qwen3-0.6B`. Every checkpoint uses the same Unsloth
4-bit QLoRA hyperparameters: LoRA rank 16, alpha 32, dropout 0, attention and
MLP projections, three epochs, effective batch size 8, learning rate `2e-4`,
cosine schedule, 5% warmup, `adamw_8bit`, prompt-token masking, and seed
`20260823`. Only N changes within the v1 efficiency curve. V2 changes negative
selection; v3 changes paired sampling and replaces the default token-weighted
reduction with a preregistered per-example assistant-token mean. Training,
saving, reloading, and deterministic generation ran on an NVIDIA L4 through
Modal.

## Results

The untouched base had 0% schema validity, 0% exact-patch accuracy, and 0% edge
F1. Fine-tuning clearly taught the output contract and some relation behavior,
but not reliable graph recovery.

| Run | Exact patch | Edge F1 | Relation macro-F1 | False edges/update | Robustness /4 |
|---|---:|---:|---:|---:|---:|
| Base | 0.0% | 0.0% | 0.0% | 0.000 | 1.40 |
| v1 / 256 | 16.7% | 9.5% | 5.6% | 0.533 | 2.47 |
| v1 / 512 | 16.7% | 9.5% | 6.3% | 0.533 | 2.17 |
| v1 / 1024 | 16.7% | 8.5% | 8.9% | 0.700 | 1.87 |
| v1 / 2048 | 26.7% | 16.7% | 16.7% | 0.667 | 2.10 |
| v2 hard-negative / 2048 | 23.3% | 21.4% | 18.8% | 0.867 | 1.43 |
| **v3 paired-loss / 4096** | **43.3%** | **35.0%** | **33.9%** | **0.300** | **2.97** |

Every tuned checkpoint reached 100% JSON and schema validity with zero invalid
IDs. V3 is selected because it clears every preregistered material-improvement
condition over v1/v2. It still misses the original 85% edge-F1, 80% exact-patch,
75% macro-F1, 0.2 false-edge, and 3.5 Robustness thresholds. Because the v3
data/loss intervention differs from the fixed v1 curve and no run is reliable,
the minimum viable dataset size remains **not established**.

## Error-driven v2, v3, and diagnosis

The dominant v1 error was false-positive SUPPORT prediction. A controlled v2
kept the model, N, configuration, seed, and evaluation fixed while raising
difficult NONE coverage to 1,024 blocks selected for lexical overlap and a true
positive sibling elsewhere in the update. Edge F1 rose from 16.7% to 21.4%, but
false edges increased from 0.667 to 0.867 per update, exact accuracy fell from
26.7% to 23.3%, and robustness fell from 2.10 to 1.43. It failed the
preregistered improvement rule and was rejected.

Post-run analysis found that only 166/1,024 v2 hard negatives were accompanied
by their positive sibling in the selected slice. Row balance also overstated
negative supervision: positive/mixed assistant targets contained 4.77 times as
many characters as all NONE targets combined. V3 directly tested that diagnosis
by pairing every hard NONE block with its exact same-update positive sibling and
averaging loss within each example before averaging the batch. Its separate
development set reached 21.1% edge F1 and 0.333 false edges/update; the single
frozen pass reached 35.0% edge F1 and 0.300 false edges/update. NONE cases with
false edges fell from 2/6 in v1 and 5/6 in v2 to 0/6.

The paired/loss-balanced intervention therefore fixed much of the dominant
false-positive SUPPORT calibration problem. The remaining primary failure is
conservative underprediction plus relation-label/target confusion: 17 gold
edges were missed on the frozen set, and ATTACK remains weakest at 18.2% F1.
Another learning-rate or epoch sweep is lower priority because it would not
address that semantic bottleneck. If work continues beyond the assignment, the
strongest next formulation is pairwise edge classification with a fixed-length
`SUPPORT|ATTACK|REPHRASE|NONE` target followed by deterministic patch assembly;
that should be registered as a new experiment, not folded into this result.

## Public artifacts and reproduction

- Model: [mr-mc/flowjudge-dialam-qwen3-0.6b-v3-n4096](https://huggingface.co/mr-mc/flowjudge-dialam-qwen3-0.6b-v3-n4096), commit `56371373be622ea997c5723ceebf35af27cb5711`.
- Dataset/reconstruction artifact: [mr-mc/flowjudge-dialam-reconstruction-v3](https://huggingface.co/datasets/mr-mc/flowjudge-dialam-reconstruction-v3), commit `e1dcc6834de431505fc8301a00d950f28506498e`. Its publication tests exclude original QT30 episode/map/proposition/example IDs as well as text.
- Live demo: [mr-mc/flowjudge-dialam-demo](https://huggingface.co/spaces/mr-mc/flowjudge-dialam-demo), static Space commit `b34685eab4b02044d62dbfdf4c3ab244281a179c`, backed by the scale-to-zero Modal CPU endpoint `https://cloudspiral--flowjudge-dialam-public-demo-web.modal.run`.
- Evaluation and v3 release code: Git commit `f881a6b5c0b4fd537b0be34d8543626053e2bdd7`.

The assignment-prescribed evaluator auto-detects DialAM `PatchExample` JSONL,
rejects parent-episode leakage, evaluates both the canonical base and adapter,
unions block predictions for update-level exact-patch metrics, uses deterministic
gold scoring for relation correctness, and uses the frozen blinded judge only
for Spec adherence and Robustness:

```bash
uv sync --group train
uv run python eval.py \
  --model mr-mc/flowjudge-dialam-qwen3-0.6b-v3-n4096 \
  --eval-set <dialam-patch-example-jsonl>
```

The staff-held-out set is intentionally unavailable before grading. The command
accepts it in the same `dialam_incremental_patch_v1` JSONL schema; its results
cannot honestly be precomputed here. Raw candidate and judge transcripts are
written under the ignored local `results/` tree.

## Conclusion

Data produced a real, measurable improvement over the untouched base and made
the strict output contract reliable. Paired examples plus per-example loss also
materially corrected false-edge overprediction, validating the post-v2
diagnosis. Direct relation selection is still not reliable, primarily because
of missed edges and weak ATTACK discrimination. The defensible submission claim
is a successful end-to-end specialization experiment with an informative
negative reliability finding, not a production-ready argument mapper.

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

Splitting is by original parent episode, not map ID: 24 episodes train and six
episodes remain untouched for the 30-scenario evaluation. The v1 training
slices are nested deterministic prefixes at N=256, 512, 1024, and 2048. The
largest slice contains 819 NONE, 410 SUPPORT, 307 ATTACK, 410 REPHRASE, and 102
mixed-label blocks. QT30-derived text stays private; the public dataset artifact
contains only aggregate metadata, schemas, hashes, and deterministic
reconstruction code.

## Model and fixed training method

The only base model is `Qwen/Qwen3-0.6B`. Every v1 checkpoint uses the same
Unsloth 4-bit QLoRA configuration: LoRA rank 16, alpha 32, dropout 0, attention
and MLP projections, three epochs, effective batch size 8, learning rate
`2e-4`, cosine schedule, 5% warmup, `adamw_8bit`, prompt-token masking, and seed
`20260823`. Only N changes. Training, saving, reloading, and deterministic
generation ran on an NVIDIA L4 through Modal.

## Results

The untouched base had 0% schema validity, 0% exact-patch accuracy, and 0% edge
F1. Fine-tuning clearly taught the output contract and some relation behavior,
but not reliable graph recovery.

| N | Exact patch | Edge F1 | Relation macro-F1 | False edges/update | Robustness /4 |
|---:|---:|---:|---:|---:|---:|
| Base | 0.0% | 0.0% | 0.0% | 0.000 | 1.40 |
| 256 | 16.7% | 9.5% | 5.6% | 0.533 | 2.47 |
| 512 | 16.7% | 9.5% | 6.3% | 0.533 | 2.17 |
| 1024 | 16.7% | 8.5% | 8.9% | 0.700 | 1.87 |
| 2048 | **26.7%** | **16.7%** | **16.7%** | 0.667 | 2.10 |

Every tuned checkpoint reached 100% JSON and schema validity with zero invalid
IDs. N=2048 is the selected checkpoint because it has the best v1 exact-patch,
edge-F1, and macro-F1 results. No tested N clears the frozen reliability bar,
so the minimum viable dataset size is **not established at N <= 2048**.

## Error-driven v2 and diagnosis

The dominant v1 error was false-positive SUPPORT prediction. A controlled v2
kept the model, N, configuration, seed, and evaluation fixed while raising
difficult NONE coverage to 1,024 blocks selected for lexical overlap and a true
positive sibling elsewhere in the update. Edge F1 rose from 16.7% to 21.4%, but
false edges increased from 0.667 to 0.867 per update, exact accuracy fell from
26.7% to 23.3%, and robustness fell from 2.10 to 1.43. It failed the
preregistered improvement rule and was rejected.

Post-run analysis found that only 166/1,024 v2 hard negatives were accompanied
by their positive sibling in the selected training slice. The model therefore
usually did not receive the intended matched contrast between "same update,
wrong block" and "same update, correct target block." The low training loss
(0.087 for v1 and 0.081 for v2) combined with weak held-out metrics also points
to data/formulation mismatch or annotation ambiguity rather than simple
undertraining.

If another training run is justified, the highest-value change is a paired
contrastive v3 dataset: include each difficult negative with its exact positive
sibling and preserve class balance at the update level. A learning-rate or
epoch sweep is lower priority because it would optimize already-low training
loss without fixing the missing contrast. A pairwise edge-classification
objective with deterministic block assembly is a stronger redesign, but it
should be treated as a new formulation rather than silently mixed into the
completed fixed experiment.

## Public artifacts and reproduction

- Model: [mr-mc/flowjudge-dialam-qwen3-0.6b-v1-n2048](https://huggingface.co/mr-mc/flowjudge-dialam-qwen3-0.6b-v1-n2048), commit `18ee7ee16a48159e8a18997c4719c5dd87d54a6f`.
- Dataset/reconstruction artifact: [mr-mc/flowjudge-dialam-reconstruction](https://huggingface.co/datasets/mr-mc/flowjudge-dialam-reconstruction), commit `649950879893bfcc1b5b6fb53ec5feff77ab3e66`.
- Live demo: [mr-mc/flowjudge-dialam-demo](https://huggingface.co/spaces/mr-mc/flowjudge-dialam-demo), static Space commit `13edded07c43faf456c08462039c37c56efda26c`, backed by a scale-to-zero Modal CPU endpoint.

The assignment-prescribed evaluator auto-detects DialAM `PatchExample` JSONL,
rejects parent-episode leakage, evaluates both the canonical base and adapter,
unions block predictions for update-level exact-patch metrics, uses deterministic
gold scoring for relation correctness, and uses the frozen blinded judge only
for Spec adherence and Robustness:

```bash
uv sync --group train
uv run python eval.py \
  --model mr-mc/flowjudge-dialam-qwen3-0.6b-v1-n2048 \
  --eval-set <dialam-patch-example-jsonl>
```

The staff-held-out set is intentionally unavailable before grading. The command
accepts it in the same `dialam_incremental_patch_v1` JSONL schema; its results
cannot honestly be precomputed here. Raw candidate and judge transcripts are
written under the ignored local `results/` tree.

## Conclusion

Data produced a real, measurable improvement over the untouched base and made
the strict output contract reliable. It did not make direct relation selection
reliable, and neither more v1 data nor unpaired topical hard negatives solved
false-edge overprediction. The defensible submission claim is a successful
end-to-end specialization experiment with a negative reliability finding and a
specific paired-contrastive next hypothesis, not a production-ready argument
mapper.

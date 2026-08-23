# FlowJudge DialAM publication readiness

## Selected artifact

- Base: `Qwen/Qwen3-0.6B`.
- Selected checkpoint: v5 QLoRA adapter trained on 8,192 candidate rows.
- Selected inference pipeline: v5.1 fixed four-label likelihood scoring with
  the preregistered 3.0 NONE margin and deterministic patch assembly.
- Frozen result: 53.3% exact patches, 47.6% edge F1, 47.4% macro-F1, 0.267
  false edges/update, 0/6 false-positive NONE cases, and 3.03/4 Robustness.
- Selection: all nine frozen promotion checks pass; the original high
  reliability bar does not.

The publishable local model package is built at
`artifacts/hf_publish/dialam-qwen3-0.6b-v5-1-n8192/`. Its manifest hashes every
adapter, tokenizer, calibration, result, and inference-config file. The model
package contains no raw or transformed QT30 text.

## Dataset artifact

The publishable dataset package is assembled at `hf_dataset/dialam_patch/`.
Under the project owner's explicit project-specific redistribution permission,
it includes the actual transformed v1/v2/v3/v5 training JSONL, development and
frozen evaluation JSONL, prompt-ceiling and base/tuned candidate/judge evidence,
schemas, manifests, reports, the improvement chart, and reconstruction code.

It deliberately excludes the official QT30 archive and extracted raw maps. The
package does not claim a general QT30 license. `publish_manifest.json` records
the permission basis, source-code commit, path, byte count, and SHA-256 for
every file.

## Reproduce and publish

```bash
.venv/bin/python scripts/build_dialam_v5_1_report.py
.venv/bin/python scripts/prepare_dialam_hf_model_v5_1.py
.venv/bin/python scripts/prepare_dialam_hf_dataset.py
.venv/bin/pytest

.venv/bin/python scripts/publish_dialam_hf.py \
  --model-repo mr-mc/flowjudge-dialam-qwen3-0.6b-v5-1-n8192 \
  --dataset-repo mr-mc/flowjudge-dialam

modal deploy scripts/modal_dialam_demo.py
.venv/bin/python scripts/publish_dialam_space.py
```

The public demo compares the untouched base with the selected adapter using the
same complete-block pairwise scorer and fixed margin. The static Hugging Face
Space calls the scale-to-zero Modal endpoint.

## Published revisions and smoke proof

- Model: `mr-mc/flowjudge-dialam-qwen3-0.6b-v5-1-n8192@888f710037af43d6965c75163b240c785dee3cad`.
- Dataset: `mr-mc/flowjudge-dialam@920617aa6e8a9780e3f5db9399eaa859ae6b16b0`.
- Static Space: `mr-mc/flowjudge-dialam-demo@d540bc489db5fc5dc3c51e620222a93400ddddf4`.
- Exact source commit embedded in both model/dataset manifests:
  `01b68aed7918c919a681ba881c31c9e7d3a9328d`.
- Anonymous model, dataset, Space, and exact-revision requests returned HTTP
  200. The synthetic public prediction returned schema-valid base/tuned JSON
  with HTTP 200 in 5.93 warm seconds.

## Remaining external submission actions

- Run the unchanged `eval.py --model ... --eval-set ...` harness on the
  staff-held-out JSONL when staff supplies it; that result cannot be precomputed.
- Record and submit the required three-to-five-minute video. The project owner
  has explicitly retained that task.

Exact Hub commits, demo smoke evidence, and the final Git release commit are
recorded in the publication/status manifests after upload.

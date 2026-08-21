---
license: other
license_name: cc-by-nc-sa-4.0
base_model: Qwen/Qwen3-0.6B
library_name: transformers
pipeline_tag: text-generation
tags:
  - qwen3
  - qlora
  - argument-mining
  - response-graph
---

# FlowJudge Qwen3 0.6B

FlowJudge is a QLoRA-tuned Qwen3 0.6B checkpoint for reconstructing direct
cross-side response edges from short, chronologically ordered debate excerpts.
It returns a bare JSON object with a `relations` array of later-to-earlier
`responds_to` edges.

The model was trained on a 4-bit MLX Qwen3 checkpoint with prompt masking, then
fused and dequantized into standard Hugging Face-compatible safetensors for
portable evaluation. Training data derives from VivesDebate v3 Debates 1–7 and
is subject to CC BY-NC-SA 4.0. Evaluation uses untouched Debates 8–10.

Use the exact project harness for base-versus-tuned behavioral scoring:

```bash
python eval.py --model <this-repo-id> --eval-set <BenchmarkCase-jsonl>
```

The checkpoint is specialized and should not be treated as a general debate
reasoner. Its gold graph follows the project’s documented conservative mapping
from VivesDebate conflict annotations.

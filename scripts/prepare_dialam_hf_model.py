#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from flowjudge.patch_data import PROJECT_ROOT


DEFAULT_CHECKPOINT = PROJECT_ROOT / "artifacts" / "dialam_qlora" / "v2_n2048" / "adapter"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "hf_publish" / "dialam-qwen3-0.6b-v2"
REPORT = PROJECT_ROOT / "reports" / "dialam_v1_to_v2.json"
DATA_MANIFEST = PROJECT_ROOT / "data" / "dialam" / "training" / "training_v2_manifest.json"
TRAINING_RECORD = (
    PROJECT_ROOT / "artifacts" / "dialam_qlora" / "v2_n2048" / "remote_training_result.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the material-improvement DialAM v2 PEFT adapter for Hugging Face"
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    if not report.get("material_improvement"):
        raise RuntimeError("v2 did not clear the preregistered material-improvement bar")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing publication directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    copied = []
    for source in sorted(path for path in args.checkpoint.iterdir() if path.is_file()):
        if source.name == "README.md":
            continue
        destination = args.output_dir / source.name
        shutil.copyfile(source, destination)
        copied.append(destination)
    for source, name in (
        (REPORT, "dialam_v1_to_v2.json"),
        (DATA_MANIFEST, "dialam_v2_data_manifest.json"),
        (TRAINING_RECORD, "dialam_v2_training_manifest.json"),
    ):
        destination = args.output_dir / name
        shutil.copyfile(source, destination)
        copied.append(destination)

    v1 = report["v1_n2048"]
    v2 = report["v2_n2048"]
    model_card = f"""---
base_model: Qwen/Qwen3-0.6B
library_name: peft
pipeline_tag: text-generation
license: other
tags:
  - qwen3
  - qlora
  - argument-mining
  - dialam
---

# Qwen3-0.6B DialAM incremental patch adapter v2

This is the QLoRA adapter for the narrow FlowJudge behavior: given one new
proposition and a complete block of earlier propositions, emit one bare JSON
object containing only direct SUPPORT, ATTACK, or REPHRASE edges to supplied
IDs, or an empty relation list.

The adapter uses Qwen/Qwen3-0.6B with LoRA r=16, alpha=32, three epochs, seed
20260823, and 2,048 private transformed QT30 training examples. The training
text is not redistributed. Consumers must obtain the official corpus and
appropriate permission separately.

## Frozen 30-scenario result

| Metric | v1 n=2048 | v2 n=2048 |
|---|---:|---:|
| Exact patch accuracy | {v1['exact_patch_accuracy'] * 100:.1f}% | {v2['exact_patch_accuracy'] * 100:.1f}% |
| Edge precision | {v1['edge_precision'] * 100:.1f}% | {v2['edge_precision'] * 100:.1f}% |
| Edge recall | {v1['edge_recall'] * 100:.1f}% | {v2['edge_recall'] * 100:.1f}% |
| Edge F1 | {v1['edge_f1'] * 100:.1f}% | {v2['edge_f1'] * 100:.1f}% |
| Relation macro-F1 | {v1['relation_macro_f1'] * 100:.1f}% | {v2['relation_macro_f1'] * 100:.1f}% |
| False edges/update | {v1['false_edges_per_update']:.3f} | {v2['false_edges_per_update']:.3f} |

The included manifests record the frozen eval/rubric hashes, deterministic
metrics, data mix, fixed training config, adapter hash, and private artifact
hashes. The evaluation is small and corpus-specific; this is not evidence of
general argument-mining reliability.

## Load

Load this PEFT adapter with a compatible Transformers/PEFT or Unsloth runtime.
The training runtime used the 4-bit `unsloth/qwen3-0.6b-unsloth-bnb-4bit`
mirror of the canonical `Qwen/Qwen3-0.6B` base.
"""
    readme = args.output_dir / "README.md"
    readme.write_text(model_card, encoding="utf-8")
    copied.append(readme)

    manifest = {
        "schema_version": "dialam_hf_model_package_v1",
        "prepared_at": datetime.now(UTC).isoformat(),
        "model": "Qwen3-0.6B DialAM v2 n=2048 QLoRA adapter",
        "contains_raw_or_transformed_qt30_text": False,
        "material_improvement_verified": True,
        "files": [
            {
                "path": path.name,
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(copied)
        ],
    }
    manifest_path = args.output_dir / "publish_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()

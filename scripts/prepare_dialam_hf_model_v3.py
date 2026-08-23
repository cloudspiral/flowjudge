#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from flowjudge.patch_data import PROJECT_ROOT


DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "artifacts" / "dialam_qlora" / "v3_n4096" / "adapter"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "artifacts" / "hf_publish" / "dialam-qwen3-0.6b-v3-n4096"
)
RESULT_REPORT = PROJECT_ROOT / "reports" / "dialam_v1_v2_v3.json"
DATA_MANIFEST = (
    PROJECT_ROOT / "data" / "dialam" / "training" / "training_v3_manifest.json"
)
PRIVATE_TRAINING_MANIFEST = (
    PROJECT_ROOT / "artifacts" / "dialam_qlora" / "v3_n4096" / "training_manifest.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the selected DialAM v3 n=4096 PEFT adapter for Hugging Face"
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    result = json.loads(RESULT_REPORT.read_text(encoding="utf-8"))
    data = json.loads(DATA_MANIFEST.read_text(encoding="utf-8"))
    private_training = json.loads(PRIVATE_TRAINING_MANIFEST.read_text(encoding="utf-8"))
    if not result.get("material_improvement"):
        raise RuntimeError("v3 did not clear its preregistered material-improvement rule")
    if result.get("decision") != "PROMOTE_V3_AS_FINAL_DIRECTION":
        raise RuntimeError("the v3 model-selection decision is missing or unexpected")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite publication directory: {args.output_dir}")
    if not args.checkpoint.is_dir():
        raise FileNotFoundError(args.checkpoint)
    args.output_dir.mkdir(parents=True)

    copied: list[Path] = []
    for source in sorted(path for path in args.checkpoint.iterdir() if path.is_file()):
        if source.name == "README.md":
            continue
        destination = args.output_dir / source.name
        shutil.copyfile(source, destination)
        copied.append(destination)
    for source, name in (
        (RESULT_REPORT, "dialam_v1_v2_v3_result.json"),
        (DATA_MANIFEST, "dialam_v3_data_manifest.json"),
    ):
        destination = args.output_dir / name
        shutil.copyfile(source, destination)
        copied.append(destination)

    # The private remote manifest contains a reload probe with original QT30 IDs.
    # Publish only aggregate runtime/config evidence plus the private manifest hash.
    public_training = {
        key: private_training[key]
        for key in (
            "schema_version",
            "completed_at",
            "size",
            "dataset_version",
            "train_sha256",
            "fixed_config",
            "loss_config",
            "assistant_token_counts",
            "gpu",
            "cuda",
            "torch",
            "metrics",
            "adapter_files",
            "reload_verified",
        )
    }
    public_training["private_training_manifest_sha256"] = _sha256(PRIVATE_TRAINING_MANIFEST)
    public_training_path = args.output_dir / "dialam_v3_checkpoint_manifest.json"
    public_training_path.write_text(
        json.dumps(public_training, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    copied.append(public_training_path)

    base = result["base"]["deterministic_metrics"]
    v1 = result["v1_n2048"]
    v2 = result["v2_n2048"]
    v3 = result["v3_frozen_n4096"]
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

# Qwen3-0.6B DialAM incremental patch adapter v3 / n=4096

This is the selected FlowJudge QLoRA adapter for one narrow behavior: given one
new proposition and a complete fixed-size block of earlier propositions, emit a
bare JSON object containing every and only direct SUPPORT, ATTACK, or REPHRASE
edge to supplied IDs, or an empty relation list.

V3 uses 4,096 private transformed QT30 examples arranged as 2,048 exact
same-update positive/NONE pairs. It keeps the v1 Qwen3-0.6B QLoRA optimization
configuration and changes the data plus loss reduction: assistant-token loss is
averaged within each example before examples are averaged. This corrects the
4.50x positive-to-NONE output-length weighting imbalance diagnosed after v2.

## Frozen 30-scenario result

| Metric | Base | v1 n=2048 | v2 n=2048 | Selected v3 n=4096 |
|---|---:|---:|---:|---:|
| Exact patch accuracy | {base['exact_patch_accuracy'] * 100:.1f}% | {v1['exact_patch_accuracy'] * 100:.1f}% | {v2['exact_patch_accuracy'] * 100:.1f}% | **{v3['exact_patch_accuracy'] * 100:.1f}%** |
| Edge precision | {base['edge_precision'] * 100:.1f}% | {v1['edge_precision'] * 100:.1f}% | {v2['edge_precision'] * 100:.1f}% | **{v3['edge_precision'] * 100:.1f}%** |
| Edge recall | {base['edge_recall'] * 100:.1f}% | {v1['edge_recall'] * 100:.1f}% | {v2['edge_recall'] * 100:.1f}% | **{v3['edge_recall'] * 100:.1f}%** |
| Edge F1 | {base['edge_f1'] * 100:.1f}% | {v1['edge_f1'] * 100:.1f}% | {v2['edge_f1'] * 100:.1f}% | **{v3['edge_f1'] * 100:.1f}%** |
| Relation macro-F1 | {base['relation_macro_f1'] * 100:.1f}% | {v1['relation_macro_f1'] * 100:.1f}% | {v2['relation_macro_f1'] * 100:.1f}% | **{v3['relation_macro_f1'] * 100:.1f}%** |
| False edges/update | {base['false_edges_per_update']:.3f} | {v1['false_edges_per_update']:.3f} | {v2['false_edges_per_update']:.3f} | **{v3['false_edges_per_update']:.3f}** |
| NONE cases with false edges | n/a | {v1['none_diagnostics']['scenarios_with_false_edges']}/6 | {v2['none_diagnostics']['scenarios_with_false_edges']}/6 | **{v3['none_diagnostics']['scenarios_with_false_edges']}/6** |
| Judge Robustness /4 | {result['base']['judge_metrics']['mean_robustness']:.2f} | {v1['judge_metrics']['mean_robustness']:.2f} | {v2['judge_metrics']['mean_robustness']:.2f} | **{v3['judge_metrics']['mean_robustness']:.2f}** |

V3 cleared its preregistered material-improvement rule and is the selected model
direction. It still failed the original high reliability bar: edge F1 is 35.0%
versus the required 85%, exact patch accuracy is 43.3% versus 80%, and ATTACK is
the weakest relation at 18.2% F1. This is an educational research artifact, not
a claim of production-ready argument mining.

The training text is not redistributed. Consumers must obtain the official QT30
corpus and appropriate permission separately. Included evidence contains only
aggregate metrics, configuration, hashes, and text-free reconstruction metadata.

## Load

Load this PEFT adapter over `Qwen/Qwen3-0.6B` with a compatible
Transformers/PEFT or Unsloth runtime. Use greedy decoding with Qwen thinking
disabled and the project behavior prompt.
"""
    readme = args.output_dir / "README.md"
    readme.write_text(model_card, encoding="utf-8")
    copied.append(readme)

    checkpoint_hash = result["training"]["checkpoint_tree_sha256"]
    observed_checkpoint_files = {
        item["path"]: item["sha256"] for item in result["training"]["checkpoint_files"]
    }
    for source in args.checkpoint.iterdir():
        if not source.is_file() or source.name == "README.md":
            continue
        if observed_checkpoint_files.get(source.name) != _sha256(source):
            raise RuntimeError(f"checkpoint hash differs from frozen report: {source.name}")
    manifest = {
        "schema_version": "dialam_hf_model_package_v3",
        "prepared_at": datetime.now(UTC).isoformat(),
        "model": "Qwen3-0.6B DialAM v3 n=4096 QLoRA adapter",
        "contains_raw_or_transformed_qt30_text": False,
        "selected_submission_checkpoint": True,
        "selection_reason": (
            "v3 cleared every preregistered material-improvement condition, improved edge "
            "F1 to 35.0%, and reduced NONE false positives to 0/6"
        ),
        "frozen_reliability_threshold_cleared": False,
        "checkpoint_tree_sha256": checkpoint_hash,
        "training_data_sha256": data["output"]["sha256"],
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

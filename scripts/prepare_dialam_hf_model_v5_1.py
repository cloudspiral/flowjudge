#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from flowjudge.patch_data import PROJECT_ROOT


DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "artifacts" / "dialam_qlora" / "v5_n8192" / "adapter"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "artifacts" / "hf_publish" / "dialam-qwen3-0.6b-v5-1-n8192"
)
RESULT_REPORT = PROJECT_ROOT / "reports" / "dialam_v5_1_result.json"
RAW_V5_REPORT = PROJECT_ROOT / "reports" / "dialam_v5_pairwise_classification.json"
CALIBRATION_REPORT = PROJECT_ROOT / "reports" / "dialam_v5_1_calibration.json"
DATA_MANIFEST = (
    PROJECT_ROOT / "data" / "dialam" / "training" / "training_v5_manifest.json"
)
PRIVATE_TRAINING_MANIFEST = (
    PROJECT_ROOT / "artifacts" / "dialam_qlora" / "v5_n8192" / "training_manifest.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_code_commit() -> str:
    value = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("git HEAD is not a full lowercase SHA-1")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the selected DialAM v5.1 n=8192 PEFT adapter for Hugging Face"
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    result = json.loads(RESULT_REPORT.read_text(encoding="utf-8"))
    raw_v5 = json.loads(RAW_V5_REPORT.read_text(encoding="utf-8"))
    calibration = json.loads(CALIBRATION_REPORT.read_text(encoding="utf-8"))
    data = json.loads(DATA_MANIFEST.read_text(encoding="utf-8"))
    private_training = json.loads(
        PRIVATE_TRAINING_MANIFEST.read_text(encoding="utf-8")
    )
    if result.get("selection_decision") != "PROMOTE_V5_1_AS_FINAL_DIRECTION":
        raise RuntimeError("v5.1 did not clear its preregistered promotion gate")
    if not result.get("promotion_passed") or not all(result["promotion_checks"].values()):
        raise RuntimeError("v5.1 promotion evidence is incomplete")
    if result["calibration"]["margin"] != calibration["selected"]["margin"]:
        raise RuntimeError("published calibration margin differs from the selected grid point")
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
        (RESULT_REPORT, "dialam_v5_1_result.json"),
        (CALIBRATION_REPORT, "dialam_v5_1_calibration.json"),
        (DATA_MANIFEST, "dialam_v5_data_manifest.json"),
    ):
        destination = args.output_dir / name
        shutil.copyfile(source, destination)
        copied.append(destination)

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
            "reload_probe",
            "reload_probe_label_scores",
        )
    }
    public_training["private_training_manifest_sha256"] = _sha256(
        PRIVATE_TRAINING_MANIFEST
    )
    public_training_path = args.output_dir / "dialam_v5_checkpoint_manifest.json"
    _write_json(public_training_path, public_training)
    copied.append(public_training_path)

    inference_config = {
        "schema_version": "dialam_pairwise_inference_v5_1",
        "task": "incremental_grounded_argument_graph_patching",
        "external_output": "one bare JSON object with a relations array",
        "candidate_scoring": "mean_assistant_token_log_probability",
        "max_sequence_length": 2048,
        "allowed_labels": ["NONE", "SUPPORT", "ATTACK", "REPHRASE"],
        "positive_tie_order": ["SUPPORT", "ATTACK", "REPHRASE"],
        "none_margin": result["calibration"]["margin"],
        "decision_rule": (
            "emit the highest-scoring positive label only when its mean token "
            "log probability minus NONE exceeds none_margin; otherwise emit NONE"
        ),
        "prompt_template_path": "prompts/dialam/pairwise_v5.txt",
        "complete_block_required_for_every_candidate": True,
        "source_is_always_new_proposition_id": True,
        "assembly": "omit NONE decisions and preserve supplied candidate order",
    }
    inference_path = args.output_dir / "dialam_inference_config.json"
    _write_json(inference_path, inference_config)
    copied.append(inference_path)

    v3 = result["v3_frozen_n4096"]
    v5_1 = result["v5_1_frozen_n8192"]
    model_card = f"""---
base_model: Qwen/Qwen3-0.6B
library_name: peft
pipeline_tag: text-classification
license: other
tags:
  - qwen3
  - qlora
  - argument-mining
  - dialam
  - calibrated-classification
---

# Qwen3-0.6B DialAM pairwise patch adapter v5.1 / n=8192

This is the selected FlowJudge QLoRA adapter for one narrow behavior: given one
new proposition and a complete block of earlier propositions, emit every and
only direct SUPPORT, ATTACK, or REPHRASE edge to supplied IDs as one bare JSON
object. Return an empty relation list when no direct edge exists.

V5 trains 8,192 candidate-level rows as 4,096 exact positive/NONE contrast
pairs. V5.1 does not retrain or alter the checkpoint. It applies the
preregistered fixed `3.0` NONE margin selected on a separate four-parent-episode
development set to correct the known 50.0% training versus 7.379% natural
positive-prior shift.

## Required inference contract

This adapter is a fixed-label classifier, not an unconstrained text generator.
For every earlier candidate, render the complete block with the v5 pairwise
prompt, compute the mean assistant-token log probability of `NONE`, `SUPPORT`,
`ATTACK`, and `REPHRASE`, and apply `dialam_inference_config.json`. Emitting the
result of ordinary `.generate()` is not equivalent. The source repository's
`eval.py` implements the exact pipeline and deterministic block assembly.

```bash
uv sync --group train
uv run python eval.py \\
  --model mr-mc/flowjudge-dialam-qwen3-0.6b-v5-1-n8192 \\
  --eval-set <dialam-patch-example-jsonl>
```

## Frozen 30-scenario result

| Metric | Previous v3 / 4096 | Selected v5.1 / 8192 |
|---|---:|---:|
| Exact patch accuracy | {v3['exact_patch_accuracy'] * 100:.1f}% | **{v5_1['exact_patch_accuracy'] * 100:.1f}%** |
| Edge precision | {v3['edge_precision'] * 100:.1f}% | **{v5_1['edge_precision'] * 100:.1f}%** |
| Edge recall | {v3['edge_recall'] * 100:.1f}% | **{v5_1['edge_recall'] * 100:.1f}%** |
| Edge F1 | {v3['edge_f1'] * 100:.1f}% | **{v5_1['edge_f1'] * 100:.1f}%** |
| Relation macro-F1 | {v3['relation_macro_f1'] * 100:.1f}% | **{v5_1['relation_macro_f1'] * 100:.1f}%** |
| ATTACK F1 | {v3['relation_metrics']['ATTACK']['f1'] * 100:.1f}% | **{v5_1['relation_metrics']['ATTACK']['f1'] * 100:.1f}%** |
| False edges/update | {v3['false_edges_per_update']:.3f} | **{v5_1['false_edges_per_update']:.3f}** |
| NONE cases with false edges | {v3['none_diagnostics']['scenarios_with_false_edges']}/6 | **{v5_1['none_diagnostics']['scenarios_with_false_edges']}/6** |
| Judge Robustness /4 | {v3['judge_metrics']['mean_robustness']:.3f} | **{v5_1['judge_metrics']['mean_robustness']:.3f}** |

V5.1 passed all nine preregistered reused-benchmark promotion checks and is the
selected submission model. It still fails the original production-like
reliability bar (80% exact, 85% edge F1, 75% macro-F1, at most 0.20 false
edges/update, and 3.5/4 Robustness), so this is an educational research artifact,
not a production-ready argument-mining claim.

## Source and permission

Training data is transformed from English QT30 as distributed for DialAM-2024.
The project owner directly attested project-specific use and redistribution
permission on 2026-08-23. No general QT30 license is claimed. The official raw
archive/maps are not included; the companion dataset repository publishes the
permission-cleared transformed JSONL, manifests, evidence, and reconstruction
code.
"""
    readme = args.output_dir / "README.md"
    readme.write_text(model_card, encoding="utf-8")
    copied.append(readme)

    observed_checkpoint_files = {
        item["path"]: item["sha256"]
        for item in raw_v5["training"]["checkpoint_files"]
    }
    for source in args.checkpoint.iterdir():
        if not source.is_file() or source.name == "README.md":
            continue
        if observed_checkpoint_files.get(source.name) != _sha256(source):
            raise RuntimeError(f"checkpoint hash differs from frozen report: {source.name}")

    manifest = {
        "schema_version": "dialam_hf_model_package_v5_1",
        "prepared_at": datetime.now(UTC).isoformat(),
        "source_code_commit": _source_code_commit(),
        "model": "Qwen3-0.6B DialAM v5.1 n=8192 pairwise QLoRA adapter",
        "contains_raw_or_transformed_qt30_text": False,
        "selected_submission_checkpoint": True,
        "selection_reason": (
            "v5.1 passed every preregistered frozen promotion check, improving "
            "exact accuracy to 53.3%, edge F1 to 47.6%, macro-F1 to 47.4%, "
            "and false edges/update to 0.267"
        ),
        "frozen_reliability_threshold_cleared": result[
            "clears_original_reliability_bar"
        ],
        "checkpoint_tree_sha256": raw_v5["training"]["checkpoint_tree_sha256"],
        "training_data_sha256": data["output"]["sha256"],
        "inference_config_sha256": _sha256(inference_path),
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
    _write_json(manifest_path, manifest)
    print(args.output_dir)


if __name__ == "__main__":
    main()

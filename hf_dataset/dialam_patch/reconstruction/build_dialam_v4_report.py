#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from flowjudge.dialam_training import file_sha256
from flowjudge.patch_data import PROJECT_ROOT
from flowjudge.patch_metrics import score_patch_prediction_file


V3_REPORT = PROJECT_ROOT / "reports" / "dialam_v1_v2_v3.json"
V4_DATA_MANIFEST = (
    PROJECT_ROOT / "data" / "dialam" / "training" / "training_v4_manifest.json"
)
V4_TRAINING_MANIFEST = (
    PROJECT_ROOT / "artifacts" / "dialam_qlora" / "v4_n8192" / "training_manifest.json"
)
V4_CHECKPOINT = V4_TRAINING_MANIFEST.parent / "adapter"
DEV_GOLD = (
    PROJECT_ROOT / "data" / "dialam" / "training" / "v3_dev_eval_examples.jsonl"
)
V3_DEV_PREDICTIONS = (
    PROJECT_ROOT
    / "results"
    / "dialam_model_generation"
    / "v3_n4096_dev"
    / "predictions.jsonl"
)
V4_DEV_PREDICTIONS = (
    PROJECT_ROOT
    / "results"
    / "dialam_model_generation"
    / "v4_n8192_dev"
    / "predictions.jsonl"
)
OUTPUT = PROJECT_ROOT / "reports" / "dialam_v4_class_balanced_rehearsal.json"
DOC = PROJECT_ROOT / "docs" / "dialam_v4_results.md"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _tree_manifest(path: Path) -> tuple[str, list[dict[str, Any]]]:
    digest = hashlib.sha256()
    files: list[dict[str, Any]] = []
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        relative = str(item.relative_to(path))
        item_hash = file_sha256(item)
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(item_hash.encode())
        digest.update(b"\n")
        files.append(
            {
                "path": relative,
                "sha256": item_hash,
                "bytes": item.stat().st_size,
            }
        )
    if not files:
        raise FileNotFoundError(f"checkpoint is empty or missing: {path}")
    return digest.hexdigest(), files


def _private(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
        "redistribution": "private/local only",
    }


def _aggregate(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metrics.items()
        if key not in {"failure_cases", "scenario_failure_cases"}
    }


def _none_diagnostics(examples_path: Path, predictions_path: Path) -> dict[str, Any]:
    examples = _load_jsonl(examples_path)
    predictions = {row["example_id"]: row for row in _load_jsonl(predictions_path)}
    none_examples = [row for row in examples if not row["gold_patch"]["relations"]]
    false_labels: Counter[str] = Counter()
    with_false_edges = 0
    for example in none_examples:
        response = predictions[example["example_id"]]["raw_response"]
        try:
            relations = json.loads(response)["relations"]
        except (json.JSONDecodeError, KeyError, TypeError):
            relations = []
        if relations:
            with_false_edges += 1
        false_labels.update(item.get("type", "INVALID") for item in relations)
    return {
        "none_scenarios": len(none_examples),
        "scenarios_with_false_edges": with_false_edges,
        "false_positive_rate": with_false_edges / len(none_examples),
        "false_edge_labels": dict(sorted(false_labels.items())),
    }


def _delta(current: dict[str, Any], previous: dict[str, Any]) -> dict[str, float | int]:
    keys = (
        "exact_patch_accuracy",
        "edge_precision",
        "edge_recall",
        "edge_f1",
        "relation_macro_f1",
        "false_edges_per_update",
        "false_positive_edges",
        "false_negative_edges",
    )
    return {key: current[key] - previous[key] for key in keys}


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def build_report() -> dict[str, Any]:
    v3_report = json.loads(V3_REPORT.read_text(encoding="utf-8"))
    data_manifest = json.loads(V4_DATA_MANIFEST.read_text(encoding="utf-8"))
    training = json.loads(V4_TRAINING_MANIFEST.read_text(encoding="utf-8"))
    v3 = score_patch_prediction_file(V3_DEV_PREDICTIONS, DEV_GOLD)
    v4 = score_patch_prediction_file(V4_DEV_PREDICTIONS, DEV_GOLD)
    v3_none = _none_diagnostics(DEV_GOLD, V3_DEV_PREDICTIONS)
    v4_none = _none_diagnostics(DEV_GOLD, V4_DEV_PREDICTIONS)

    if file_sha256(DEV_GOLD) != data_manifest["development_gate"]["eval_examples_sha256"]:
        raise ValueError("v4 development gold differs from the preregistered hash")
    if training["train_sha256"] != data_manifest["output"]["sha256"]:
        raise ValueError("v4 training input differs from its preregistered manifest")
    if training["fixed_config"] != v3_report["training"]["fixed_config"]:
        raise ValueError("v4 changed a fixed optimization hyperparameter")
    if training["loss_config"]["name"] != data_manifest["loss_policy"]["name"]:
        raise ValueError("v4 runtime loss differs from the preregistered policy")
    if not training["reload_verified"]:
        raise ValueError("v4 adapter reload was not verified")

    predictions = _load_jsonl(V4_DEV_PREDICTIONS)
    if len(predictions) != 30 or len({row["example_id"] for row in predictions}) != 30:
        raise ValueError("v4 development predictions are not 30 unique examples")
    if any(
        row.get("dataset_version") != "v4"
        or row.get("adapter_size") != 8192
        or row.get("eval_split") != "v3_dev"
        for row in predictions
    ):
        raise ValueError("v4 development prediction metadata is invalid")

    threshold = data_manifest["development_gate"][
        "requirements_to_run_reused_frozen_benchmark"
    ]
    checks = {
        "exact_patch_accuracy": (
            v4["exact_patch_accuracy"] >= threshold["minimum_exact_patch_accuracy"]
        ),
        "edge_f1": v4["edge_f1"] >= threshold["minimum_edge_f1"],
        "attack_f1": (
            v4["relation_metrics"]["ATTACK"]["f1"]
            >= threshold["minimum_attack_f1"]
        ),
        "false_edges_per_update": (
            v4["false_edges_per_update"]
            <= threshold["maximum_false_edges_per_update"]
        ),
        "none_scenarios_with_false_edges": (
            v4_none["scenarios_with_false_edges"]
            <= threshold["maximum_none_scenarios_with_false_edges"]
        ),
        "json_validity": (
            v4["json_validity_rate"] >= threshold["json_and_schema_validity"]
        ),
        "schema_validity": (
            v4["schema_validity_rate"] >= threshold["json_and_schema_validity"]
        ),
    }
    development_passed = all(checks.values())
    if development_passed:
        raise ValueError(
            "v4 passed development; run the preregistered reused-frozen benchmark "
            "before building the final report"
        )

    checkpoint_hash, checkpoint_files = _tree_manifest(V4_CHECKPOINT)
    delta = _delta(v4, v3)
    delta.update(
        {
            "attack_f1": (
                v4["relation_metrics"]["ATTACK"]["f1"]
                - v3["relation_metrics"]["ATTACK"]["f1"]
            ),
            "rephrase_f1": (
                v4["relation_metrics"]["REPHRASE"]["f1"]
                - v3["relation_metrics"]["REPHRASE"]["f1"]
            ),
            "support_f1": (
                v4["relation_metrics"]["SUPPORT"]["f1"]
                - v3["relation_metrics"]["SUPPORT"]["f1"]
            ),
            "none_scenarios_with_false_edges": (
                v4_none["scenarios_with_false_edges"]
                - v3_none["scenarios_with_false_edges"]
            ),
        }
    )
    report: dict[str, Any] = {
        "schema_version": "dialam_v4_class_balanced_rehearsal_result_v1",
        "experiment": "v4/n8192 class-balanced rehearsal versus v3/n4096",
        "decision": "RETAIN_V3_V4_FAILED_DEVELOPMENT_GATE",
        "development_gate_passed": development_passed,
        "development_gate_checks": checks,
        "failed_development_conditions": sorted(
            name for name, passed in checks.items() if not passed
        ),
        "reused_frozen_benchmark": {
            "status": "NOT_RUN_PREREGISTERED_DEVELOPMENT_GATE_FAILED",
            "candidate_calls": 0,
            "judge_calls": 0,
            "eval_sha256": data_manifest["reused_frozen_benchmark"]["eval_sha256"],
            "judge_rubric_sha256": data_manifest["reused_frozen_benchmark"][
                "judge_rubric_sha256"
            ],
        },
        "controlled_variables": {
            "base_model": training["fixed_config"]["base_model"],
            "optimization_config_unchanged_from_v3": True,
            "loss_reduction_unchanged_from_v3": True,
            "development_eval_unchanged_from_v3": True,
            "development_eval_sha256": file_sha256(DEV_GOLD),
            "declared_intervention": (
                "double training rows to 8192; balance single-label positive rows; "
                "retain v3 foundation; add unique hard NONE rehearsal; repeat only ATTACK"
            ),
        },
        "v3_development_n4096": {
            **_aggregate(v3),
            "none_diagnostics": v3_none,
        },
        "v4_development_n8192": {
            **_aggregate(v4),
            "none_diagnostics": v4_none,
        },
        "delta_v4_minus_v3_development": delta,
        "training": {
            "metrics": training["metrics"],
            "fixed_config": training["fixed_config"],
            "loss_config": training["loss_config"],
            "assistant_token_counts": training["assistant_token_counts"],
            "reload_verified": training["reload_verified"],
            "train_sha256": training["train_sha256"],
            "training_manifest_sha256": file_sha256(V4_TRAINING_MANIFEST),
            "checkpoint_tree_sha256": checkpoint_hash,
            "checkpoint_files": checkpoint_files,
        },
        "data": {
            "size": data_manifest["size"],
            "category_counts": data_manifest["category_counts"],
            "category_unique_example_counts": data_manifest[
                "category_unique_example_counts"
            ],
            "relation_counts": data_manifest["relation_counts"],
            "source_stage_counts": data_manifest["source_stage_counts"],
            "sampling_role_counts": data_manifest["sampling_role_counts"],
            "repetition_policy": data_manifest["repetition_policy"],
            "supplement_policy": data_manifest["supplement_policy"],
            "training_parent_episode_count": data_manifest[
                "training_parent_episode_count"
            ],
            "development_parent_episode_count": data_manifest[
                "development_parent_episode_count"
            ],
            "frozen_parent_episode_count": data_manifest["frozen_parent_episode_count"],
            "dialogue_leakage": data_manifest["dialogue_leakage"],
            "private_jsonl_sha256": data_manifest["output"]["sha256"],
        },
        "failure_diagnosis": {
            "successful_effect": (
                "ATTACK F1 rose from 0.0% to 54.5% and total edge F1 rose from "
                "21.1% to 34.1% on the episode-disjoint development set."
            ),
            "blocking_regression": (
                "NONE cases with a false edge doubled from 2/6 to 4/6; every false "
                "edge on those cases was SUPPORT. REPHRASE F1 also fell to 0.0%."
            ),
            "interpretation": (
                "Class balance repaired rare ATTACK recall but did not preserve v3's "
                "paired calibration. More rows alone are not the limiting factor; the "
                "joint variable-length patch target still trades relation recall against "
                "NONE precision and label discrimination."
            ),
            "recommended_next_experiment": (
                "If work continues, preregister pairwise SUPPORT/ATTACK/REPHRASE/NONE "
                "classification with deterministic patch assembly, while retaining v3 as "
                "the selected published model."
            ),
        },
        "private_artifacts": {
            "training_jsonl": _private(
                PROJECT_ROOT
                / "data"
                / "dialam"
                / "training"
                / "dialam_v4_n8192.jsonl"
            ),
            "development_gold": _private(DEV_GOLD),
            "development_predictions": _private(V4_DEV_PREDICTIONS),
            "training_log": _private(V4_TRAINING_MANIFEST),
            "checkpoint": {
                "path": str(V4_CHECKPOINT.relative_to(PROJECT_ROOT)),
                "tree_sha256": checkpoint_hash,
                "redistribution": (
                    "publishable model adapter; preserved locally and in Modal, not "
                    "promoted because the development gate failed"
                ),
            },
        },
        "reproduction_commands": [
            ".venv/bin/python scripts/build_dialam_training_v4.py",
            "modal run scripts/modal_dialam_qlora.py --action train --size 8192 --dataset-version v4",
            "modal run scripts/modal_dialam_qlora.py --action evaluate --target tuned --size 8192 --dataset-version v4 --eval-split v3_dev --output-path results/dialam_model_generation/v4_n8192_dev/predictions.jsonl",
            ".venv/bin/python scripts/score_dialam_predictions.py results/dialam_model_generation/v4_n8192_dev/predictions.jsonl --examples data/dialam/training/v3_dev_eval_examples.jsonl",
            ".venv/bin/python scripts/build_dialam_v4_report.py",
        ],
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    markdown = f"""# DialAM v4 class-balanced rehearsal result

Decision: **{report['decision']}**.

V4 improved semantic edge recovery on the episode-disjoint development set,
especially ATTACK, but failed its preregistered NONE-calibration condition. The
reused frozen benchmark and judge were therefore **not run**. V3 remains the
selected public model.

| Development metric | V3 / n=4096 | V4 / n=8192 | Delta |
|---|---:|---:|---:|
| Exact patch accuracy | {_pct(v3['exact_patch_accuracy'])} | **{_pct(v4['exact_patch_accuracy'])}** | {_pct(delta['exact_patch_accuracy'])} |
| Edge precision | {_pct(v3['edge_precision'])} | **{_pct(v4['edge_precision'])}** | {_pct(delta['edge_precision'])} |
| Edge recall | {_pct(v3['edge_recall'])} | **{_pct(v4['edge_recall'])}** | {_pct(delta['edge_recall'])} |
| Edge F1 | {_pct(v3['edge_f1'])} | **{_pct(v4['edge_f1'])}** | {_pct(delta['edge_f1'])} |
| Relation macro-F1 | {_pct(v3['relation_macro_f1'])} | **{_pct(v4['relation_macro_f1'])}** | {_pct(delta['relation_macro_f1'])} |
| ATTACK F1 | {_pct(v3['relation_metrics']['ATTACK']['f1'])} | **{_pct(v4['relation_metrics']['ATTACK']['f1'])}** | {_pct(delta['attack_f1'])} |
| REPHRASE F1 | {_pct(v3['relation_metrics']['REPHRASE']['f1'])} | {_pct(v4['relation_metrics']['REPHRASE']['f1'])} | {_pct(delta['rephrase_f1'])} |
| SUPPORT F1 | {_pct(v3['relation_metrics']['SUPPORT']['f1'])} | **{_pct(v4['relation_metrics']['SUPPORT']['f1'])}** | {_pct(delta['support_f1'])} |
| False edges/update | {v3['false_edges_per_update']:.3f} | {v4['false_edges_per_update']:.3f} | {delta['false_edges_per_update']:+.3f} |
| NONE cases with false edges | {v3_none['scenarios_with_false_edges']}/6 | **{v4_none['scenarios_with_false_edges']}/6** | {delta['none_scenarios_with_false_edges']:+d} |
| JSON/schema validity | {_pct(v3['json_validity_rate'])} / {_pct(v3['schema_validity_rate'])} | {_pct(v4['json_validity_rate'])} / {_pct(v4['schema_validity_rate'])} | unchanged |

## Gate decision

V4 passed exact accuracy, edge F1, ATTACK F1, false edges/update, JSON validity,
and schema validity. It failed the condition allowing at most two of six NONE
cases to contain a false edge: v4 produced false edges on 4/6, all SUPPORT.
Following the preregistration, no reused-frozen candidate calls or judge calls
were made.

## What this learned

The intervention achieved its intended rare-class effect: ATTACK F1 rose from
0.0% to 54.5%, edge recall rose from 16.7% to 29.2%, and edge F1 rose from
21.1% to 34.1%. But it weakened calibrated sparsity and label balance:
REPHRASE F1 fell from 16.7% to 0.0%, and NONE false-positive cases doubled.

This rules out "just add more class-balanced rows" as the final fix. The
joint, variable-length patch objective still trades recall against NONE
precision. If work continues, the strongest next experiment is a separately
preregistered pairwise `SUPPORT|ATTACK|REPHRASE|NONE` classifier followed by
deterministic patch assembly. That is a formulation change, not a v4 retune.

## Reproduce

```bash
{chr(10).join(report['reproduction_commands'])}
```

The v4 checkpoint is preserved locally and in the authenticated Modal volume.
QT30-derived training/evaluation text and predictions remain ignored. The JSON
report publishes only aggregate metrics, configuration, paths, and hashes.
"""
    DOC.write_text(markdown, encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(build_report(), indent=2, sort_keys=True))

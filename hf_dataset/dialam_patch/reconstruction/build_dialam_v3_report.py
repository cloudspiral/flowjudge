#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from flowjudge.dialam_training import file_sha256
from flowjudge.patch_data import DEFAULT_DIAGNOSTIC_EVAL_EXAMPLES_PATH, PROJECT_ROOT
from flowjudge.patch_metrics import score_patch_prediction_file


V1_REPORT = PROJECT_ROOT / "reports" / "dialam_v1_efficiency_curve.json"
V2_REPORT = PROJECT_ROOT / "reports" / "dialam_v1_to_v2.json"
V2_SUMMARY = PROJECT_ROOT / "results" / "dialam_model_eval" / "v2_n2048" / "summary.json"
V3_SUMMARY = PROJECT_ROOT / "results" / "dialam_model_eval" / "v3_n4096" / "summary.json"
V3_TRAINING = (
    PROJECT_ROOT / "artifacts" / "dialam_qlora" / "v3_n4096" / "training_manifest.json"
)
V3_DATA_MANIFEST = (
    PROJECT_ROOT / "data" / "dialam" / "training" / "training_v3_manifest.json"
)
V3_DEV_GOLD = (
    PROJECT_ROOT / "data" / "dialam" / "training" / "v3_dev_eval_examples.jsonl"
)
V3_DEV_PREDICTIONS = (
    PROJECT_ROOT
    / "results"
    / "dialam_model_generation"
    / "v3_n4096_dev"
    / "predictions.jsonl"
)
BASE_SUMMARY = PROJECT_ROOT / "results" / "dialam_model_eval" / "base" / "summary.json"
OUTPUT = PROJECT_ROOT / "reports" / "dialam_v1_v2_v3.json"
DOC = PROJECT_ROOT / "docs" / "dialam_v3_results.md"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _tree_manifest(path: Path) -> tuple[str, list[dict[str, Any]]]:
    digest = hashlib.sha256()
    files = []
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
    false_labels = Counter()
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


def _checks(
    metrics: dict[str, Any],
    judge: dict[str, Any],
    threshold: dict[str, Any],
) -> dict[str, bool]:
    return {
        "direction_accuracy": metrics["direction_accuracy"]
        >= threshold["direction_accuracy_min"],
        "edge_f1": metrics["edge_f1"] >= threshold["edge_f1_min"],
        "exact_patch_accuracy": metrics["exact_patch_accuracy"]
        >= threshold["exact_patch_accuracy_min"],
        "false_edges_per_update": metrics["false_edges_per_update"]
        <= threshold["false_edges_per_update_max"],
        "invalid_id_count": metrics["invalid_id_count"]
        <= threshold["invalid_id_count_max"],
        "json_validity_rate": metrics["json_validity_rate"]
        >= threshold["json_validity_rate_min"],
        "judge_robustness_mean": judge["mean_robustness"]
        >= threshold["judge_robustness_mean_min"],
        "judge_spec_adherence_mean": judge["mean_spec_adherence"]
        >= threshold["judge_spec_adherence_mean_min"],
        "judge_validity_rate": judge["judge_validity_rate"]
        >= threshold["judge_validity_rate_min"],
        "relation_macro_f1": metrics["relation_macro_f1"]
        >= threshold["relation_macro_f1_min"],
        "schema_validity_rate": metrics["schema_validity_rate"]
        >= threshold["schema_validity_rate_min"],
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
    v1_report = json.loads(V1_REPORT.read_text(encoding="utf-8"))
    v2_report = json.loads(V2_REPORT.read_text(encoding="utf-8"))
    v2_summary = json.loads(V2_SUMMARY.read_text(encoding="utf-8"))
    v3_summary = json.loads(V3_SUMMARY.read_text(encoding="utf-8"))
    training = json.loads(V3_TRAINING.read_text(encoding="utf-8"))
    data_manifest = json.loads(V3_DATA_MANIFEST.read_text(encoding="utf-8"))
    base_summary = json.loads(BASE_SUMMARY.read_text(encoding="utf-8"))
    v1 = next(item for item in v1_report["curve"] if item["n"] == 2048)
    v2_evidence = v2_report["v2_n2048"]
    v2 = v2_summary["deterministic_metrics"]
    v3 = v3_summary["deterministic_metrics"]
    dev = score_patch_prediction_file(V3_DEV_PREDICTIONS, V3_DEV_GOLD)
    frozen_none = _none_diagnostics(
        DEFAULT_DIAGNOSTIC_EVAL_EXAMPLES_PATH,
        V3_SUMMARY.parent / "predictions.jsonl",
    )
    dev_none = _none_diagnostics(V3_DEV_GOLD, V3_DEV_PREDICTIONS)

    if v3_summary["eval_sha256"] != v1_report["frozen_eval_sha256"]:
        raise ValueError("v3 did not use the frozen v1/v2 evaluation")
    if v3_summary["judge_rubric_sha256"] != v1_report["frozen_judge_rubric_sha256"]:
        raise ValueError("v3 did not use the frozen judge rubric")
    if training["train_sha256"] != data_manifest["output"]["sha256"]:
        raise ValueError("v3 training input differs from its preregistered manifest")
    if training["fixed_config"] != v1_report["fixed_training_config"]:
        raise ValueError("v3 changed a fixed optimization hyperparameter")
    if training["loss_config"]["name"] != data_manifest["loss_policy"]["name"]:
        raise ValueError("v3 runtime loss differs from the preregistered policy")

    criterion = data_manifest["preregistered_material_improvement"]
    material_checks = {
        "edge_f1": v3["edge_f1"] >= criterion["minimum_edge_f1"],
        "false_edges_per_update": v3["false_edges_per_update"]
        <= criterion["maximum_false_edges_per_update"],
        "none_scenarios_with_false_edges": frozen_none["scenarios_with_false_edges"]
        <= criterion["maximum_none_scenarios_with_false_edges"],
        "json_validity": v3["json_validity_rate"]
        >= criterion["json_and_schema_validity_must_remain"],
        "schema_validity": v3["schema_validity_rate"]
        >= criterion["json_and_schema_validity_must_remain"],
    }
    material = all(material_checks.values())
    reliability_checks = _checks(
        v3,
        v3_summary["judge_metrics"],
        v1_report["reliability_threshold"],
    )
    checkpoint_dir = V3_TRAINING.parent / "adapter"
    checkpoint_hash, checkpoint_files = _tree_manifest(checkpoint_dir)
    v1_metrics = {
        key: v1[key]
        for key in (
            "exact_patch_accuracy",
            "edge_precision",
            "edge_recall",
            "edge_f1",
            "relation_macro_f1",
            "false_edges_per_update",
            "false_positive_edges",
            "false_negative_edges",
            "relation_metrics",
            "json_validity_rate",
            "schema_validity_rate",
        )
    }
    v2_metrics = {
        key: v2[key]
        for key in (
            "exact_patch_accuracy",
            "edge_precision",
            "edge_recall",
            "edge_f1",
            "relation_macro_f1",
            "false_edges_per_update",
            "false_positive_edges",
            "false_negative_edges",
            "relation_metrics",
        )
    }
    report = {
        "schema_version": "dialam_v1_v2_v3_comparison_v1",
        "experiment": "v3/n4096 paired-loss correction versus v1/v2 n2048",
        "material_improvement": material,
        "material_improvement_checks": material_checks,
        "decision": (
            "PROMOTE_V3_AS_FINAL_DIRECTION"
            if material
            else "V3_DID_NOT_CLEAR_MATERIAL_IMPROVEMENT_BAR"
        ),
        "v3_clears_frozen_reliability_bar": all(reliability_checks.values()),
        "v3_failed_reliability_thresholds": sorted(
            name for name, passed in reliability_checks.items() if not passed
        ),
        "preregistered_criterion": criterion,
        "controlled_variables": {
            "base_model": v3_summary["model"],
            "optimization_config_unchanged": True,
            "optimization_config_sha256": v1_report["fixed_training_config_sha256"],
            "declared_interventions": [
                "same-update positive/NONE paired sampling at n=4096",
                "per-example assistant-token mean loss before batch mean",
                "four parent episodes removed from training for development evaluation",
            ],
            "eval_unchanged": True,
            "eval_sha256": v3_summary["eval_sha256"],
            "judge_rubric_unchanged": True,
            "judge_rubric_sha256": v3_summary["judge_rubric_sha256"],
        },
        "base": {
            "deterministic_metrics": _aggregate(base_summary["deterministic_metrics"]),
            "judge_metrics": base_summary["judge_metrics"],
        },
        "v1_n2048": {
            **v1_metrics,
            "none_diagnostics": v1["none_diagnostics"],
            "judge_metrics": v1["judge_metrics"],
        },
        "v2_n2048": {
            **v2_metrics,
            "none_diagnostics": v2_evidence["none_diagnostics"],
            "judge_metrics": v2_evidence["judge_metrics"],
        },
        "v3_development_n4096": {
            **_aggregate(dev),
            "none_diagnostics": dev_none,
        },
        "v3_frozen_n4096": {
            **_aggregate(v3),
            "none_diagnostics": frozen_none,
            "judge_metrics": v3_summary["judge_metrics"],
            "reliability_checks": reliability_checks,
        },
        "delta_v3_minus_v1": {
            **_delta(v3, v1),
            "none_scenarios_with_false_edges": (
                frozen_none["scenarios_with_false_edges"]
                - v1["none_diagnostics"]["scenarios_with_false_edges"]
            ),
            "mean_judge_robustness": (
                v3_summary["judge_metrics"]["mean_robustness"]
                - v1["judge_metrics"]["mean_robustness"]
            ),
        },
        "delta_v3_minus_v2": {
            **_delta(v3, v2),
            "none_scenarios_with_false_edges": (
                frozen_none["scenarios_with_false_edges"]
                - v2_evidence["none_diagnostics"]["scenarios_with_false_edges"]
            ),
            "mean_judge_robustness": (
                v3_summary["judge_metrics"]["mean_robustness"]
                - v2_evidence["judge_metrics"]["mean_robustness"]
            ),
        },
        "training": {
            "metrics": training["metrics"],
            "fixed_config": training["fixed_config"],
            "loss_config": training["loss_config"],
            "assistant_token_counts": training["assistant_token_counts"],
            "reload_verified": training["reload_verified"],
            "train_sha256": training["train_sha256"],
            "training_manifest_sha256": file_sha256(V3_TRAINING),
            "checkpoint_tree_sha256": checkpoint_hash,
            "checkpoint_files": checkpoint_files,
        },
        "data": {
            "size": data_manifest["size"],
            "pair_count": data_manifest["pair_count"],
            "category_counts": data_manifest["category_counts"],
            "relation_counts": data_manifest["relation_counts"],
            "pair_policy": data_manifest["pair_policy"],
            "loss_policy": data_manifest["loss_policy"],
            "training_parent_episode_count": data_manifest["training_parent_episode_count"],
            "development_parent_episode_count": data_manifest[
                "development_parent_episode_count"
            ],
            "frozen_parent_episode_count": data_manifest["frozen_parent_episode_count"],
            "dialogue_leakage": data_manifest["dialogue_leakage"],
            "private_jsonl_sha256": data_manifest["output"]["sha256"],
        },
        "failure_diagnosis": {
            "resolved_primary_failure": (
                "false-positive SUPPORT overprediction on NONE cases fell from 2/6 in v1 "
                "and 5/6 in v2 to 0/6 in v3"
            ),
            "remaining_primary_failure": (
                "false negatives and relation-label/target confusions; ATTACK remains the "
                "weakest class at 18.2% F1"
            ),
            "interpretation": (
                "Pairing and equal per-example loss fixed much of the calibration problem, "
                "but 0.6B capacity and sparse paired ATTACK coverage still limit reliable semantics."
            ),
        },
        "private_artifacts": {
            "training_jsonl": _private(
                PROJECT_ROOT / "data" / "dialam" / "training" / "dialam_v3_n4096.jsonl"
            ),
            "development_gold": _private(V3_DEV_GOLD),
            "development_predictions": _private(V3_DEV_PREDICTIONS),
            "frozen_predictions": _private(V3_SUMMARY.parent / "predictions.jsonl"),
            "frozen_records": _private(V3_SUMMARY.parent / "records.jsonl"),
            "judge_transcripts": _private(V3_SUMMARY.parent / "judge_transcripts.jsonl"),
            "training_log": _private(V3_TRAINING),
            "checkpoint": {
                "path": str(checkpoint_dir.relative_to(PROJECT_ROOT)),
                "tree_sha256": checkpoint_hash,
                "redistribution": "publishable model adapter; local copy remains ignored",
            },
        },
        "reproduction_commands": [
            ".venv/bin/python scripts/build_dialam_training_v3.py",
            "modal run scripts/modal_dialam_qlora.py --action train --size 4096 --dataset-version v3",
            "modal run scripts/modal_dialam_qlora.py --action evaluate --target tuned --size 4096 --dataset-version v3 --eval-split v3_dev --output-path results/dialam_model_generation/v3_n4096_dev/predictions.jsonl",
            ".venv/bin/python scripts/score_dialam_predictions.py results/dialam_model_generation/v3_n4096_dev/predictions.jsonl --examples data/dialam/training/v3_dev_eval_examples.jsonl",
            "modal run scripts/modal_dialam_qlora.py --action evaluate --target tuned --size 4096 --dataset-version v3 --eval-split frozen --output-path results/dialam_model_generation/v3_n4096/predictions.jsonl",
            ".venv/bin/python scripts/evaluate_dialam_model.py --predictions results/dialam_model_generation/v3_n4096/predictions.jsonl --output-dir results/dialam_model_eval/v3_n4096 --approval APPROVE_DIALAM_MODEL_JUDGING",
            ".venv/bin/python scripts/build_dialam_v3_report.py",
        ],
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    base = report["base"]["deterministic_metrics"]
    frozen = report["v3_frozen_n4096"]
    d1 = report["delta_v3_minus_v1"]
    d2 = report["delta_v3_minus_v2"]
    markdown = f"""# DialAM v3 paired-loss result

Decision: **{report['decision']}**.

V3 clears every preregistered material-improvement condition and becomes the selected model direction. It does **not** clear the original frozen reliability bar, so it remains a research/assignment artifact rather than a reliable argument-mining system.

| Metric | Base | v1 / n=2048 | v2 / n=2048 | v3 / n=4096 |
|---|---:|---:|---:|---:|
| Exact patch accuracy | {_pct(base['exact_patch_accuracy'])} | {_pct(v1['exact_patch_accuracy'])} | {_pct(v2['exact_patch_accuracy'])} | **{_pct(v3['exact_patch_accuracy'])}** |
| Edge precision | {_pct(base['edge_precision'])} | {_pct(v1['edge_precision'])} | {_pct(v2['edge_precision'])} | **{_pct(v3['edge_precision'])}** |
| Edge recall | {_pct(base['edge_recall'])} | {_pct(v1['edge_recall'])} | {_pct(v2['edge_recall'])} | **{_pct(v3['edge_recall'])}** |
| Edge F1 | {_pct(base['edge_f1'])} | {_pct(v1['edge_f1'])} | {_pct(v2['edge_f1'])} | **{_pct(v3['edge_f1'])}** |
| Relation macro-F1 | {_pct(base['relation_macro_f1'])} | {_pct(v1['relation_macro_f1'])} | {_pct(v2['relation_macro_f1'])} | **{_pct(v3['relation_macro_f1'])}** |
| False edges/update | {base['false_edges_per_update']:.3f} | {v1['false_edges_per_update']:.3f} | {v2['false_edges_per_update']:.3f} | **{v3['false_edges_per_update']:.3f}** |
| NONE cases with false edges | n/a | {v1['none_diagnostics']['scenarios_with_false_edges']}/6 | {v2_evidence['none_diagnostics']['scenarios_with_false_edges']}/6 | **{frozen_none['scenarios_with_false_edges']}/6** |
| Judge Robustness /4 | {report['base']['judge_metrics']['mean_robustness']:.2f} | {v1['judge_metrics']['mean_robustness']:.2f} | {v2_evidence['judge_metrics']['mean_robustness']:.2f} | **{v3_summary['judge_metrics']['mean_robustness']:.2f}** |

## What changed

- 4,096 rows arranged as 2,048 same-update positive/NONE pairs.
- One pair per update, including all 189 paired ATTACK examples available after the development split.
- Prompt-masked loss averaged over each example before averaging the batch, eliminating the 4.50x output-length weighting imbalance.
- Four original parent episodes reserved for a new 30-case development set; the frozen six episodes and judge rubric stayed unchanged.

The separate development set reached 8/30 exact, 21.1% edge F1, 0.333 false edges/update, and 2/6 NONE false-positive cases. The single frozen pass improved further to 13/30 exact, 35.0% edge F1, 0.300 false edges/update, and 0/6 NONE false-positive cases.

## Before to after

Versus selected v1, v3 gains {d1['edge_f1']:+.3f} edge F1, {d1['exact_patch_accuracy']:+.3f} exact accuracy, and {d1['mean_judge_robustness']:+.3f} judge Robustness while reducing false edges/update by {-d1['false_edges_per_update']:.3f}. Versus v2, it gains {d2['edge_f1']:+.3f} edge F1 and reduces false edges/update by {-d2['false_edges_per_update']:.3f}.

The original false-positive SUPPORT problem is substantially corrected. The remaining failure is conservative underprediction plus label/target confusion: ATTACK is still weakest at 18.2% F1. The model fails the original 85% edge-F1, 80% exact-patch, 75% macro-F1, 0.2 false-edge, and 3.5 Robustness reliability thresholds.

## Reproduce

```bash
{chr(10).join(report['reproduction_commands'])}
```

QT30-derived rows, predictions, records, and judge transcripts remain local and ignored. This report publishes only aggregate metrics, paths, hashes, configuration, and checkpoint file hashes.
"""
    DOC.write_text(markdown, encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(build_report(), indent=2, sort_keys=True))

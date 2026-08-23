#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from flowjudge.dialam_training import file_sha256
from flowjudge.experiment_history import write_experiment_history
from flowjudge.patch_data import PROJECT_ROOT
from flowjudge.patch_metrics import score_patch_prediction_file


DATA_MANIFEST = (
    PROJECT_ROOT / "data" / "dialam" / "training" / "training_v5_manifest.json"
)
TRAINING_RESULT = (
    PROJECT_ROOT
    / "artifacts"
    / "dialam_qlora"
    / "v5_n8192"
    / "remote_training_result.json"
)
CHECKPOINT = TRAINING_RESULT.parent / "adapter"
DEV_GOLD = (
    PROJECT_ROOT / "data" / "dialam" / "training" / "v3_dev_eval_examples.jsonl"
)
BASE_DEV_PREDICTIONS = (
    PROJECT_ROOT
    / "results"
    / "dialam_model_generation"
    / "v5_base_dev"
    / "predictions.jsonl"
)
V5_DEV_PREDICTIONS = (
    PROJECT_ROOT
    / "results"
    / "dialam_model_generation"
    / "v5_n8192_dev"
    / "predictions.jsonl"
)
FROZEN_PREDICTIONS = (
    PROJECT_ROOT
    / "results"
    / "dialam_model_generation"
    / "v5_n8192"
    / "predictions.jsonl"
)
FROZEN_SUMMARY = (
    PROJECT_ROOT / "results" / "dialam_model_eval" / "v5_n8192" / "summary.json"
)
V4_REPORT = PROJECT_ROOT / "reports" / "dialam_v4_class_balanced_rehearsal.json"
V1_V2_V3_REPORT = PROJECT_ROOT / "reports" / "dialam_v1_v2_v3.json"
OUTPUT = PROJECT_ROOT / "reports" / "dialam_v5_pairwise_classification.json"
DOC = PROJECT_ROOT / "docs" / "dialam_v5_results.md"
PREREGISTRATION = PROJECT_ROOT / "docs" / "dialam_v5_preregistration.md"
V5_1_RESULT = PROJECT_ROOT / "reports" / "dialam_v5_1_result.json"


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
            {"path": relative, "sha256": item_hash, "bytes": item.stat().st_size}
        )
    if not files:
        raise FileNotFoundError(f"checkpoint is empty or missing: {path}")
    return digest.hexdigest(), files


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


def _validate_predictions(path: Path, *, target: str) -> list[dict[str, Any]]:
    rows = _load_jsonl(path)
    if len(rows) != 30 or len({row.get("example_id") for row in rows}) != 30:
        raise ValueError(f"{path} is not 30 unique prediction rows")
    expected_dataset_version = "v5" if target == "tuned" else None
    expected_size = 8192 if target == "tuned" else None
    for row in rows:
        if (
            row.get("target") != target
            or row.get("dataset_version") != expected_dataset_version
            or row.get("adapter_size") != expected_size
            or row.get("eval_split") != "v3_dev"
        ):
            raise ValueError(f"invalid v5 {target} development prediction metadata")
        decisions = row.get("pairwise_decisions")
        if not isinstance(decisions, list) or not decisions:
            raise ValueError("v5 predictions must preserve pairwise decision evidence")
        if any(
            item.get("label") not in {"NONE", "SUPPORT", "ATTACK", "REPHRASE"}
            or set(item.get("label_scores", {}))
            != {"NONE", "SUPPORT", "ATTACK", "REPHRASE"}
            for item in decisions
        ):
            raise ValueError("v5 prediction has incomplete allowed-label scores")
    return rows


def _development_checks(
    metrics: dict[str, Any],
    none: dict[str, Any],
    threshold: dict[str, Any],
) -> dict[str, bool]:
    return {
        "exact_patch_accuracy": (
            metrics["exact_patch_accuracy"] >= threshold["exact_patch_accuracy_min"]
        ),
        "edge_f1": metrics["edge_f1"] >= threshold["edge_f1_min"],
        "relation_macro_f1": (
            metrics["relation_macro_f1"] >= threshold["relation_macro_f1_min"]
        ),
        "false_edges_per_update": (
            metrics["false_edges_per_update"]
            <= threshold["false_edges_per_update_max"]
        ),
        "none_scenarios_with_false_edges": (
            none["scenarios_with_false_edges"]
            <= threshold["none_scenarios_with_false_edges_max"]
        ),
        "json_validity": (
            metrics["json_validity_rate"] >= threshold["json_validity_rate_min"]
        ),
        "schema_validity": (
            metrics["schema_validity_rate"]
            >= threshold["schema_validity_rate_min"]
        ),
    }


def _pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def _write_doc(report: dict[str, Any]) -> None:
    base = report["base_development_n8192"]
    v5 = report["v5_development_n8192"]
    v4 = report["v4_development_n8192"]
    decision = report["development_decision"]
    lines = [
        "# DialAM v5 pairwise-classification result",
        "",
        f"Development decision: **{decision}**.",
        "",
        "| Development metric | Pairwise base | v3 / 4096 | v4 / 8192 | v5 / 8192 |",
        "|---|---:|---:|---:|---:|",
    ]
    v3 = report["v3_development_n4096"]
    for label, key in (
        ("Exact patch accuracy", "exact_patch_accuracy"),
        ("Edge precision", "edge_precision"),
        ("Edge recall", "edge_recall"),
        ("Edge F1", "edge_f1"),
        ("Relation macro-F1", "relation_macro_f1"),
    ):
        lines.append(
            f"| {label} | {_pct(base[key])} | {_pct(v3[key])} | {_pct(v4[key])} | **{_pct(v5[key])}** |"
        )
    lines.extend(
        [
            f"| False edges/update | {base['false_edges_per_update']:.3f} | {v3['false_edges_per_update']:.3f} | {v4['false_edges_per_update']:.3f} | **{v5['false_edges_per_update']:.3f}** |",
            f"| NONE cases with false edges | {base['none_diagnostics']['scenarios_with_false_edges']}/6 | {v3['none_diagnostics']['scenarios_with_false_edges']}/6 | {v4['none_diagnostics']['scenarios_with_false_edges']}/6 | **{v5['none_diagnostics']['scenarios_with_false_edges']}/6** |",
            "",
            "## Gate",
            "",
        ]
    )
    for name, passed in report["development_gate_checks"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: `{name}`")
    lines.extend(
        [
            "",
            "## Failure diagnosis",
            "",
            f"V5 produced {v5['false_positive_edges']} false-positive and {v5['false_negative_edges']} false-negative development edges. "
            f"Its six NONE scenarios contained {v5['none_diagnostics']['scenarios_with_false_edges']} cases with a false edge. "
            f"The naturally eligible training-candidate positive rate is {report['failure_diagnosis']['natural_candidate_positive_rate']:.3%}, versus {report['failure_diagnosis']['selected_training_positive_rate']:.1%} in the exact contrast corpus.",
            "",
            "The external block-level contract is unchanged. V5 scores one allowed label",
            "for every supplied candidate ID and deterministically assembles non-NONE",
            "decisions into the exact patch schema.",
            "",
            "The complete chronological record and chart are in",
            "[`docs/dialam_experiment_history.md`](dialam_experiment_history.md).",
            "",
        ]
    )
    if "v5_frozen_n8192" in report:
        frozen = report["v5_frozen_n8192"]
        frozen_v3 = report["v3_frozen_n4096"]
        lines.extend(
            [
                "## Reused frozen benchmark",
                "",
                f"Promotion decision: **{report['promotion_decision']}**.",
                "",
                "| Frozen metric | Selected v3 / 4096 | v5 / 8192 |",
                "|---|---:|---:|",
                f"| Exact patch accuracy | {_pct(frozen_v3['exact_patch_accuracy'])} | **{_pct(frozen['exact_patch_accuracy'])}** |",
                f"| Edge F1 | {_pct(frozen_v3['edge_f1'])} | **{_pct(frozen['edge_f1'])}** |",
                f"| Relation macro-F1 | {_pct(frozen_v3['relation_macro_f1'])} | **{_pct(frozen['relation_macro_f1'])}** |",
                f"| ATTACK F1 | {_pct(frozen_v3['relation_metrics']['ATTACK']['f1'])} | **{_pct(frozen['relation_metrics']['ATTACK']['f1'])}** |",
                f"| False edges/update | {frozen_v3['false_edges_per_update']:.3f} | **{frozen['false_edges_per_update']:.3f}** |",
                f"| Judge Robustness /4 | {frozen_v3['judge_metrics']['mean_robustness']:.3f} | **{frozen['judge_metrics']['mean_robustness']:.3f}** |",
                "",
            ]
        )
        for name, passed in report["promotion_checks"].items():
            lines.append(f"- {'PASS' if passed else 'FAIL'}: `{name}`")
        lines.append("")
    else:
        lines.extend(["## Reused frozen benchmark", ""])
        if V5_1_RESULT.exists():
            lines.extend(
                [
                    "Raw argmax v5 did not advance because its development gate failed.",
                    "The separately preregistered v5.1 prior-correction fallback did pass",
                    "development, completed one frozen evaluation, and replaced v3; see",
                    "[`docs/dialam_v5_1_results.md`](dialam_v5_1_results.md).",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    "Not run: the preregistered development gate did not pass. The frozen",
                    "benchmark and judge therefore remain untouched, and v3 remains selected.",
                    "",
                ]
            )
    lines.extend(
        [
            "## Reproduce",
            "",
            "```bash",
            ".venv/bin/python scripts/build_dialam_training_v5.py",
            "modal run scripts/modal_dialam_qlora.py --action train --size 8192 --dataset-version v5",
            "modal run scripts/modal_dialam_qlora.py --action evaluate --target base --size 8192 --dataset-version v5 --eval-split v3_dev --output-path results/dialam_model_generation/v5_base_dev/predictions.jsonl",
            "modal run scripts/modal_dialam_qlora.py --action evaluate --target tuned --size 8192 --dataset-version v5 --eval-split v3_dev --output-path results/dialam_model_generation/v5_n8192_dev/predictions.jsonl",
            ".venv/bin/python scripts/build_dialam_v5_report.py",
            "```",
            "",
        ]
    )
    DOC.write_text("\n".join(lines), encoding="utf-8")


def build_report() -> dict[str, Any]:
    data_manifest = json.loads(DATA_MANIFEST.read_text(encoding="utf-8"))
    training = json.loads(TRAINING_RESULT.read_text(encoding="utf-8"))
    v4_report = json.loads(V4_REPORT.read_text(encoding="utf-8"))
    selected_report = json.loads(V1_V2_V3_REPORT.read_text(encoding="utf-8"))
    _validate_predictions(BASE_DEV_PREDICTIONS, target="base")
    _validate_predictions(V5_DEV_PREDICTIONS, target="tuned")
    base = score_patch_prediction_file(BASE_DEV_PREDICTIONS, DEV_GOLD)
    v5 = score_patch_prediction_file(V5_DEV_PREDICTIONS, DEV_GOLD)
    base_none = _none_diagnostics(DEV_GOLD, BASE_DEV_PREDICTIONS)
    v5_none = _none_diagnostics(DEV_GOLD, V5_DEV_PREDICTIONS)

    if training["train_sha256"] != data_manifest["output"]["sha256"]:
        raise ValueError("v5 training input differs from the preregistered manifest")
    if training["loss_config"]["name"] != data_manifest["loss_policy"]["name"]:
        raise ValueError("v5 runtime loss differs from the preregistered policy")
    if not training["reload_verified"]:
        raise ValueError("v5 adapter reload was not verified")
    checkpoint_hash, checkpoint_files = _tree_manifest(CHECKPOINT)
    checks = _development_checks(
        v5,
        v5_none,
        data_manifest["development_gate"],
    )
    development_passed = all(checks.values())
    development_decision = (
        "PASS_RUN_REUSED_FROZEN_BENCHMARK"
        if development_passed
        else "RETAIN_V3_V5_FAILED_DEVELOPMENT_GATE"
    )
    report: dict[str, Any] = {
        "schema_version": "dialam_v5_pairwise_classification_result_v1",
        "experiment": "v5/n8192 complete-block pairwise four-label classification",
        "development_decision": development_decision,
        "development_gate_passed": development_passed,
        "development_gate_checks": checks,
        "failed_development_conditions": sorted(
            name for name, passed in checks.items() if not passed
        ),
        "base_development_n8192": {
            **_aggregate(base),
            "none_diagnostics": base_none,
        },
        "v3_development_n4096": v4_report["v3_development_n4096"],
        "v4_development_n8192": v4_report["v4_development_n8192"],
        "v3_frozen_n4096": selected_report["v3_frozen_n4096"],
        "v5_development_n8192": {
            **_aggregate(v5),
            "none_diagnostics": v5_none,
        },
        "failure_diagnosis": {
            "dominant_edge_error": (
                "false_positive"
                if v5["false_positive_edges"] > v5["false_negative_edges"]
                else "false_negative"
                if v5["false_negative_edges"] > v5["false_positive_edges"]
                else "tied"
            ),
            "false_positive_edges": v5["false_positive_edges"],
            "false_negative_edges": v5["false_negative_edges"],
            "none_false_edge_labels": v5_none["false_edge_labels"],
            "natural_candidate_positive_rate": data_manifest[
                "available_candidate_positive_rate"
            ],
            "selected_training_positive_rate": (
                1
                - data_manifest["row_label_counts"]["NONE"]
                / data_manifest["size"]
            ),
            "known_prior_shift": True,
        },
        "controlled_variables": {
            "base_model": training["fixed_config"]["base_model"],
            "optimization_config_unchanged_from_v4": (
                training["fixed_config"] == v4_report["training"]["fixed_config"]
            ),
            "development_eval_sha256": file_sha256(DEV_GOLD),
            "preregistration_sha256": file_sha256(PREREGISTRATION),
            "declared_intervention": (
                "pairwise four-label targets, fixed label-likelihood scoring, "
                "deterministic supplied-ID patch assembly, exact paired batches"
            ),
        },
        "training": {
            "metrics": training["metrics"],
            "fixed_config": training["fixed_config"],
            "loss_config": training["loss_config"],
            "train_sha256": training["train_sha256"],
            "checkpoint_tree_sha256": checkpoint_hash,
            "checkpoint_files": checkpoint_files,
            "reload_probe": training["reload_probe"],
            "reload_probe_label_scores": training["reload_probe_label_scores"],
        },
        "private_artifacts": {
            "base_development_predictions": {
                "path": str(BASE_DEV_PREDICTIONS.relative_to(PROJECT_ROOT)),
                "sha256": file_sha256(BASE_DEV_PREDICTIONS),
            },
            "v5_development_predictions": {
                "path": str(V5_DEV_PREDICTIONS.relative_to(PROJECT_ROOT)),
                "sha256": file_sha256(V5_DEV_PREDICTIONS),
            },
        },
        "reused_frozen_benchmark": {
            "status": (
                "READY_TO_RUN_DEVELOPMENT_GATE_PASSED"
                if development_passed
                else "NOT_RUN_PREREGISTERED_DEVELOPMENT_GATE_FAILED"
            ),
            "candidate_calls": 0,
            "judge_calls": 0,
            "eval_sha256": data_manifest["frozen_evaluation"]["eval_sha256"],
            "judge_rubric_sha256": data_manifest["frozen_evaluation"][
                "judge_rubric_sha256"
            ],
        },
    }

    if FROZEN_SUMMARY.exists() or FROZEN_PREDICTIONS.exists():
        if not development_passed:
            raise ValueError("frozen v5 artifacts exist even though the development gate failed")
        if not FROZEN_SUMMARY.exists() or not FROZEN_PREDICTIONS.exists():
            raise ValueError("v5 frozen prediction and judge artifacts are incomplete")
        frozen_summary = json.loads(FROZEN_SUMMARY.read_text(encoding="utf-8"))
        frozen = frozen_summary["deterministic_metrics"]
        frozen_judge = frozen_summary["judge_metrics"]
        none = frozen.get("none_diagnostics") or _none_diagnostics(
            PROJECT_ROOT / "data" / "dialam" / "balanced_diagnostic_eval_examples.jsonl",
            FROZEN_PREDICTIONS,
        )
        frozen["none_diagnostics"] = none
        promotion_checks = {
            "exact_patch_accuracy": frozen["exact_patch_accuracy"] >= 0.43333333333333335,
            "edge_f1": frozen["edge_f1"] >= 0.40,
            "relation_macro_f1": frozen["relation_macro_f1"] >= 0.39,
            "attack_f1": frozen["relation_metrics"]["ATTACK"]["f1"] >= 0.30,
            "false_edges_per_update": frozen["false_edges_per_update"] <= 0.30,
            "none_scenarios_with_false_edges": none["scenarios_with_false_edges"] <= 1,
            "judge_robustness": frozen_judge["mean_robustness"] >= 2.966666666666667,
            "json_validity": frozen["json_validity_rate"] == 1.0,
            "schema_validity": frozen["schema_validity_rate"] == 1.0,
        }
        report.update(
            {
                "v5_frozen_n8192": {
                    **frozen,
                    "judge_metrics": frozen_judge,
                },
                "promotion_checks": promotion_checks,
                "promotion_decision": (
                    "PROMOTE_V5_AS_FINAL_DIRECTION"
                    if all(promotion_checks.values())
                    else "RETAIN_V3_V5_FAILED_REUSED_FROZEN_PROMOTION_GATE"
                ),
            }
        )
        report["reused_frozen_benchmark"].update(
            {
                "status": "COMPLETE",
                "candidate_calls": 30,
                "judge_calls": 30,
                "predictions_sha256": file_sha256(FROZEN_PREDICTIONS),
                "summary_sha256": file_sha256(FROZEN_SUMMARY),
            }
        )

    OUTPUT.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_doc(report)
    write_experiment_history()
    return report


def main() -> None:
    report = build_report()
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

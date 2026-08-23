#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from flowjudge.dialam_training import file_sha256
from flowjudge.experiment_history import write_experiment_history
from flowjudge.patch_data import PROJECT_ROOT


CALIBRATION = PROJECT_ROOT / "reports" / "dialam_v6_calibration.json"
FROZEN_APPLICATION = PROJECT_ROOT / "reports" / "dialam_v6_frozen_application.json"
FROZEN_SUMMARY = (
    PROJECT_ROOT / "results" / "dialam_model_eval" / "v6_1_n12288" / "summary.json"
)
TRAINING_RESULT = (
    PROJECT_ROOT
    / "artifacts"
    / "dialam_qlora"
    / "v6_n12288"
    / "remote_training_result.json"
)
CHECKPOINT = TRAINING_RESULT.parent / "adapter"
DATA_MANIFEST = (
    PROJECT_ROOT / "data" / "dialam" / "training" / "training_v6_manifest.json"
)
PREREGISTRATION = PROJECT_ROOT / "docs" / "dialam_v6_preregistration.md"
V5_1_RESULT = PROJECT_ROOT / "reports" / "dialam_v5_1_result.json"
DEV_RAW = (
    PROJECT_ROOT
    / "results"
    / "dialam_model_generation"
    / "v6_n12288_dev_raw"
    / "predictions.jsonl"
)
DEV_SELECTED = (
    PROJECT_ROOT
    / "results"
    / "dialam_model_generation"
    / "v6_1_n12288_dev"
    / "predictions.jsonl"
)
FROZEN_RAW = (
    PROJECT_ROOT
    / "results"
    / "dialam_model_generation"
    / "v6_n12288_frozen_raw"
    / "predictions.jsonl"
)
FROZEN_SELECTED = (
    PROJECT_ROOT
    / "results"
    / "dialam_model_generation"
    / "v6_1_n12288"
    / "predictions.jsonl"
)
JUDGE_TRANSCRIPTS = FROZEN_SUMMARY.parent / "judge_transcripts.jsonl"
JUDGE_RECORDS = FROZEN_SUMMARY.parent / "records.jsonl"
OUTPUT = PROJECT_ROOT / "reports" / "dialam_v6_result.json"
DOC = PROJECT_ROOT / "docs" / "dialam_v6_results.md"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }


def _available_artifacts(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    return {name: _artifact(path) for name, path in paths.items() if path.exists()}


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


def _promotion_checks(
    metrics: dict[str, Any],
    none: dict[str, Any],
    thresholds: dict[str, Any],
) -> dict[str, bool]:
    return {
        "exact_patch_accuracy": (
            metrics["exact_patch_accuracy"]
            >= thresholds["exact_patch_accuracy_min"]
        ),
        "edge_f1": metrics["edge_f1"] >= thresholds["edge_f1_min"],
        "relation_macro_f1": (
            metrics["relation_macro_f1"]
            >= thresholds["relation_macro_f1_min"]
        ),
        "attack_f1": (
            metrics["relation_metrics"]["ATTACK"]["f1"]
            >= thresholds["attack_f1_min"]
        ),
        "false_edges_per_update": (
            metrics["false_edges_per_update"]
            <= thresholds["false_edges_per_update_max"]
        ),
        "none_scenarios_with_false_edges": (
            none["scenarios_with_false_edges"]
            <= thresholds["none_scenarios_with_false_edges_max"]
        ),
        "json_validity": (
            metrics["json_validity_rate"]
            >= thresholds["json_validity_rate_min"]
        ),
        "schema_validity": (
            metrics["schema_validity_rate"]
            >= thresholds["schema_validity_rate_min"]
        ),
    }


def _pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def _metric_row(label: str, key: str, baseline: dict, selected: dict) -> str:
    delta = 100 * (selected[key] - baseline[key])
    return (
        f"| {label} | {_pct(baseline[key])} | **{_pct(selected[key])}** | "
        f"{delta:+.1f} pp |"
    )


def _write_doc(report: dict[str, Any]) -> None:
    baseline_dev = report["v5_1_development_n8192"]
    selected_dev = report["v6_1_development_n12288"]
    lines = [
        "# DialAM v6 Vives→QT30 transfer result",
        "",
        f"Selection decision: **{report['selection_decision']}**.",
        "",
        "V6 tested one preregistered intervention: a balanced VivesDebate",
        "relation-classification warm-up followed by the exact unchanged QT30 v5",
        "target corpus. The Qwen3-0.6B base, QLoRA configuration, pairwise prompt,",
        "scorer, frozen benchmark, and judge rubric were held fixed.",
        "",
        "## Episode-disjoint development gate",
        "",
        "| Metric | v5.1 / 8192 | v6.1 / 12288 | Delta |",
        "|---|---:|---:|---:|",
        _metric_row("Exact patch accuracy", "exact_patch_accuracy", baseline_dev, selected_dev),
        _metric_row("Edge F1", "edge_f1", baseline_dev, selected_dev),
        _metric_row("Relation macro-F1", "relation_macro_f1", baseline_dev, selected_dev),
        f"| False edges/update | {baseline_dev['false_edges_per_update']:.3f} | **{selected_dev['false_edges_per_update']:.3f}** | {selected_dev['false_edges_per_update'] - baseline_dev['false_edges_per_update']:+.3f} |",
        f"| NONE cases with false edges | {baseline_dev['none_diagnostics']['scenarios_with_false_edges']}/6 | **{selected_dev['none_diagnostics']['scenarios_with_false_edges']}/6** | {selected_dev['none_diagnostics']['scenarios_with_false_edges'] - baseline_dev['none_diagnostics']['scenarios_with_false_edges']:+d} |",
        "",
        "Development checks:",
        "",
    ]
    for name, passed in report["development_gate_checks"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: `{name}`")

    if "v6_1_frozen_n12288" not in report:
        lines.extend(
            [
                "",
                "## Frozen benchmark",
                "",
                "Not run. The preregistered development gate failed, so no v6",
                "frozen predictions were generated or inspected and v5.1 remains",
                "the selected submission model.",
            ]
        )
    else:
        baseline_frozen = report["v5_1_frozen_n8192"]
        selected_frozen = report["v6_1_frozen_n12288"]
        lines.extend(
            [
                "",
                "## One locked-margin frozen evaluation",
                "",
                "| Metric | v5.1 / 8192 | v6.1 / 12288 | Delta |",
                "|---|---:|---:|---:|",
                _metric_row("Exact patch accuracy", "exact_patch_accuracy", baseline_frozen, selected_frozen),
                _metric_row("Edge F1", "edge_f1", baseline_frozen, selected_frozen),
                _metric_row("Relation macro-F1", "relation_macro_f1", baseline_frozen, selected_frozen),
                f"| ATTACK F1 | {_pct(baseline_frozen['relation_metrics']['ATTACK']['f1'])} | **{_pct(selected_frozen['relation_metrics']['ATTACK']['f1'])}** | {100 * (selected_frozen['relation_metrics']['ATTACK']['f1'] - baseline_frozen['relation_metrics']['ATTACK']['f1']):+.1f} pp |",
                f"| False edges/update | {baseline_frozen['false_edges_per_update']:.3f} | **{selected_frozen['false_edges_per_update']:.3f}** | {selected_frozen['false_edges_per_update'] - baseline_frozen['false_edges_per_update']:+.3f} |",
                f"| NONE cases with false edges | {baseline_frozen['none_diagnostics']['scenarios_with_false_edges']}/6 | **{selected_frozen['none_diagnostics']['scenarios_with_false_edges']}/6** | {selected_frozen['none_diagnostics']['scenarios_with_false_edges'] - baseline_frozen['none_diagnostics']['scenarios_with_false_edges']:+d} |",
                "",
                "Frozen promotion checks:",
                "",
            ]
        )
        for name, passed in report["promotion_checks"].items():
            lines.append(f"- {'PASS' if passed else 'FAIL'}: `{name}`")
        if report.get("judge_metrics"):
            judge = report["judge_metrics"]
            lines.extend(
                [
                    "",
                    f"The unchanged blinded judge completed 30/30 calls: Spec adherence {judge['mean_spec_adherence']:.3f}/4 and Robustness {judge['mean_robustness']:.3f}/4.",
                ]
            )
        else:
            lines.extend(
                [
                    "",
                    "The judge was not run because deterministic promotion did not pass.",
                ]
            )

    lines.extend(
        [
            "",
            "## Reproduce",
            "",
            "```bash",
            ".venv/bin/python scripts/build_dialam_training_v6.py",
            "modal run scripts/modal_dialam_qlora.py --action train --size 12288 --dataset-version v6",
            "modal run scripts/modal_dialam_qlora.py --action evaluate --target tuned --size 12288 --dataset-version v6 --eval-split v3_dev --output-path results/dialam_model_generation/v6_n12288_dev_raw/predictions.jsonl",
            ".venv/bin/python scripts/calibrate_dialam_v6.py",
            "# Run the locked frozen sequence below only when the development report says PASS.",
            "modal run scripts/modal_dialam_qlora.py --action evaluate --target tuned --size 12288 --dataset-version v6 --eval-split frozen --output-path results/dialam_model_generation/v6_n12288_frozen_raw/predictions.jsonl",
            ".venv/bin/python scripts/apply_dialam_v5_margin.py --predictions results/dialam_model_generation/v6_n12288_frozen_raw/predictions.jsonl --examples data/dialam/balanced_diagnostic_eval_examples.jsonl --margin <locked-development-margin> --dataset-version v6.1 --output results/dialam_model_generation/v6_1_n12288/predictions.jsonl --summary reports/dialam_v6_frozen_application.json",
            "# Run the judge only when every deterministic frozen promotion check passes.",
            ".venv/bin/python scripts/evaluate_dialam_model.py --predictions results/dialam_model_generation/v6_1_n12288/predictions.jsonl --output-dir results/dialam_model_eval/v6_1_n12288 --approval APPROVE_DIALAM_MODEL_JUDGING",
            ".venv/bin/python scripts/build_dialam_v6_report.py",
            "```",
            "",
        ]
    )
    DOC.write_text("\n".join(lines), encoding="utf-8")


def build_report() -> dict[str, Any]:
    calibration = _load(CALIBRATION)
    training = _load(TRAINING_RESULT)
    data = _load(DATA_MANIFEST)
    baseline = _load(V5_1_RESULT)
    selected = calibration["selected"]

    if training["dataset_version"] != "v6" or training["size"] != 12288:
        raise ValueError("v6 training manifest identity is wrong")
    if training["train_sha256"] != data["output"]["sha256"]:
        raise ValueError("v6 training bytes differ from the preregistered manifest")
    if training["fixed_config"] != baseline["training"]["fixed_config"]:
        raise ValueError("v6 QLoRA configuration differs from v5.1")
    if not training["reload_verified"]:
        raise ValueError("v6 checkpoint reload smoke did not pass")
    checkpoint_hash, checkpoint_files = _tree_manifest(CHECKPOINT)

    report: dict[str, Any] = {
        "schema_version": "dialam_v6_vives_transfer_result_v1",
        "experiment": "v6.1/n12288 VivesDebate warm-up then exact QT30 v5 target stage",
        "selection_decision": calibration["development_decision"],
        "development_gate_passed": selected["development_gate_passed"],
        "development_gate_checks": selected["development_gate_checks"],
        "failed_development_conditions": sorted(
            name
            for name, passed in selected["development_gate_checks"].items()
            if not passed
        ),
        "v5_1_development_n8192": baseline["v5_1_development_n8192"],
        "v6_1_development_n12288": {
            **selected["metrics"],
            "none_diagnostics": selected["none_diagnostics"],
        },
        "v5_1_frozen_n8192": baseline["v5_1_frozen_n8192"],
        "training": {
            "base_model": training["fixed_config"]["base_model"],
            "size": training["size"],
            "dataset_version": training["dataset_version"],
            "train_sha256": training["train_sha256"],
            "fixed_config": training["fixed_config"],
            "loss_config": training["loss_config"],
            "metrics": training["metrics"],
            "reload_verified": training["reload_verified"],
            "reload_probe": training["reload_probe"],
            "reload_probe_label_scores": training["reload_probe_label_scores"],
            "checkpoint_tree_sha256": checkpoint_hash,
            "checkpoint_files": checkpoint_files,
        },
        "curriculum": data["curriculum"],
        "row_label_counts": data["row_label_counts"],
        "row_source_counts": data["row_source_counts"],
        "frozen_status": calibration["frozen_status"],
        "frozen_candidate_calls": 0,
        "frozen_judge_calls": 0,
        "promotion_passed": False,
        "artifacts": _available_artifacts(
            {
                "preregistration": PREREGISTRATION,
                "data_manifest": DATA_MANIFEST,
                "training_result": TRAINING_RESULT,
                "development_raw_predictions": DEV_RAW,
                "development_selected_predictions": DEV_SELECTED,
                "calibration": CALIBRATION,
            }
        ),
    }

    frozen_files = (FROZEN_RAW, FROZEN_SELECTED, FROZEN_APPLICATION, FROZEN_SUMMARY)
    if not selected["development_gate_passed"]:
        if any(path.exists() for path in frozen_files):
            raise ValueError(
                "v6 frozen artifacts exist despite the failed preregistered development gate"
            )
    else:
        if not all(path.exists() for path in frozen_files[:3]):
            raise FileNotFoundError(
                "development passed; complete the one frozen generation and locked-margin application"
            )
        frozen_application = _load(FROZEN_APPLICATION)
        if frozen_application["dataset_version"] != "v6.1":
            raise ValueError("frozen application is not labeled v6.1")
        if frozen_application["margin"] != selected["margin"]:
            raise ValueError("frozen v6 margin differs from development selection")
        if (
            frozen_application["examples_sha256"]
            != baseline["evaluation_accounting"]["eval_sha256"]
        ):
            raise ValueError("v6 reused frozen evaluation hash changed")

        metrics = frozen_application["metrics"]
        none = frozen_application["none_diagnostics"]
        checks = _promotion_checks(metrics, none, data["frozen_promotion_gate"])
        deterministic_passed = all(checks.values())
        report.update(
            {
                "selection_decision": (
                    "AWAITING_REQUIRED_FROZEN_JUDGE"
                    if deterministic_passed and not FROZEN_SUMMARY.exists()
                    else "PROMOTE_V6_1_AS_NEXT_DIRECTION"
                    if deterministic_passed
                    else "RETAIN_V5_1_V6_FAILED_FROZEN_PROMOTION_GATE"
                ),
                "frozen_status": (
                    "DETERMINISTIC_PASS_JUDGE_REQUIRED"
                    if deterministic_passed and not FROZEN_SUMMARY.exists()
                    else "COMPLETE"
                ),
                "frozen_candidate_calls": 30,
                "v6_1_frozen_n12288": {
                    **metrics,
                    "none_diagnostics": none,
                },
                "promotion_checks": checks,
                "deterministic_promotion_passed": deterministic_passed,
                "failed_promotion_conditions": sorted(
                    name for name, passed in checks.items() if not passed
                ),
            }
        )
        report["artifacts"].update(
            _available_artifacts(
                {
                    "frozen_raw_predictions": FROZEN_RAW,
                    "frozen_selected_predictions": FROZEN_SELECTED,
                    "frozen_application": FROZEN_APPLICATION,
                }
            )
        )

        if FROZEN_SUMMARY.exists():
            if not deterministic_passed:
                raise ValueError("v6 judge ran despite failed deterministic promotion")
            summary = _load(FROZEN_SUMMARY)
            if summary["eval_sha256"] != baseline["evaluation_accounting"]["eval_sha256"]:
                raise ValueError("v6 judge summary used a different frozen evaluation")
            if (
                summary["judge_rubric_sha256"]
                != baseline["evaluation_accounting"]["judge_rubric_sha256"]
            ):
                raise ValueError("v6 judge rubric differs from the frozen rubric")
            if summary["judge_identity_blinded"] is not True:
                raise ValueError("v6 judge was not identity blinded")
            if summary["deterministic_metrics"] != metrics:
                raise ValueError("v6 judge summary deterministic metrics changed")
            report.update(
                {
                    "judge_metrics": summary["judge_metrics"],
                    "frozen_judge_calls": 30,
                    "promotion_passed": True,
                }
            )
            report["v6_1_frozen_n12288"]["judge_metrics"] = summary[
                "judge_metrics"
            ]
            report["artifacts"].update(
                _available_artifacts(
                    {
                        "frozen_summary": FROZEN_SUMMARY,
                        "judge_transcripts": JUDGE_TRANSCRIPTS,
                        "judge_records": JUDGE_RECORDS,
                    }
                )
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
    print(
        json.dumps(
            {
                "selection_decision": report["selection_decision"],
                "development_gate_checks": report["development_gate_checks"],
                "promotion_checks": report.get("promotion_checks"),
                "promotion_passed": report["promotion_passed"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

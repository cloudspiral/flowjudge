#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from flowjudge.dialam_training import file_sha256
from flowjudge.experiment_history import write_experiment_history
from flowjudge.patch_data import PROJECT_ROOT


CALIBRATION = PROJECT_ROOT / "reports" / "dialam_v7_calibration.json"
FROZEN_APPLICATION = PROJECT_ROOT / "reports" / "dialam_v7_frozen_application.json"
FROZEN_SUMMARY = (
    PROJECT_ROOT / "results" / "dialam_model_eval" / "v7_1_n8192" / "summary.json"
)
TRAINING_RESULT = (
    PROJECT_ROOT
    / "artifacts"
    / "dialam_qlora"
    / "v7_n8192"
    / "remote_training_result.json"
)
IDEMPOTENT_RESULT = (
    PROJECT_ROOT
    / "artifacts"
    / "dialam_qlora"
    / "v7_n8192"
    / "idempotent_reuse_result.json"
)
DATA_MANIFEST = (
    PROJECT_ROOT / "data" / "dialam" / "training" / "training_v7_manifest.json"
)
RESUME_SMOKE = PROJECT_ROOT / "reports" / "dialam_v7_resume_smoke.json"
PREREGISTRATION = PROJECT_ROOT / "docs" / "dialam_v7_preregistration.md"
V5_1_RESULT = PROJECT_ROOT / "reports" / "dialam_v5_1_result.json"
DEV_RAW = (
    PROJECT_ROOT
    / "results"
    / "dialam_model_generation"
    / "v7_n8192_dev_raw"
    / "predictions.jsonl"
)
DEV_SELECTED = (
    PROJECT_ROOT
    / "results"
    / "dialam_model_generation"
    / "v7_1_n8192_dev"
    / "predictions.jsonl"
)
FROZEN_RAW = (
    PROJECT_ROOT
    / "results"
    / "dialam_model_generation"
    / "v7_n8192_frozen_raw"
    / "predictions.jsonl"
)
FROZEN_SELECTED = (
    PROJECT_ROOT
    / "results"
    / "dialam_model_generation"
    / "v7_1_n8192"
    / "predictions.jsonl"
)
JUDGE_TRANSCRIPTS = FROZEN_SUMMARY.parent / "judge_transcripts.jsonl"
JUDGE_RECORDS = FROZEN_SUMMARY.parent / "records.jsonl"
OUTPUT = PROJECT_ROOT / "reports" / "dialam_v7_result.json"
DOC = PROJECT_ROOT / "docs" / "dialam_v7_results.md"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }


def _available(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    return {name: _artifact(path) for name, path in paths.items() if path.exists()}


def _aggregate(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metrics.items()
        if key not in {"failure_cases", "scenario_failure_cases"}
    }


def _promotion_checks(
    metrics: dict[str, Any], none: dict[str, Any], thresholds: dict[str, Any]
) -> dict[str, bool]:
    return {
        "exact_patch_accuracy": metrics["exact_patch_accuracy"]
        >= thresholds["exact_patch_accuracy_min"],
        "edge_f1": metrics["edge_f1"] >= thresholds["edge_f1_min"],
        "relation_macro_f1": metrics["relation_macro_f1"]
        >= thresholds["relation_macro_f1_min"],
        "attack_f1": metrics["relation_metrics"]["ATTACK"]["f1"]
        >= thresholds["attack_f1_min"],
        "false_edges_per_update": metrics["false_edges_per_update"]
        <= thresholds["false_edges_per_update_max"],
        "none_scenarios_with_false_edges": none["scenarios_with_false_edges"]
        <= thresholds["none_scenarios_with_false_edges_max"],
        "json_validity": metrics["json_validity_rate"]
        >= thresholds["json_validity_rate_min"],
        "schema_validity": metrics["schema_validity_rate"]
        >= thresholds["schema_validity_rate_min"],
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
    baseline = report["v5_1_development_n8192"]
    selected = report["v7_1_development_n8192"]
    lines = [
        "# DialAM v7 QT30 reciprocal-preference result",
        "",
        f"Selection decision: **{report['selection_decision']}**.",
        "",
        "V7 tested one preregistered intervention: one epoch of reciprocal",
        "candidate-preference continuation from the frozen v5 adapter on the exact",
        "QT30 v5 pair corpus. No new corpus or frozen-derived training signal was used.",
        "",
        "The interruption/resume smoke passed before the full run. Full Trainer state",
        "was committed periodically, and the final adapter was reload-verified.",
        "",
        "## Episode-disjoint development gate",
        "",
        "| Metric | v5.1 / 8192 | v7.1 / 8192 | Delta |",
        "|---|---:|---:|---:|",
        _metric_row("Exact patch accuracy", "exact_patch_accuracy", baseline, selected),
        _metric_row("Edge F1", "edge_f1", baseline, selected),
        _metric_row("Relation macro-F1", "relation_macro_f1", baseline, selected),
        f"| False edges/update | {baseline['false_edges_per_update']:.3f} | **{selected['false_edges_per_update']:.3f}** | {selected['false_edges_per_update'] - baseline['false_edges_per_update']:+.3f} |",
        f"| NONE cases with false edges | {baseline['none_diagnostics']['scenarios_with_false_edges']}/6 | **{selected['none_diagnostics']['scenarios_with_false_edges']}/6** | {selected['none_diagnostics']['scenarios_with_false_edges'] - baseline['none_diagnostics']['scenarios_with_false_edges']:+d} |",
        "",
        "Development checks:",
        "",
    ]
    for name, passed in report["development_gate_checks"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: `{name}`")

    diagnosis = report["failure_diagnosis"]
    lines.extend(
        [
            "",
            "## Development diagnosis",
            "",
            f"V7 produced {diagnosis['false_positive_edges']} false-positive edges and "
            f"{diagnosis['false_negative_edges']} false negatives. SUPPORT caused "
            f"{diagnosis['support_false_positive_edges']} of the false positives, and "
            f"{diagnosis['none_scenarios_with_false_edges']}/6 all-NONE scenarios "
            "received at least one false edge.",
            "",
            "The reciprocal preference continuation therefore moved the relation-vs-NONE",
            "boundary in the wrong direction on development. The selected fixed-grid",
            "margin also saturated at its registered maximum of 3.0, so a separately",
            "preregistered score-scale calibration is the only warranted zero-training",
            "follow-up before rejecting the checkpoint itself.",
        ]
    )

    if "v7_1_frozen_n8192" not in report:
        lines.extend(["", "## Frozen benchmark", ""])
        if report["development_gate_passed"]:
            lines.extend(
                [
                    "Not yet run. The development gate passed, but the locked frozen",
                    "sequence has not completed.",
                ]
            )
        else:
            lines.extend(
                [
                    "Not run. The preregistered development gate failed, so no v7",
                    "frozen predictions were generated or inspected and v5.1 remains",
                    "the selected submission model.",
                ]
            )
    else:
        baseline_frozen = report["v5_1_frozen_n8192"]
        selected_frozen = report["v7_1_frozen_n8192"]
        lines.extend(
            [
                "",
                "## One locked-margin frozen evaluation",
                "",
                "| Metric | v5.1 / 8192 | v7.1 / 8192 | Delta |",
                "|---|---:|---:|---:|",
                _metric_row(
                    "Exact patch accuracy",
                    "exact_patch_accuracy",
                    baseline_frozen,
                    selected_frozen,
                ),
                _metric_row("Edge F1", "edge_f1", baseline_frozen, selected_frozen),
                _metric_row(
                    "Relation macro-F1",
                    "relation_macro_f1",
                    baseline_frozen,
                    selected_frozen,
                ),
                f"| False edges/update | {baseline_frozen['false_edges_per_update']:.3f} | **{selected_frozen['false_edges_per_update']:.3f}** | {selected_frozen['false_edges_per_update'] - baseline_frozen['false_edges_per_update']:+.3f} |",
                f"| NONE cases with false edges | {baseline_frozen['none_diagnostics']['scenarios_with_false_edges']}/6 | **{selected_frozen['none_diagnostics']['scenarios_with_false_edges']}/6** | {selected_frozen['none_diagnostics']['scenarios_with_false_edges'] - baseline_frozen['none_diagnostics']['scenarios_with_false_edges']:+d} |",
                "",
                "Frozen promotion checks:",
                "",
            ]
        )
        for name, passed in report["promotion_checks"].items():
            lines.append(f"- {'PASS' if passed else 'FAIL'}: `{name}`")

    lines.extend(
        [
            "",
            "## Reproduce",
            "",
            "```bash",
            ".venv/bin/python scripts/build_dialam_training_v7.py",
            "modal run --detach scripts/modal_dialam_qlora.py --action train --size 8192 --dataset-version v7 --resume-mode auto",
            "modal run scripts/modal_dialam_qlora.py --action evaluate --target tuned --size 8192 --dataset-version v7 --eval-split v3_dev --output-path results/dialam_model_generation/v7_n8192_dev_raw/predictions.jsonl",
            ".venv/bin/python scripts/calibrate_dialam_v7.py",
            "# Run frozen generation only if the development report says PASS.",
            "modal run scripts/modal_dialam_qlora.py --action evaluate --target tuned --size 8192 --dataset-version v7 --eval-split frozen --output-path results/dialam_model_generation/v7_n8192_frozen_raw/predictions.jsonl",
            ".venv/bin/python scripts/apply_dialam_v5_margin.py --predictions results/dialam_model_generation/v7_n8192_frozen_raw/predictions.jsonl --examples data/dialam/balanced_diagnostic_eval_examples.jsonl --margin <locked-development-margin> --dataset-version v7.1 --output results/dialam_model_generation/v7_1_n8192/predictions.jsonl --summary reports/dialam_v7_frozen_application.json",
            "# Run the judge only if every deterministic promotion condition passes.",
            ".venv/bin/python scripts/evaluate_dialam_model.py --predictions results/dialam_model_generation/v7_1_n8192/predictions.jsonl --output-dir results/dialam_model_eval/v7_1_n8192 --approval APPROVE_DIALAM_MODEL_JUDGING",
            ".venv/bin/python scripts/build_dialam_v7_report.py",
            "```",
            "",
        ]
    )
    DOC.write_text("\n".join(lines), encoding="utf-8")


def build_report() -> dict[str, Any]:
    calibration = _load(CALIBRATION)
    training = _load(TRAINING_RESULT)
    idempotent = _load(IDEMPOTENT_RESULT)
    data = _load(DATA_MANIFEST)
    baseline = _load(V5_1_RESULT)
    resume_smoke = _load(RESUME_SMOKE)
    selected = calibration["selected"]

    if training["dataset_version"] != "v7" or training["size"] != 8192:
        raise ValueError("v7 training manifest identity is wrong")
    if training["train_sha256"] != data["output"]["sha256"]:
        raise ValueError("v7 training bytes differ from the preregistered manifest")
    if (
        training["source_adapter"]["tree_sha256"]
        != data["continuation"]["source_adapter_tree_sha256"]
    ):
        raise ValueError("v7 did not continue from the frozen v5 adapter")
    if not training["reload_verified"]:
        raise ValueError("v7 final adapter reload failed")
    if idempotent.get("idempotent_reuse") is not True:
        raise ValueError("v7 completed-run idempotence check did not short circuit")
    if (
        idempotent["global_step"] != training["global_step"]
        or idempotent["adapter_tree_sha256"] != training["adapter_tree_sha256"]
    ):
        raise ValueError("v7 idempotent rerun returned different completed state")
    if not resume_smoke["all_checks_passed"]:
        raise ValueError("v7 full run proceeded without a passing resume smoke")

    selected_metrics = {**selected["metrics"], "none_diagnostics": selected["none_diagnostics"]}
    report: dict[str, Any] = {
        "schema_version": "dialam_v7_preference_result_v1",
        "experiment": "v7.1/n8192 QT30 reciprocal-preference continuation",
        "selection_decision": calibration["development_decision"],
        "development_gate_passed": selected["development_gate_passed"],
        "development_gate_checks": selected["development_gate_checks"],
        "failed_development_conditions": sorted(
            name
            for name, passed in selected["development_gate_checks"].items()
            if not passed
        ),
        "v5_1_development_n8192": baseline["v5_1_development_n8192"],
        "v7_1_development_n8192": selected_metrics,
        "v5_1_frozen_n8192": baseline["v5_1_frozen_n8192"],
        "training": {
            "modal_training_app_id": "ap-eDrcBEahd3VhgCFVdLqgYP",
            "modal_development_eval_app_id": "ap-riM3TVuUmvwzB5sDFTMUZt",
            "size": training["size"],
            "dataset_version": training["dataset_version"],
            "train_sha256": training["train_sha256"],
            "fixed_config": training["fixed_config"],
            "loss_config": training["loss_config"],
            "metrics": training["metrics"],
            "global_step": training["global_step"],
            "adapter_tree_sha256": training["adapter_tree_sha256"],
            "adapter_files": training["adapter_files"],
            "reload_verified": training["reload_verified"],
            "resumability": training["resumability"],
            "idempotent_reuse_check": {
                "returned_without_training": True,
                "global_step": idempotent["global_step"],
                "adapter_tree_sha256": idempotent["adapter_tree_sha256"],
                "artifact_sha256": file_sha256(IDEMPOTENT_RESULT),
            },
        },
        "failure_diagnosis": {
            "false_positive_edges": selected["metrics"]["false_positive_edges"],
            "false_negative_edges": selected["metrics"]["false_negative_edges"],
            "support_false_positive_edges": selected["metrics"]["relation_metrics"]
            ["SUPPORT"]["false_positive"],
            "none_scenarios_with_false_edges": selected["none_diagnostics"]
            ["scenarios_with_false_edges"],
        },
        "frozen_status": calibration["frozen_status"],
        "frozen_candidate_calls": 0,
        "frozen_judge_calls": 0,
        "promotion_passed": False,
        "artifacts": _available(
            {
                "preregistration": PREREGISTRATION,
                "data_manifest": DATA_MANIFEST,
                "resume_smoke": RESUME_SMOKE,
                "training_result": TRAINING_RESULT,
                "idempotent_training_result": IDEMPOTENT_RESULT,
                "development_raw_predictions": DEV_RAW,
                "development_selected_predictions": DEV_SELECTED,
                "calibration": CALIBRATION,
            }
        ),
    }

    frozen_files = (FROZEN_RAW, FROZEN_SELECTED, FROZEN_APPLICATION, FROZEN_SUMMARY)
    if not selected["development_gate_passed"]:
        if any(path.exists() for path in frozen_files):
            raise ValueError("v7 frozen artifacts exist despite the failed development gate")
    elif FROZEN_APPLICATION.exists():
        if not FROZEN_RAW.exists() or not FROZEN_SELECTED.exists():
            raise FileNotFoundError("v7 frozen application lacks raw or selected predictions")
        frozen = _load(FROZEN_APPLICATION)
        if frozen["dataset_version"] != "v7.1":
            raise ValueError("frozen application is not labeled v7.1")
        if frozen["margin"] != selected["margin"]:
            raise ValueError("frozen v7 margin differs from development selection")
        if frozen["examples_sha256"] != baseline["evaluation_accounting"]["eval_sha256"]:
            raise ValueError("v7 frozen evaluation hash changed")
        metrics = frozen["metrics"]
        none = frozen["none_diagnostics"]
        checks = _promotion_checks(metrics, none, data["frozen_promotion_gate"])
        deterministic_passed = all(checks.values())
        report.update(
            {
                "selection_decision": (
                    "AWAITING_REQUIRED_FROZEN_JUDGE"
                    if deterministic_passed and not FROZEN_SUMMARY.exists()
                    else "PROMOTE_V7_1_AS_NEXT_DIRECTION"
                    if deterministic_passed
                    else "RETAIN_V5_1_V7_FAILED_FROZEN_PROMOTION_GATE"
                ),
                "frozen_status": (
                    "DETERMINISTIC_PASS_JUDGE_REQUIRED"
                    if deterministic_passed and not FROZEN_SUMMARY.exists()
                    else "COMPLETE"
                ),
                "frozen_candidate_calls": 30,
                "v7_1_frozen_n8192": {**metrics, "none_diagnostics": none},
                "promotion_checks": checks,
                "deterministic_promotion_passed": deterministic_passed,
                "failed_promotion_conditions": sorted(
                    name for name, passed in checks.items() if not passed
                ),
            }
        )
        report["artifacts"].update(
            _available(
                {
                    "frozen_raw_predictions": FROZEN_RAW,
                    "frozen_selected_predictions": FROZEN_SELECTED,
                    "frozen_application": FROZEN_APPLICATION,
                }
            )
        )
        if FROZEN_SUMMARY.exists():
            if not deterministic_passed:
                raise ValueError("v7 judge ran despite failed deterministic promotion")
            summary = _load(FROZEN_SUMMARY)
            if summary["eval_sha256"] != baseline["evaluation_accounting"]["eval_sha256"]:
                raise ValueError("v7 judge used a different frozen evaluation")
            if (
                summary["judge_rubric_sha256"]
                != baseline["evaluation_accounting"]["judge_rubric_sha256"]
            ):
                raise ValueError("v7 judge rubric changed")
            if summary["judge_identity_blinded"] is not True:
                raise ValueError("v7 judge was not identity blinded")
            if _aggregate(summary["deterministic_metrics"]) != _aggregate(metrics):
                raise ValueError("v7 judge deterministic metrics changed")
            report.update(
                {
                    "judge_metrics": summary["judge_metrics"],
                    "frozen_judge_calls": 30,
                    "promotion_passed": True,
                }
            )
            report["artifacts"].update(
                _available(
                    {
                        "frozen_summary": FROZEN_SUMMARY,
                        "judge_transcripts": JUDGE_TRANSCRIPTS,
                        "judge_records": JUDGE_RECORDS,
                    }
                )
            )

    OUTPUT.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_doc(report)
    write_experiment_history()
    return report


def main() -> None:
    report = build_report()
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

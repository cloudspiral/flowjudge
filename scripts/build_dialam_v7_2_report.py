#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from flowjudge.dialam_training import file_sha256
from flowjudge.experiment_history import write_experiment_history
from flowjudge.patch_data import PROJECT_ROOT


CALIBRATION = PROJECT_ROOT / "reports" / "dialam_v7_2_calibration.json"
V7_RESULT = PROJECT_ROOT / "reports" / "dialam_v7_result.json"
PREREGISTRATION = PROJECT_ROOT / "docs" / "dialam_v7_2_preregistration.md"
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
    / "v7_2_n8192_dev"
    / "predictions.jsonl"
)
FROZEN_PATHS = (
    PROJECT_ROOT
    / "results"
    / "dialam_model_generation"
    / "v7_n8192_frozen_raw"
    / "predictions.jsonl",
    PROJECT_ROOT
    / "results"
    / "dialam_model_generation"
    / "v7_2_n8192"
    / "predictions.jsonl",
    PROJECT_ROOT / "reports" / "dialam_v7_2_frozen_application.json",
)
OUTPUT = PROJECT_ROOT / "reports" / "dialam_v7_2_result.json"
DOC = PROJECT_ROOT / "docs" / "dialam_v7_2_results.md"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
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
    v7_1 = report["v7_1_development_n8192"]
    v7_2 = report["v7_2_development_n8192"]
    v5_1 = report["v5_1_development_n8192"]
    calibration = report["calibration"]
    diagnosis = report["failure_diagnosis"]
    lines = [
        "# DialAM v7.2 transition-complete calibration result",
        "",
        f"Selection decision: **{report['selection_decision']}**.",
        "",
        "V7.2 made no model calls and performed no training. It reused the exact",
        "v7 raw development scores and evaluated every possible NONE-margin output",
        "configuration above the v7.1 boundary.",
        "",
        "## Development result",
        "",
        f"The score-only rescue evaluated {calibration['candidate_margin_count']} unique",
        f"decision states and selected margin `{calibration['selected_margin']}`.",
        "",
        "| Metric | v7.1 / 8192 | v7.2 / 8192 | Delta |",
        "|---|---:|---:|---:|",
        _metric_row("Exact patch accuracy", "exact_patch_accuracy", v7_1, v7_2),
        _metric_row("Edge F1", "edge_f1", v7_1, v7_2),
        _metric_row("Relation macro-F1", "relation_macro_f1", v7_1, v7_2),
        f"| False edges/update | {v7_1['false_edges_per_update']:.3f} | **{v7_2['false_edges_per_update']:.3f}** | {v7_2['false_edges_per_update'] - v7_1['false_edges_per_update']:+.3f} |",
        f"| NONE cases with false edges | {v7_1['none_diagnostics']['scenarios_with_false_edges']}/6 | **{v7_2['none_diagnostics']['scenarios_with_false_edges']}/6** | {v7_2['none_diagnostics']['scenarios_with_false_edges'] - v7_1['none_diagnostics']['scenarios_with_false_edges']:+d} |",
        "",
        "Development checks:",
        "",
    ]
    for name, passed in report["development_gate_checks"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: `{name}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The larger margin recovered exact accuracy to the v5.1 development",
            f"baseline ({_pct(v5_1['exact_patch_accuracy'])}) and reduced false positives",
            f"from {v7_1['false_positive_edges']} to {v7_2['false_positive_edges']}.",
            f"However, edge F1 remained {_pct(v7_2['edge_f1'])} versus the locked",
            "56.0% gate, macro-F1 remained below its gate, and false edges/update",
            "remained above 0.300. The checkpoint therefore was not eligible for",
            "frozen evaluation.",
            "",
            f"SUPPORT still accounted for {diagnosis['support_false_positive_edges']} of",
            f"{diagnosis['false_positive_edges']} false positives. Calibration corrected",
            "much of the score-scale shift, but cannot repair the remaining relation",
            "ranking errors—especially weak REPHRASE recall—without changing training.",
            "",
            "## Frozen benchmark",
            "",
            "Not run. The unchanged development gate failed. Candidate calls: 0;",
            "judge calls: 0. V5.1 remains the selected submission model.",
            "",
            "## Reproduce",
            "",
            "```bash",
            "PYTHONPATH=src .venv/bin/python scripts/calibrate_dialam_v7_2.py",
            "PYTHONPATH=src .venv/bin/python scripts/build_dialam_v7_2_report.py",
            "```",
            "",
        ]
    )
    DOC.write_text("\n".join(lines), encoding="utf-8")


def build_report() -> dict[str, Any]:
    calibration = _load(CALIBRATION)
    v7 = _load(V7_RESULT)
    selected = calibration["selected"]

    if not calibration["post_v7_1_preregistered_before_extended_scan"]:
        raise ValueError("v7.2 extended calibration was not preregistered")
    if calibration["new_training"] or calibration["new_model_calls"]:
        raise ValueError("v7.2 must be a score-only calibration")
    if calibration["artifacts"]["input_predictions_sha256"] != v7["artifacts"][
        "development_raw_predictions"
    ]["sha256"]:
        raise ValueError("v7.2 raw development scores differ from v7.1")
    if calibration["development_decision"] != (
        "PASS_RUN_FROZEN_ONCE"
        if selected["development_gate_passed"]
        else "RETAIN_V5_1_V7_2_FAILED_DEVELOPMENT_GATE"
    ):
        raise ValueError("v7.2 development decision disagrees with its checks")
    if not selected["development_gate_passed"] and any(
        path.exists() for path in FROZEN_PATHS
    ):
        raise ValueError("v7.2 frozen artifacts exist despite a failed development gate")
    if selected["development_gate_passed"]:
        raise NotImplementedError(
            "v7.2 passed development; complete its preregistered frozen sequence"
        )

    v7_1 = v7["v7_1_development_n8192"]
    v7_2 = {
        **selected["metrics"],
        "none_diagnostics": selected["none_diagnostics"],
    }
    report = {
        "schema_version": "dialam_v7_2_score_scale_result_v1",
        "experiment": "v7.2 transition-complete score-only NONE calibration",
        "selection_decision": calibration["development_decision"],
        "development_gate_passed": selected["development_gate_passed"],
        "development_gate_checks": selected["development_gate_checks"],
        "failed_development_conditions": sorted(
            name
            for name, passed in selected["development_gate_checks"].items()
            if not passed
        ),
        "v5_1_development_n8192": v7["v5_1_development_n8192"],
        "v7_1_development_n8192": v7_1,
        "v7_2_development_n8192": v7_2,
        "v5_1_frozen_n8192": v7["v5_1_frozen_n8192"],
        "calibration": {
            "selected_margin": selected["margin"],
            "candidate_margin_count": calibration["candidate_margin_count"],
            "minimum_margin": min(calibration["margins"]),
            "maximum_margin": max(calibration["margins"]),
            "candidate_margin_construction": calibration[
                "candidate_margin_construction"
            ],
            "new_training": calibration["new_training"],
            "new_model_calls": calibration["new_model_calls"],
        },
        "improvement_from_v7_1": {
            "exact_patch_accuracy_delta": (
                v7_2["exact_patch_accuracy"] - v7_1["exact_patch_accuracy"]
            ),
            "edge_f1_delta": v7_2["edge_f1"] - v7_1["edge_f1"],
            "relation_macro_f1_delta": (
                v7_2["relation_macro_f1"] - v7_1["relation_macro_f1"]
            ),
            "false_edges_per_update_delta": (
                v7_2["false_edges_per_update"] - v7_1["false_edges_per_update"]
            ),
            "none_false_case_delta": (
                v7_2["none_diagnostics"]["scenarios_with_false_edges"]
                - v7_1["none_diagnostics"]["scenarios_with_false_edges"]
            ),
        },
        "failure_diagnosis": {
            "false_positive_edges": v7_2["false_positive_edges"],
            "false_negative_edges": v7_2["false_negative_edges"],
            "support_false_positive_edges": v7_2["relation_metrics"]["SUPPORT"][
                "false_positive"
            ],
            "rephrase_recall": v7_2["relation_metrics"]["REPHRASE"]["recall"],
            "none_scenarios_with_false_edges": v7_2["none_diagnostics"][
                "scenarios_with_false_edges"
            ],
        },
        "frozen_status": calibration["frozen_status"],
        "frozen_candidate_calls": 0,
        "frozen_judge_calls": 0,
        "promotion_passed": False,
        "artifacts": {
            "preregistration": _artifact(PREREGISTRATION),
            "v7_result": _artifact(V7_RESULT),
            "development_raw_predictions": _artifact(DEV_RAW),
            "development_selected_predictions": _artifact(DEV_SELECTED),
            "calibration": _artifact(CALIBRATION),
        },
    }
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
                "improvement_from_v7_1": report["improvement_from_v7_1"],
                "frozen_candidate_calls": report["frozen_candidate_calls"],
                "frozen_judge_calls": report["frozen_judge_calls"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

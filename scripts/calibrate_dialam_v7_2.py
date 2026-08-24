#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from flowjudge.dialam_calibration_v5 import (
    none_diagnostics,
    select_none_margin,
    transition_complete_none_margins,
)
from flowjudge.patch_data import PROJECT_ROOT, PatchExample, load_jsonl
from flowjudge.patch_model_eval import deterministic_model_metrics, load_predictions


LOWER_BOUND = 3.0
DEFAULT_PREDICTIONS = (
    PROJECT_ROOT
    / "results"
    / "dialam_model_generation"
    / "v7_n8192_dev_raw"
    / "predictions.jsonl"
)
DEFAULT_EXAMPLES = (
    PROJECT_ROOT / "data" / "dialam" / "training" / "v3_dev_eval_examples.jsonl"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "dialam_model_generation"
    / "v7_2_n8192_dev"
    / "predictions.jsonl"
)
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "dialam_v7_2_calibration.json"
DATA_MANIFEST = (
    PROJECT_ROOT / "data" / "dialam" / "training" / "training_v7_manifest.json"
)
PREREGISTRATION = PROJECT_ROOT / "docs" / "dialam_v7_2_preregistration.md"
V5_1_RESULT = PROJECT_ROOT / "reports" / "dialam_v5_1_result.json"
V7_1_CALIBRATION = PROJECT_ROOT / "reports" / "dialam_v7_calibration.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the preregistered v7.2 transition-complete NONE-margin scan "
            "on development scores"
        )
    )
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--examples", type=Path, default=DEFAULT_EXAMPLES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    examples = load_jsonl(args.examples, PatchExample)
    predictions = load_predictions(args.predictions)
    data_manifest = json.loads(DATA_MANIFEST.read_text(encoding="utf-8"))
    margins = transition_complete_none_margins(
        predictions,
        lower_bound=LOWER_BOUND,
    )
    selection = select_none_margin(
        examples,
        predictions,
        thresholds=data_manifest["development_gate"],
        margins=margins,
        dataset_version="v7.2",
    )
    _write_jsonl(args.output, selection["selected_predictions"])

    raw_metrics = deterministic_model_metrics(examples, predictions)
    v5_1 = json.loads(V5_1_RESULT.read_text(encoding="utf-8"))[
        "v5_1_development_n8192"
    ]
    v7_1 = json.loads(V7_1_CALIBRATION.read_text(encoding="utf-8"))["selected"]
    passed = selection["selected"]["development_gate_passed"]
    report = {
        "schema_version": "dialam_v7_2_transition_complete_calibration_v1",
        "experiment": "v7.2 transition-complete development NONE-margin calibration",
        "post_v7_1_preregistered_before_extended_scan": True,
        "new_training": False,
        "new_model_calls": False,
        "lower_bound": LOWER_BOUND,
        "candidate_margin_count": len(margins),
        "candidate_margin_construction": (
            "3.0 plus every distinct best-positive-minus-NONE score gap above "
            "3.0; constructed without gold labels"
        ),
        "margins": list(margins),
        "selection_rule": (
            "all gates, edge F1, exact patch, macro-F1, lower false edges, "
            "lower NONE false-positive cases, larger margin"
        ),
        "v5_1_development_baseline": v5_1,
        "v7_1_selected": v7_1,
        "raw_v7": {
            "metrics": {
                key: value
                for key, value in raw_metrics.items()
                if key not in {"failure_cases", "scenario_failure_cases"}
            },
            "none_diagnostics": none_diagnostics(examples, predictions),
        },
        "selected": selection["selected"],
        "development_decision": (
            "PASS_RUN_FROZEN_ONCE"
            if passed
            else "RETAIN_V5_1_V7_2_FAILED_DEVELOPMENT_GATE"
        ),
        "frozen_status": (
            "READY_TO_RUN_DEVELOPMENT_GATE_PASSED"
            if passed
            else "NOT_RUN_PREREGISTERED_DEVELOPMENT_GATE_FAILED"
        ),
        "candidate_grid": selection["candidates"],
        "artifacts": {
            "input_predictions": _display_path(args.predictions),
            "input_predictions_sha256": _sha256(args.predictions),
            "output_predictions": _display_path(args.output),
            "output_predictions_sha256": _sha256(args.output),
            "examples_sha256": _sha256(args.examples),
            "training_manifest_sha256": _sha256(DATA_MANIFEST),
            "v7_1_calibration_sha256": _sha256(V7_1_CALIBRATION),
            "preregistration_sha256": _sha256(PREREGISTRATION),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["selected"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from flowjudge.dialam_calibration_v5 import (
    CALIBRATION_MARGINS,
    none_diagnostics,
    select_none_margin,
)
from flowjudge.patch_data import PROJECT_ROOT, PatchExample, load_jsonl
from flowjudge.patch_model_eval import deterministic_model_metrics, load_predictions


DEFAULT_PREDICTIONS = (
    PROJECT_ROOT
    / "results"
    / "dialam_model_generation"
    / "v6_n12288_dev_raw"
    / "predictions.jsonl"
)
DEFAULT_EXAMPLES = (
    PROJECT_ROOT / "data" / "dialam" / "training" / "v3_dev_eval_examples.jsonl"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "dialam_model_generation"
    / "v6_1_n12288_dev"
    / "predictions.jsonl"
)
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "dialam_v6_calibration.json"
DATA_MANIFEST = (
    PROJECT_ROOT / "data" / "dialam" / "training" / "training_v6_manifest.json"
)
PREREGISTRATION = PROJECT_ROOT / "docs" / "dialam_v6_preregistration.md"
V5_1_RESULT = PROJECT_ROOT / "reports" / "dialam_v5_1_result.json"


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
        description="Select the preregistered v6.1 NONE margin on development data"
    )
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--examples", type=Path, default=DEFAULT_EXAMPLES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    examples = load_jsonl(args.examples, PatchExample)
    predictions = load_predictions(args.predictions)
    data_manifest = json.loads(DATA_MANIFEST.read_text(encoding="utf-8"))
    selection = select_none_margin(
        examples,
        predictions,
        thresholds=data_manifest["development_gate"],
        margins=CALIBRATION_MARGINS,
        dataset_version="v6.1",
    )
    _write_jsonl(args.output, selection["selected_predictions"])
    raw_metrics = deterministic_model_metrics(examples, predictions)
    v5_1 = json.loads(V5_1_RESULT.read_text(encoding="utf-8"))[
        "v5_1_development_n8192"
    ]
    report = {
        "schema_version": "dialam_v6_1_none_margin_calibration_v1",
        "experiment": "v6.1 fixed-grid development NONE-margin calibration",
        "preregistered_before_v6_training": True,
        "margins": list(CALIBRATION_MARGINS),
        "selection_rule": (
            "all gates, edge F1, exact patch, macro-F1, lower false edges, "
            "lower NONE false-positive cases, larger margin"
        ),
        "v5_1_development_baseline": v5_1,
        "raw_v6": {
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
            if selection["selected"]["development_gate_passed"]
            else "RETAIN_V5_1_V6_FAILED_DEVELOPMENT_GATE"
        ),
        "frozen_status": (
            "READY_TO_RUN_DEVELOPMENT_GATE_PASSED"
            if selection["selected"]["development_gate_passed"]
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
            "preregistration_sha256": _sha256(PREREGISTRATION),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["selected"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

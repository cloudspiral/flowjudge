#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from flowjudge.dialam_calibration_v5 import (
    calibrate_prediction_rows,
    none_diagnostics,
)
from flowjudge.patch_data import PatchExample, load_jsonl
from flowjudge.patch_model_eval import deterministic_model_metrics, load_predictions


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply one already-frozen v5.1 NONE margin without tuning"
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--margin", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    examples = load_jsonl(args.examples, PatchExample)
    predictions = load_predictions(args.predictions)
    calibrated = calibrate_prediction_rows(
        examples,
        predictions,
        margin=args.margin,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in calibrated:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    metrics = deterministic_model_metrics(examples, calibrated)
    summary = {
        "schema_version": "dialam_v5_1_fixed_margin_application_v1",
        "margin": args.margin,
        "input_predictions": str(args.predictions),
        "input_predictions_sha256": _sha256(args.predictions),
        "examples": str(args.examples),
        "examples_sha256": _sha256(args.examples),
        "output_predictions": str(args.output),
        "output_predictions_sha256": _sha256(args.output),
        "metrics": {
            key: value
            for key, value in metrics.items()
            if key not in {"failure_cases", "scenario_failure_cases"}
        },
        "none_diagnostics": none_diagnostics(examples, calibrated),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

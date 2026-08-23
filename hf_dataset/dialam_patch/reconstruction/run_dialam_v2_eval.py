#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from flowjudge.patch_model_eval import APPROVAL_PHRASE, evaluate_prediction_file
from flowjudge.patch_data import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate and fully evaluate the preserved DialAM v2 n=2048 Modal checkpoint "
            "with the frozen deterministic metrics and blinded 30-call judge"
        )
    )
    parser.add_argument("--run-label", default="v2_n2048_reproduction")
    args = parser.parse_args()
    modal = shutil.which("modal")
    if modal is None:
        raise RuntimeError("the authenticated Modal CLI is required")

    generation_dir = PROJECT_ROOT / "results" / "dialam_model_generation" / args.run_label
    evaluation_dir = PROJECT_ROOT / "results" / "dialam_model_eval" / args.run_label
    if generation_dir.exists() or evaluation_dir.exists():
        raise FileExistsError(
            "refusing to overwrite an existing eval; choose a new --run-label"
        )
    predictions = generation_dir / "predictions.jsonl"
    subprocess.run(
        [
            modal,
            "run",
            str(PROJECT_ROOT / "scripts" / "modal_dialam_qlora.py"),
            "--action",
            "evaluate",
            "--target",
            "tuned",
            "--size",
            "2048",
            "--dataset-version",
            "v2",
            "--output-path",
            str(predictions),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )
    summary = evaluate_prediction_file(
        APPROVAL_PHRASE,
        predictions_path=predictions,
        output_dir=evaluation_dir,
    )
    print(summary)


if __name__ == "__main__":
    main()

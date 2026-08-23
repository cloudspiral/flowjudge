#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from flowjudge.patch_ceiling import DEFAULT_JUDGE_MODEL
from flowjudge.patch_model_eval import APPROVAL_PHRASE, evaluate_prediction_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score persisted Qwen DialAM predictions on the frozen eval"
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument(
        "--approval",
        required=True,
        help=f"exact approval phrase: {APPROVAL_PHRASE}",
    )
    args = parser.parse_args()
    print(
        evaluate_prediction_file(
            args.approval,
            predictions_path=args.predictions,
            output_dir=args.output_dir,
            judge_model=args.judge_model,
        )
    )


if __name__ == "__main__":
    main()

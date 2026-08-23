from __future__ import annotations

import argparse
import json
from pathlib import Path

from flowjudge.patch_data import DEFAULT_EVAL_EXAMPLES_PATH
from flowjudge.patch_metrics import score_patch_prediction_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Score incremental argument-patch predictions")
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--examples", type=Path, default=DEFAULT_EVAL_EXAMPLES_PATH)
    args = parser.parse_args()
    print(json.dumps(score_patch_prediction_file(args.predictions, args.examples), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

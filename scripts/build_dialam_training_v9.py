#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from flowjudge.dialam_training import DEFAULT_TRAINING_DIR
from flowjudge.dialam_training_v9 import (
    V9_MINING_SCORE_FILENAME,
    build_dialam_v9_training_corpus,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the private V9 corrective corpus from complete mining scores"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_TRAINING_DIR)
    parser.add_argument("--scores", type=Path)
    args = parser.parse_args()
    scores = args.scores or args.output_dir / V9_MINING_SCORE_FILENAME
    manifest = build_dialam_v9_training_corpus(
        scores_path=scores,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

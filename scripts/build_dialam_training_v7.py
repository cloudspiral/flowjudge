#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from flowjudge.dialam_training import DEFAULT_TRAINING_DIR
from flowjudge.dialam_training_v7 import build_dialam_v7_preference_corpus


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the private QT30-only v7 reciprocal candidate-preference corpus "
            "and nested interruption/resume smoke subset"
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_TRAINING_DIR)
    parser.add_argument("--source-v5", type=Path)
    args = parser.parse_args()
    manifest = build_dialam_v7_preference_corpus(
        source_v5_path=args.source_v5,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

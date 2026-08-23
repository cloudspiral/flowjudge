#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from flowjudge.dialam_training import DEFAULT_TRAINING_DIR
from flowjudge.dialam_training_v6 import (
    DEFAULT_VIVES_SOURCE_DIR,
    build_dialam_v6_training_corpus,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the private DialAM v6 curriculum: a deterministic "
            "VivesDebate pairwise warm-up followed by the exact v5 QT30 rows"
        )
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_VIVES_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_TRAINING_DIR)
    parser.add_argument("--v5-path", type=Path)
    args = parser.parse_args()
    manifest = build_dialam_v6_training_corpus(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        v5_path=args.v5_path,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

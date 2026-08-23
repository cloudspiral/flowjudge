#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from flowjudge.dialam_training import DEFAULT_TRAINING_DIR
from flowjudge.dialam_training_v5 import build_dialam_v5_training_corpus


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the private DialAM v5 n=8192 pairwise-classification corpus "
            "with exact within-block positive/NONE contrasts"
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_TRAINING_DIR)
    args = parser.parse_args()
    manifest = build_dialam_v5_training_corpus(output_dir=args.output_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

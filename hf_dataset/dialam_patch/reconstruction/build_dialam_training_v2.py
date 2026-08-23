#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from flowjudge.dialam import DEFAULT_DIALOGUES_PATH, DEFAULT_SOURCE_DIR
from flowjudge.dialam_training import DEFAULT_TRAINING_DIR
from flowjudge.dialam_training_v2 import build_dialam_v2_training_corpus
from flowjudge.patch_data import DEFAULT_EVAL_SUMMARY_PATH


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the private DialAM v2 n=2048 hard-negative QLoRA corpus"
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--dialogues", type=Path, default=DEFAULT_DIALOGUES_PATH)
    parser.add_argument("--eval-summary", type=Path, default=DEFAULT_EVAL_SUMMARY_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_TRAINING_DIR)
    args = parser.parse_args()
    manifest = build_dialam_v2_training_corpus(
        source_dir=args.source_dir,
        dialogues_path=args.dialogues,
        eval_summary_path=args.eval_summary,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

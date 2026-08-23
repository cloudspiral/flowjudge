from __future__ import annotations

import argparse
import json
from pathlib import Path

from flowjudge.dialam import DEFAULT_DIALOGUES_PATH, DEFAULT_SOURCE_DIR, load_canonical_maps
from flowjudge.patch_data import (
    DEFAULT_DIAGNOSTIC_EVAL_SCENARIOS_PATH,
    DEFAULT_EVAL_SCENARIOS_PATH,
    DEFAULT_TRAIN_PATH,
    PatchExample,
    PatchScenario,
    load_jsonl,
    validate_patch_datasets,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the generated DialAM/QT30 feasibility-gate data")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--dialogues", type=Path, default=DEFAULT_DIALOGUES_PATH)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--eval-scenarios", type=Path, default=DEFAULT_EVAL_SCENARIOS_PATH)
    parser.add_argument(
        "--diagnostic-eval-scenarios",
        type=Path,
        default=DEFAULT_DIAGNOSTIC_EVAL_SCENARIOS_PATH,
    )
    args = parser.parse_args()

    maps = load_canonical_maps(args.source_dir, args.dialogues)
    train_examples = load_jsonl(args.train, PatchExample)
    eval_scenarios = load_jsonl(args.eval_scenarios, PatchScenario)
    diagnostic_eval_scenarios = load_jsonl(
        args.diagnostic_eval_scenarios,
        PatchScenario,
    )
    validation = validate_patch_datasets(train_examples, eval_scenarios, maps)
    diagnostic_validation = validate_patch_datasets(
        train_examples,
        diagnostic_eval_scenarios,
        maps,
    )
    print(
        json.dumps(
            {
                "valid": True,
                "train_examples": len(train_examples),
                "eval_scenarios": len(eval_scenarios),
                "diagnostic_eval_scenarios": len(diagnostic_eval_scenarios),
                "validation": validation,
                "diagnostic_validation": diagnostic_validation,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

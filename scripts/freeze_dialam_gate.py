#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from flowjudge.patch_ceiling import FROZEN_JUDGE_RUBRIC_SHA256
from flowjudge.patch_data import (
    DEFAULT_DIAGNOSTIC_EVAL_EXAMPLES_PATH,
    DEFAULT_DIAGNOSTIC_EVAL_SCENARIOS_PATH,
    PROJECT_ROOT,
)


DEFAULT_RUN = (
    PROJECT_ROOT
    / "results"
    / "dialam_prompt_ceiling"
    / "20260823T043624.705184Z"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "dialam" / "frozen_gate"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _entry(path: Path) -> dict[str, object]:
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def freeze_gate(run_directory: Path, output_directory: Path) -> Path:
    required = [
        DEFAULT_DIAGNOSTIC_EVAL_EXAMPLES_PATH,
        DEFAULT_DIAGNOSTIC_EVAL_SCENARIOS_PATH,
        PROJECT_ROOT / "BEHAVIOR_SPEC.md",
        PROJECT_ROOT / "prompts" / "dialam" / "zero_shot.txt",
        PROJECT_ROOT / "prompts" / "dialam" / "few_shot.txt",
        PROJECT_ROOT / "prompts" / "dialam" / "strong_structured.txt",
        PROJECT_ROOT / "prompts" / "dialam" / "judge.txt",
        run_directory / "manifest.json",
        run_directory / "metrics.json",
        run_directory / "records.jsonl",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"cannot freeze missing gate artifacts: {missing}")
    if sha256(PROJECT_ROOT / "prompts" / "dialam" / "judge.txt") != FROZEN_JUDGE_RUBRIC_SHA256:
        raise RuntimeError("judge rubric no longer matches its frozen SHA-256")

    raw_files = sorted(path for path in (run_directory / "raw").rglob("*") if path.is_file())
    if len(raw_files) != 180 * 6:
        raise ValueError(f"expected 1080 raw prompt-ceiling artifacts, found {len(raw_files)}")

    output_directory.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(run_directory / "manifest.json", output_directory / "prompt_ceiling_run_manifest.json")
    shutil.copyfile(run_directory / "metrics.json", output_directory / "prompt_ceiling_metrics.json")

    eval_examples = sum(
        1
        for line in DEFAULT_DIAGNOSTIC_EVAL_EXAMPLES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    eval_scenarios = sum(
        1
        for line in DEFAULT_DIAGNOSTIC_EVAL_SCENARIOS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    manifest = {
        "schema_version": "dialam_gate_freeze_v1",
        "frozen_at": datetime.now(UTC).isoformat(),
        "immutable": True,
        "eval_scenario_count": eval_scenarios,
        "eval_block_count": eval_examples,
        "judge_rubric_sha256": FROZEN_JUDGE_RUBRIC_SHA256,
        "prompt_ceiling_run": str(run_directory.relative_to(PROJECT_ROOT)),
        "formal_call_counts": {"candidate": 180, "judge": 180},
        "core_artifacts": [_entry(path) for path in required],
        "raw_artifact_count": len(raw_files),
        "raw_artifacts": [_entry(path) for path in raw_files],
        "privacy": (
            "Raw/text-bearing QT30 eval and transcript artifacts remain local and ignored; "
            "this text-free manifest freezes them by path, byte count, and SHA-256."
        ),
    }
    output = output_directory / "freeze_manifest.json"
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the completed DialAM prompt-ceiling gate")
    parser.add_argument("--run-directory", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(freeze_gate(args.run_directory.resolve(), args.output_directory.resolve()))


if __name__ == "__main__":
    main()

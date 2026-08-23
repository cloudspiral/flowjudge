#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "hf_dataset" / "dialam_patch"

WHITELIST = {
    "data/dialam/source_manifest.json": "metadata/source_manifest.json",
    "data/dialam/smoke_manifest.json": "metadata/corpus_audit_manifest.json",
    "data/dialam/eval_summary.json": "metadata/eval_summary.json",
    "data/dialam/frozen_gate/freeze_manifest.json": "metadata/frozen_gate_manifest.json",
    "data/dialam/frozen_gate/prompt_ceiling_run_manifest.json": "metadata/prompt_ceiling_run_manifest.json",
    "data/dialam/frozen_gate/prompt_ceiling_metrics.json": "metadata/prompt_ceiling_metrics.json",
    "data/dialam/training/training_manifest.json": "metadata/training_manifest.json",
    "data/dialam/training/training_example.schema.json": "schema/training_example.schema.json",
    "scripts/build_dialam_gate.py": "reconstruction/build_dialam_gate.py",
    "scripts/build_dialam_training.py": "reconstruction/build_dialam_training.py",
    "scripts/freeze_dialam_gate.py": "reconstruction/freeze_dialam_gate.py",
    "src/flowjudge/dialam.py": "reconstruction/flowjudge/dialam.py",
    "src/flowjudge/patch_data.py": "reconstruction/flowjudge/patch_data.py",
    "src/flowjudge/dialam_training.py": "reconstruction/flowjudge/dialam_training.py",
    "src/flowjudge/patch_ceiling.py": "reconstruction/flowjudge/patch_ceiling.py",
    "src/flowjudge/patch_metrics.py": "reconstruction/flowjudge/patch_metrics.py",
    "src/flowjudge/schemas.py": "reconstruction/flowjudge/schemas.py",
    "prompts/dialam/zero_shot.txt": "reconstruction/prompts/dialam/zero_shot.txt",
    "prompts/dialam/few_shot.txt": "reconstruction/prompts/dialam/few_shot.txt",
    "prompts/dialam/strong_structured.txt": "reconstruction/prompts/dialam/strong_structured.txt",
    "prompts/dialam/judge.txt": "reconstruction/prompts/dialam/judge.txt",
    "BEHAVIOR_SPEC.md": "reconstruction/BEHAVIOR_SPEC.md",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    files = []
    for source_name, destination_name in WHITELIST.items():
        source = PROJECT_ROOT / source_name
        destination = OUTPUT / destination_name
        if not source.is_file():
            raise FileNotFoundError(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        files.append(
            {
                "path": destination_name,
                "sha256": sha256(destination),
                "bytes": destination.stat().st_size,
                "source": source_name,
            }
        )
    manifest = {
        "schema_version": "dialam_hf_dataset_structure_v1",
        "prepared_at": datetime.now(UTC).isoformat(),
        "contains_raw_or_transformed_qt30_text": False,
        "publishable_scope": "manifests, metadata, schemas, aggregate statistics, and reconstruction code only",
        "files": files,
    }
    (OUTPUT / "publish_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()

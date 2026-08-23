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
    "data/dialam/training/training_v2_manifest.json": "metadata/training_v2_manifest.json",
    "data/dialam/training/training_v3_manifest.json": "metadata/training_v3_manifest.json",
    "data/dialam/training/training_example.schema.json": "schema/training_example.schema.json",
    "data/dialam/training/training_example_v2.schema.json": "schema/training_example_v2.schema.json",
    "data/dialam/training/training_example_v3.schema.json": "schema/training_example_v3.schema.json",
    "reports/dialam_v1_efficiency_curve.json": "metadata/dialam_v1_efficiency_curve.json",
    "reports/dialam_v1_to_v2.json": "metadata/dialam_v1_to_v2.json",
    "docs/dialam_v3_preregistration.md": "metadata/dialam_v3_preregistration.md",
    "scripts/build_dialam_gate.py": "reconstruction/build_dialam_gate.py",
    "scripts/build_dialam_training.py": "reconstruction/build_dialam_training.py",
    "scripts/build_dialam_training_v2.py": "reconstruction/build_dialam_training_v2.py",
    "scripts/build_dialam_training_v3.py": "reconstruction/build_dialam_training_v3.py",
    "scripts/build_dialam_v1_curve_report.py": "reconstruction/build_dialam_v1_curve_report.py",
    "scripts/build_dialam_v2_report.py": "reconstruction/build_dialam_v2_report.py",
    "scripts/evaluate_dialam_model.py": "reconstruction/evaluate_dialam_model.py",
    "scripts/freeze_dialam_gate.py": "reconstruction/freeze_dialam_gate.py",
    "scripts/modal_dialam_qlora.py": "reconstruction/modal_dialam_qlora.py",
    "scripts/run_dialam_v2_eval.py": "reconstruction/run_dialam_v2_eval.py",
    "src/flowjudge/dialam.py": "reconstruction/flowjudge/dialam.py",
    "src/flowjudge/patch_data.py": "reconstruction/flowjudge/patch_data.py",
    "src/flowjudge/dialam_training.py": "reconstruction/flowjudge/dialam_training.py",
    "src/flowjudge/dialam_training_v2.py": "reconstruction/flowjudge/dialam_training_v2.py",
    "src/flowjudge/dialam_training_v3.py": "reconstruction/flowjudge/dialam_training_v3.py",
    "src/flowjudge/patch_ceiling.py": "reconstruction/flowjudge/patch_ceiling.py",
    "src/flowjudge/patch_metrics.py": "reconstruction/flowjudge/patch_metrics.py",
    "src/flowjudge/patch_model_eval.py": "reconstruction/flowjudge/patch_model_eval.py",
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
    readme = OUTPUT / "README.md"
    if not readme.is_file():
        raise FileNotFoundError(readme)
    files.append(
        {
            "path": "README.md",
            "sha256": sha256(readme),
            "bytes": readme.stat().st_size,
            "source": "hf_dataset/dialam_patch/README.md",
        }
    )
    manifest = {
        "schema_version": "dialam_hf_dataset_structure_v3",
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

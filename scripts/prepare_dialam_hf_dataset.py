#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "hf_dataset" / "dialam_patch"

WHITELIST = {
    "data/dialam/training/training_v2_manifest.json": "metadata/training_v2_manifest.json",
    "data/dialam/training/training_v3_manifest.json": "metadata/training_v3_manifest.json",
    "data/dialam/training/training_v4_manifest.json": "metadata/training_v4_manifest.json",
    "data/dialam/training/training_v5_manifest.json": "metadata/training_v5_manifest.json",
    "data/dialam/training/training_example.schema.json": "schema/training_example.schema.json",
    "data/dialam/training/training_example_v2.schema.json": "schema/training_example_v2.schema.json",
    "data/dialam/training/training_example_v3.schema.json": "schema/training_example_v3.schema.json",
    "data/dialam/training/training_example_v4.schema.json": "schema/training_example_v4.schema.json",
    "data/dialam/training/training_example_v5.schema.json": "schema/training_example_v5.schema.json",
    "reports/dialam_v1_efficiency_curve.json": "metadata/dialam_v1_efficiency_curve.json",
    "reports/dialam_v1_to_v2.json": "metadata/dialam_v1_to_v2.json",
    "reports/dialam_v1_v2_v3.json": "metadata/dialam_v1_v2_v3.json",
    "reports/dialam_v4_class_balanced_rehearsal.json": "metadata/dialam_v4_class_balanced_rehearsal.json",
    "reports/dialam_v5_pairwise_classification.json": "metadata/dialam_v5_pairwise_classification.json",
    "reports/dialam_v5_1_result.json": "metadata/dialam_v5_1_result.json",
    "reports/dialam_experiment_history.json": "metadata/dialam_experiment_history.json",
    "reports/dialam_experiment_history.svg": "metadata/dialam_experiment_history.svg",
    "artifacts/dialam_qlora/n256/training_manifest.json": "evidence/training_v1_n256.json",
    "artifacts/dialam_qlora/n512/training_manifest.json": "evidence/training_v1_n512.json",
    "artifacts/dialam_qlora/n1024/training_manifest.json": "evidence/training_v1_n1024.json",
    "artifacts/dialam_qlora/n2048/training_manifest.json": "evidence/training_v1_n2048.json",
    "artifacts/dialam_qlora/v2_n2048/training_manifest.json": "evidence/training_v2_n2048.json",
    "artifacts/dialam_qlora/v3_n4096/training_manifest.json": "evidence/training_v3_n4096.json",
    "artifacts/dialam_qlora/v4_n8192/training_manifest.json": "evidence/training_v4_n8192.json",
    "artifacts/dialam_qlora/v5_n8192/training_manifest.json": "evidence/training_v5_n8192.json",
    "docs/dialam_v3_preregistration.md": "metadata/dialam_v3_preregistration.md",
    "docs/dialam_v3_results.md": "metadata/dialam_v3_results.md",
    "docs/dialam_v4_preregistration.md": "metadata/dialam_v4_preregistration.md",
    "docs/dialam_v4_results.md": "metadata/dialam_v4_results.md",
    "docs/dialam_v5_preregistration.md": "metadata/dialam_v5_preregistration.md",
    "docs/dialam_v5_1_calibration_preregistration.md": "metadata/dialam_v5_1_calibration_preregistration.md",
    "docs/dialam_v5_results.md": "metadata/dialam_v5_results.md",
    "docs/dialam_v5_1_results.md": "metadata/dialam_v5_1_results.md",
    "docs/dialam_experiment_history.md": "metadata/dialam_experiment_history.md",
    "DATA_AUDIT.md": "metadata/DATA_AUDIT.md",
    "scripts/build_dialam_gate.py": "reconstruction/build_dialam_gate.py",
    "scripts/build_dialam_training.py": "reconstruction/build_dialam_training.py",
    "scripts/build_dialam_training_v2.py": "reconstruction/build_dialam_training_v2.py",
    "scripts/build_dialam_training_v3.py": "reconstruction/build_dialam_training_v3.py",
    "scripts/build_dialam_training_v4.py": "reconstruction/build_dialam_training_v4.py",
    "scripts/build_dialam_training_v5.py": "reconstruction/build_dialam_training_v5.py",
    "scripts/build_dialam_v1_curve_report.py": "reconstruction/build_dialam_v1_curve_report.py",
    "scripts/build_dialam_v2_report.py": "reconstruction/build_dialam_v2_report.py",
    "scripts/build_dialam_v3_report.py": "reconstruction/build_dialam_v3_report.py",
    "scripts/build_dialam_v4_report.py": "reconstruction/build_dialam_v4_report.py",
    "scripts/build_dialam_v5_report.py": "reconstruction/build_dialam_v5_report.py",
    "scripts/build_dialam_v5_1_report.py": "reconstruction/build_dialam_v5_1_report.py",
    "scripts/calibrate_dialam_v5.py": "reconstruction/calibrate_dialam_v5.py",
    "scripts/apply_dialam_v5_margin.py": "reconstruction/apply_dialam_v5_margin.py",
    "scripts/build_dialam_experiment_history.py": "reconstruction/build_dialam_experiment_history.py",
    "scripts/evaluate_dialam_model.py": "reconstruction/evaluate_dialam_model.py",
    "scripts/freeze_dialam_gate.py": "reconstruction/freeze_dialam_gate.py",
    "scripts/modal_dialam_qlora.py": "reconstruction/modal_dialam_qlora.py",
    "scripts/prepare_dialam_hf_model_v5_1.py": "reconstruction/prepare_dialam_hf_model_v5_1.py",
    "scripts/run_dialam_v2_eval.py": "reconstruction/run_dialam_v2_eval.py",
    "src/flowjudge/dialam.py": "reconstruction/flowjudge/dialam.py",
    "src/flowjudge/patch_data.py": "reconstruction/flowjudge/patch_data.py",
    "src/flowjudge/dialam_training.py": "reconstruction/flowjudge/dialam_training.py",
    "src/flowjudge/dialam_training_v2.py": "reconstruction/flowjudge/dialam_training_v2.py",
    "src/flowjudge/dialam_training_v3.py": "reconstruction/flowjudge/dialam_training_v3.py",
    "src/flowjudge/dialam_training_v4.py": "reconstruction/flowjudge/dialam_training_v4.py",
    "src/flowjudge/dialam_training_v5.py": "reconstruction/flowjudge/dialam_training_v5.py",
    "src/flowjudge/dialam_calibration_v5.py": "reconstruction/flowjudge/dialam_calibration_v5.py",
    "src/flowjudge/dialam_hf_eval.py": "reconstruction/flowjudge/dialam_hf_eval.py",
    "src/flowjudge/experiment_history.py": "reconstruction/flowjudge/experiment_history.py",
    "src/flowjudge/hf_eval.py": "reconstruction/flowjudge/hf_eval.py",
    "src/flowjudge/patch_ceiling.py": "reconstruction/flowjudge/patch_ceiling.py",
    "src/flowjudge/patch_metrics.py": "reconstruction/flowjudge/patch_metrics.py",
    "src/flowjudge/patch_model_eval.py": "reconstruction/flowjudge/patch_model_eval.py",
    "src/flowjudge/schemas.py": "reconstruction/flowjudge/schemas.py",
    "prompts/dialam/zero_shot.txt": "reconstruction/prompts/dialam/zero_shot.txt",
    "prompts/dialam/few_shot.txt": "reconstruction/prompts/dialam/few_shot.txt",
    "prompts/dialam/strong_structured.txt": "reconstruction/prompts/dialam/strong_structured.txt",
    "prompts/dialam/judge.txt": "reconstruction/prompts/dialam/judge.txt",
    "prompts/dialam/pairwise_v5.txt": "reconstruction/prompts/dialam/pairwise_v5.txt",
    "BEHAVIOR_SPEC.md": "reconstruction/BEHAVIOR_SPEC.md",
    "eval.py": "reconstruction/eval.py",
}

TRANSFORMED_DATA = {
    "data/dialam/training/dialam_n256.jsonl": "data/train_v1_n256.jsonl",
    "data/dialam/training/dialam_n512.jsonl": "data/train_v1_n512.jsonl",
    "data/dialam/training/dialam_n1024.jsonl": "data/train_v1_n1024.jsonl",
    "data/dialam/training/dialam_n2048.jsonl": "data/train_v1_n2048.jsonl",
    "data/dialam/training/dialam_v2_n2048.jsonl": "data/train_v2_n2048.jsonl",
    "data/dialam/training/dialam_v3_n4096.jsonl": "data/train_v3_n4096.jsonl",
    "data/dialam/training/dialam_v5_n8192.jsonl": "data/train_v5_n8192.jsonl",
    "data/dialam/training/v3_dev_eval_examples.jsonl": "data/development_eval_30.jsonl",
    "data/dialam/balanced_diagnostic_eval_examples.jsonl": "data/frozen_own_eval_30.jsonl",
    "results/dialam_prompt_ceiling/20260823T043624.705184Z/records.jsonl": "evidence/prompt_ceiling_candidate_and_judge_records.jsonl",
    "results/dialam_model_generation/v5_n8192_frozen_raw_scores/predictions.jsonl": "evidence/v5_n8192_frozen_raw_scores.jsonl",
    "results/dialam_model_generation/v5_1_n8192/predictions.jsonl": "evidence/v5_1_n8192_frozen_candidate_transcripts.jsonl",
    "results/dialam_model_eval/v5_1_n8192/judge_transcripts.jsonl": "evidence/v5_1_n8192_frozen_judge_transcripts.jsonl",
    "results/dialam_model_eval/v5_1_n8192/records.jsonl": "evidence/v5_1_n8192_frozen_records.jsonl",
}

for _run_name in ("base", "n256", "n512", "n1024", "n2048", "v2_n2048", "v3_n4096"):
    TRANSFORMED_DATA.update(
        {
            f"results/dialam_model_eval/{_run_name}/predictions.jsonl": (
                f"evidence/{_run_name}_candidate_transcripts.jsonl"
            ),
            f"results/dialam_model_eval/{_run_name}/judge_transcripts.jsonl": (
                f"evidence/{_run_name}_judge_transcripts.jsonl"
            ),
        }
    )

OPTIONAL_EVIDENCE = {
    "results/dialam_model_generation/v5_n8192_dev/predictions.jsonl": (
        "evidence/v5_n8192_development_predictions.jsonl"
    ),
    "results/dialam_model_generation/v5_n8192/predictions.jsonl": (
        "evidence/v5_n8192_frozen_candidate_transcripts.jsonl"
    ),
    "results/dialam_model_eval/v5_n8192/judge_transcripts.jsonl": (
        "evidence/v5_n8192_frozen_judge_transcripts.jsonl"
    ),
    "results/dialam_model_generation/v5_1_n8192_dev/predictions.jsonl": (
        "evidence/v5_1_n8192_development_predictions.jsonl"
    ),
}

OPTIONAL_FILES = {
    "reports/dialam_v5_1_calibration.json": "metadata/dialam_v5_1_calibration.json",
    "reports/dialam_v5_1_frozen_application.json": "metadata/dialam_v5_1_frozen_application.json",
    "results/dialam_model_eval/v5_1_n8192/summary.json": "evidence/v5_1_n8192_frozen_summary.json",
}

SANITIZED_JSON: dict[str, tuple[str, Callable[[dict[str, Any]], dict[str, Any]]]] = {}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_code_commit() -> str:
    value = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        text=True,
    ).strip()
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("git HEAD is not a full lowercase SHA-1")
    return value


def _redaction(source: Path, removed: list[str]) -> dict[str, Any]:
    return {
        "source_sha256": sha256(source),
        "policy": (
            "Per-example inventories are omitted from compact aggregate metadata; "
            "the permitted transformed JSONL files preserve the complete examples."
        ),
        "removed_fields": removed,
    }


def _sanitize_eval_summary(value: dict[str, Any]) -> dict[str, Any]:
    public = deepcopy(value)
    heldout = public.pop("heldout_dialogue_ids", [])
    public["heldout_parent_episode_count"] = len(heldout)
    for name in ("balanced_diagnostic", "natural_distribution"):
        section = public.get(name, {})
        episodes = section.pop("dialogue_ids", [])
        section["parent_episode_count"] = len(episodes)
    return public


def _sanitize_source_manifest(value: dict[str, Any]) -> dict[str, Any]:
    public = deepcopy(value)
    metadata = public.get("dialogue_metadata", {})
    if "archive_metadata_discrepancy" in metadata:
        metadata["archive_metadata_discrepancy"] = (
            "One archive-versus-current-API episode mismatch is documented in "
            "the private dialogue manifest; its original identifiers are omitted."
        )
    return public


def _sanitize_smoke_manifest(value: dict[str, Any]) -> dict[str, Any]:
    public = deepcopy(value)
    maps_by_episode = public.get("corpus_audit", {}).pop("maps_by_dialogue", {})
    map_counts = list(maps_by_episode.values())
    if map_counts:
        public["corpus_audit"]["parent_episode_count"] = len(map_counts)
        public["corpus_audit"]["maps_per_parent_episode"] = {
            "minimum": min(map_counts),
            "maximum": max(map_counts),
            "mean": sum(map_counts) / len(map_counts),
        }
    for name in ("validation", "diagnostic_validation"):
        validation = public.get(name, {})
        train = validation.pop("train_dialogue_ids", [])
        heldout = validation.pop("eval_dialogue_ids", [])
        validation["training_parent_episode_count"] = len(train)
        validation["heldout_parent_episode_count"] = len(heldout)
    public["eval_summary"] = _sanitize_eval_summary(public["eval_summary"])
    return public


def _sanitize_training_manifest(value: dict[str, Any]) -> dict[str, Any]:
    public = deepcopy(value)
    train = public.pop("training_parent_episode_ids", [])
    heldout = public.pop("heldout_parent_episode_ids", [])
    public["training_parent_episode_count"] = len(train)
    public["heldout_parent_episode_count"] = len(heldout)
    return public


def _sanitize_frozen_gate(value: dict[str, Any]) -> dict[str, Any]:
    public = deepcopy(value)
    raw = public.pop("raw_artifacts", [])
    if public.get("raw_artifact_count") != len(raw):
        raise ValueError("frozen-gate raw artifact count does not match its inventory")
    public["raw_artifact_inventory_omitted"] = True
    return public


def _sanitize_prompt_run(value: dict[str, Any]) -> dict[str, Any]:
    public = deepcopy(value)
    examples = public.pop("example_ids", [])
    demonstrations = public.pop("few_shot_example_ids", [])
    public["example_id_count"] = len(examples)
    public["few_shot_example_id_count"] = len(demonstrations)
    return public


def _sanitize_prompt_metrics(value: dict[str, Any]) -> dict[str, Any]:
    public = deepcopy(value)
    for metrics in public.get("by_model_prompt", {}).values():
        failures = metrics.pop("failure_cases", [])
        scenario_failures = metrics.pop("scenario_failure_cases", [])
        metrics["failure_case_count"] = len(failures)
        metrics["scenario_failure_case_count"] = len(scenario_failures)
    return public


SANITIZED_JSON.update(
    {
        "data/dialam/source_manifest.json": (
            "metadata/source_manifest.json",
            _sanitize_source_manifest,
        ),
        "data/dialam/smoke_manifest.json": (
            "metadata/corpus_audit_manifest.json",
            _sanitize_smoke_manifest,
        ),
        "data/dialam/eval_summary.json": (
            "metadata/eval_summary.json",
            _sanitize_eval_summary,
        ),
        "data/dialam/frozen_gate/freeze_manifest.json": (
            "metadata/frozen_gate_manifest.json",
            _sanitize_frozen_gate,
        ),
        "data/dialam/frozen_gate/prompt_ceiling_run_manifest.json": (
            "metadata/prompt_ceiling_run_manifest.json",
            _sanitize_prompt_run,
        ),
        "data/dialam/frozen_gate/prompt_ceiling_metrics.json": (
            "metadata/prompt_ceiling_metrics.json",
            _sanitize_prompt_metrics,
        ),
        "data/dialam/training/training_manifest.json": (
            "metadata/training_manifest.json",
            _sanitize_training_manifest,
        ),
    }
)


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
    for source_name, destination_name in OPTIONAL_FILES.items():
        source = PROJECT_ROOT / source_name
        if not source.is_file():
            continue
        destination = OUTPUT / destination_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        files.append(
            {
                "path": destination_name,
                "sha256": sha256(destination),
                "bytes": destination.stat().st_size,
                "source": source_name,
                "optional": True,
            }
        )
    transformed_files = dict(TRANSFORMED_DATA)
    transformed_files.update(
        {
            source_name: destination_name
            for source_name, destination_name in OPTIONAL_EVIDENCE.items()
            if (PROJECT_ROOT / source_name).is_file()
        }
    )
    for source_name, destination_name in transformed_files.items():
        source = PROJECT_ROOT / source_name
        destination = OUTPUT / destination_name
        if not source.is_file():
            raise FileNotFoundError(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        with destination.open(encoding="utf-8") as stream:
            row_count = sum(1 for line in stream if line.strip())
        files.append(
            {
                "path": destination_name,
                "sha256": sha256(destination),
                "bytes": destination.stat().st_size,
                "rows": row_count,
                "source": source_name,
                "transformed_qt30": True,
            }
        )
    for source_name, (destination_name, sanitizer) in SANITIZED_JSON.items():
        source = PROJECT_ROOT / source_name
        destination = OUTPUT / destination_name
        if not source.is_file():
            raise FileNotFoundError(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        value = json.loads(source.read_text(encoding="utf-8"))
        public = sanitizer(value)
        public["publication_redaction"] = _redaction(
            source,
            [
                "identifier-bearing fields and per-example inventories where present",
            ],
        )
        destination.write_text(
            json.dumps(public, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        files.append(
            {
                "path": destination_name,
                "sha256": sha256(destination),
                "bytes": destination.stat().st_size,
                "source": source_name,
                "sanitized": True,
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
        "schema_version": "dialam_hf_dataset_structure_v4",
        "prepared_at": datetime.now(UTC).isoformat(),
        "contains_raw_qt30_archive_or_maps": False,
        "contains_transformed_qt30_text": True,
        "contains_original_qt30_identifiers": True,
        "permission_basis": (
            "Project-specific use and redistribution permission directly attested "
            "by the project owner on 2026-08-23; no general QT30 license is claimed."
        ),
        "source_code_commit": source_code_commit(),
        "publishable_scope": (
            "curated transformed training/evaluation JSONL, manifests, metadata, "
            "schemas, aggregate statistics, and deterministic reconstruction code"
        ),
        "files": files,
    }
    (OUTPUT / "publish_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()

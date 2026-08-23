#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from flowjudge.patch_data import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish the vetted DialAM model and permission-cleared transformed dataset"
    )
    parser.add_argument("--model-repo", required=True)
    parser.add_argument("--dataset-repo", required=True)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "hf_publish"
            / "dialam-qwen3-0.6b-v5-1-n8192"
        ),
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=PROJECT_ROOT / "hf_dataset" / "dialam_patch",
    )
    parser.add_argument("--private", action="store_true")
    parser.add_argument(
        "--skip-model-upload",
        action="store_true",
        help="Reuse the existing model revision while updating only the dataset artifact",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=PROJECT_ROOT / "docs" / "dialam_hf_publication_manifest.json",
    )
    args = parser.parse_args()

    model_manifest = _validate_manifest(args.model_dir)
    dataset_manifest = _validate_manifest(args.dataset_dir)
    if not model_manifest.get("selected_submission_checkpoint"):
        raise ValueError("model package is not marked as the selected submission checkpoint")
    if model_manifest.get("contains_raw_or_transformed_qt30_text") is not False:
        raise ValueError("model package redistribution boundary is not safe")
    if dataset_manifest.get("contains_raw_qt30_archive_or_maps") is not False:
        raise ValueError("dataset package must not mirror the official raw archive or maps")
    if dataset_manifest.get("contains_transformed_qt30_text") is not True:
        raise ValueError("dataset package is missing the publishable transformed dataset")
    if not dataset_manifest.get("permission_basis"):
        raise ValueError("dataset package is missing its redistribution permission basis")

    load_dotenv(PROJECT_ROOT / ".env", override=False)
    token = os.getenv("HF_TOKEN", "").strip() or None
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for publication") from exc
    api = HfApi(token=token)
    if args.skip_model_upload:
        model_commit = api.model_info(args.model_repo).sha
    else:
        api.create_repo(
            args.model_repo,
            repo_type="model",
            private=args.private,
            exist_ok=True,
        )
        model_commit = api.upload_folder(
            repo_id=args.model_repo,
            repo_type="model",
            folder_path=args.model_dir,
            commit_message=(
                f"Publish selected {model_manifest['model']} with frozen evaluation "
                "and calibrated inference evidence"
            ),
        ).oid
    api.create_repo(
        args.dataset_repo,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )
    dataset_commit = api.upload_folder(
        repo_id=args.dataset_repo,
        repo_type="dataset",
        folder_path=args.dataset_dir,
        commit_message="Publish permission-cleared DialAM training/evaluation data and reproducibility artifacts",
    )
    output = args.output_manifest
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "published_at": datetime.now(UTC).isoformat(),
                "private": args.private,
                "model_repo": args.model_repo,
                "model_commit": model_commit,
                "dataset_repo": args.dataset_repo,
                "dataset_commit": dataset_commit.oid,
                "dataset_contains_raw_qt30_archive_or_maps": dataset_manifest[
                    "contains_raw_qt30_archive_or_maps"
                ],
                "dataset_contains_transformed_qt30_text": dataset_manifest[
                    "contains_transformed_qt30_text"
                ],
                "dataset_contains_original_qt30_identifiers": dataset_manifest[
                    "contains_original_qt30_identifiers"
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output)


def _validate_manifest(directory: Path) -> dict:
    path = directory / "publish_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    declared: set[Path] = set()
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe publication path: {relative}")
        if relative in declared:
            raise ValueError(f"duplicate publication path: {relative}")
        declared.add(relative)
        artifact = directory / relative
        if not artifact.is_file():
            raise FileNotFoundError(artifact)
        observed = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if observed != entry["sha256"]:
            raise ValueError(f"publication hash mismatch: {relative}")
    actual = {
        path.relative_to(directory)
        for path in directory.rglob("*")
        if path.is_file() and path.name != "publish_manifest.json"
    }
    if actual != declared:
        missing = sorted(str(path) for path in declared - actual)
        undeclared = sorted(str(path) for path in actual - declared)
        raise ValueError(
            f"publication tree mismatch; missing={missing}, undeclared={undeclared}"
        )
    return manifest


if __name__ == "__main__":
    main()

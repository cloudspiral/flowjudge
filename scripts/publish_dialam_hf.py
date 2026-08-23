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
        description="Publish only the vetted DialAM model and text-free dataset artifacts"
    )
    parser.add_argument("--model-repo", required=True)
    parser.add_argument("--dataset-repo", required=True)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "hf_publish" / "dialam-qwen3-0.6b-v2",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=PROJECT_ROOT / "hf_dataset" / "dialam_patch",
    )
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    model_manifest = _validate_manifest(args.model_dir)
    dataset_manifest = _validate_manifest(args.dataset_dir)
    if not model_manifest.get("material_improvement_verified"):
        raise ValueError("model package does not record a material v2 improvement")
    if model_manifest.get("contains_raw_or_transformed_qt30_text") is not False:
        raise ValueError("model package redistribution boundary is not safe")
    if dataset_manifest.get("contains_raw_or_transformed_qt30_text") is not False:
        raise ValueError("dataset package redistribution boundary is not safe")

    load_dotenv(PROJECT_ROOT / ".env", override=False)
    token = os.getenv("HF_TOKEN", "").strip() or None
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for publication") from exc
    api = HfApi(token=token)
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
        commit_message="Publish Qwen3-0.6B DialAM v2 hard-negative QLoRA adapter and evidence",
    )
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
        commit_message="Publish text-free DialAM reconstruction manifests, schemas, and scripts",
    )
    output = PROJECT_ROOT / "docs" / "dialam_hf_publication_manifest.json"
    output.write_text(
        json.dumps(
            {
                "published_at": datetime.now(UTC).isoformat(),
                "private": args.private,
                "model_repo": args.model_repo,
                "model_commit": model_commit.oid,
                "dataset_repo": args.dataset_repo,
                "dataset_commit": dataset_commit.oid,
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
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts or relative.suffix == ".jsonl":
            raise ValueError(f"unsafe publication path: {relative}")
        artifact = directory / relative
        if not artifact.is_file():
            raise FileNotFoundError(artifact)
        observed = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if observed != entry["sha256"]:
            raise ValueError(f"publication hash mismatch: {relative}")
    return manifest


if __name__ == "__main__":
    main()

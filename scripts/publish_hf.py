#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish the final FlowJudge dataset and fused model to public Hugging Face repos"
    )
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--model-repo", required=True)
    parser.add_argument("--dataset-repo", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    token = os.getenv("HF_TOKEN", "").strip() or None
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("install the mlx-train or train dependency group first") from exc

    _validate_inputs(args.model_dir)
    api = HfApi(token=token)
    api.create_repo(args.model_repo, repo_type="model", private=False, exist_ok=True)
    model_commit = api.upload_folder(
        repo_id=args.model_repo,
        repo_type="model",
        folder_path=args.model_dir,
        commit_message="Publish FlowJudge Qwen3 0.6B QLoRA checkpoint and provenance",
    )

    api.create_repo(args.dataset_repo, repo_type="dataset", private=False, exist_ok=True)
    dataset_dir = PROJECT_ROOT / "data" / "training"
    uploads = [
        (dataset_dir / "DATASET_CARD.md", "README.md"),
        (dataset_dir / "v1_n96.jsonl", "train.jsonl"),
        (dataset_dir / "v1_n12.jsonl", "efficiency/v1_n12.jsonl"),
        (dataset_dir / "v1_n24.jsonl", "efficiency/v1_n24.jsonl"),
        (dataset_dir / "v1_n48.jsonl", "efficiency/v1_n48.jsonl"),
        (dataset_dir / "v1_manifest.json", "v1_manifest.json"),
        (dataset_dir / "split_manifest.json", "split_manifest.json"),
    ]
    dataset_commit = None
    for source, destination in uploads:
        dataset_commit = api.upload_file(
            repo_id=args.dataset_repo,
            repo_type="dataset",
            path_or_fileobj=source,
            path_in_repo=destination,
            commit_message=f"Publish FlowJudge dataset artifact {destination}",
        )

    publication = {
        "published_at": datetime.now(UTC).isoformat(),
        "model_repo": args.model_repo,
        "model_commit": model_commit.oid,
        "dataset_repo": args.dataset_repo,
        "dataset_commit": dataset_commit.oid if dataset_commit else None,
        "public": True,
    }
    output = PROJECT_ROOT / "docs" / "publication_manifest.json"
    output.write_text(json.dumps(publication, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


def _validate_inputs(model_dir: Path) -> None:
    required_model = ["config.json", "flowjudge_training_manifest.json", "README.md"]
    missing_model = [name for name in required_model if not (model_dir / name).exists()]
    if missing_model:
        raise FileNotFoundError(f"model directory is missing: {', '.join(missing_model)}")
    dataset_dir = PROJECT_ROOT / "data" / "training"
    required_data = [
        "DATASET_CARD.md",
        "v1_n12.jsonl",
        "v1_n24.jsonl",
        "v1_n48.jsonl",
        "v1_n96.jsonl",
        "v1_manifest.json",
        "split_manifest.json",
    ]
    missing_data = [name for name in required_data if not (dataset_dir / name).exists()]
    if missing_data:
        raise FileNotFoundError(f"dataset directory is missing: {', '.join(missing_data)}")


if __name__ == "__main__":
    main()

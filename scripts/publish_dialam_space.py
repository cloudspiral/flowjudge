#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime

from dotenv import load_dotenv

from flowjudge.patch_data import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish the public DialAM Gradio demo")
    parser.add_argument("--space-repo", default="mr-mc/flowjudge-dialam-demo")
    parser.add_argument(
        "--model-repo",
        default="mr-mc/flowjudge-dialam-qwen3-0.6b-v1-n2048",
    )
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env", override=False)
    token = os.getenv("HF_TOKEN", "").strip() or None
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for publication") from exc

    api = HfApi(token=token)
    api.create_repo(
        args.space_repo,
        repo_type="space",
        space_sdk="static",
        private=False,
        exist_ok=True,
    )
    commit = api.upload_folder(
        repo_id=args.space_repo,
        repo_type="space",
        folder_path=PROJECT_ROOT / "space",
        commit_message=(
            "Publish DialAM base-versus-tuned patch demo with the selected "
            "Qwen3-0.6B v1 n=2048 adapter"
        ),
    )
    path = PROJECT_ROOT / "docs" / "dialam_space_publication_manifest.json"
    path.write_text(
        json.dumps(
            {
                "published_at": datetime.now(UTC).isoformat(),
                "public": True,
                "space_repo": args.space_repo,
                "space_commit": commit.oid,
                "model_repo": args.model_repo,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(path)


if __name__ == "__main__":
    main()

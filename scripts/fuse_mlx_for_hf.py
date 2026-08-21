#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fuse a trained MLX QLoRA adapter into a portable Hugging Face checkpoint"
    )
    parser.add_argument("--model", default="mlx-community/Qwen3-0.6B-4bit")
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    command = [
        sys.executable,
        "-m",
        "mlx_lm.fuse",
        "--model",
        args.model,
        "--adapter-path",
        str(args.adapter),
        "--save-path",
        str(args.output_dir),
        "--dequantize",
    ]
    subprocess.run(command, check=True)

    source_manifest = args.adapter / "flowjudge_training_manifest.json"
    manifest = (
        json.loads(source_manifest.read_text(encoding="utf-8"))
        if source_manifest.exists()
        else {}
    )
    manifest.update(
        {
            "fused_at": datetime.now(UTC).isoformat(),
            "publication_format": "dequantized Hugging Face-compatible safetensors",
            "evaluation_base_model": "Qwen/Qwen3-0.6B",
            "fused_from_quantized_model": args.model,
            "fused_from_adapter": str(args.adapter),
            "fuse_command": command,
        }
    )
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    (args.output_dir / "flowjudge_training_manifest.json").write_text(
        manifest_text, encoding="utf-8"
    )
    model_card = Path(__file__).parents[1] / "docs" / "model_card.md"
    if model_card.exists():
        shutil.copyfile(model_card, args.output_dir / "README.md")


if __name__ == "__main__":
    main()

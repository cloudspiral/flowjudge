#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train FlowJudge with local Apple-Silicon MLX QLoRA")
    parser.add_argument("--model", default="mlx-community/Qwen3-0.6B-4bit")
    parser.add_argument("--canonical-base-model")
    parser.add_argument("--train-set", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260821)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_training(
        model_id=args.model,
        canonical_base_model=args.canonical_base_model,
        train_set=args.train_set,
        output_dir=args.output_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        seed=args.seed,
    )


def run_training(
    *,
    model_id: str,
    canonical_base_model: str | None,
    train_set: Path,
    output_dir: Path,
    epochs: float,
    learning_rate: float,
    batch_size: int,
    gradient_accumulation_steps: int,
    max_length: int,
    seed: int,
) -> None:
    rows = [json.loads(line) for line in train_set.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError("training set is empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "mlx_data"
    adapter_dir = output_dir / "adapter"
    data_dir.mkdir(parents=True, exist_ok=True)
    mlx_rows = [_disable_qwen_thinking(row["messages"]) for row in rows]
    (data_dir / "train.jsonl").write_text(
        "".join(json.dumps({"messages": messages}, ensure_ascii=False) + "\n" for messages in mlx_rows),
        encoding="utf-8",
    )
    # MLX-LM evaluates at the first and last iterations when a validation set is
    # present. Keep this diagnostic deterministic without reducing the stated
    # training-set size; the held-out benchmark remains entirely separate.
    (data_dir / "valid.jsonl").write_text(
        json.dumps({"messages": mlx_rows[0]}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    iterations = max(1, math.ceil(len(rows) * epochs / batch_size))
    command = [
        sys.executable,
        "-m",
        "mlx_lm.lora",
        "--model",
        model_id,
        "--train",
        "--fine-tune-type",
        "lora",
        "--data",
        str(data_dir),
        "--iters",
        str(iterations),
        "--batch-size",
        str(batch_size),
        "--grad-accumulation-steps",
        str(gradient_accumulation_steps),
        "--learning-rate",
        str(learning_rate),
        "--num-layers",
        "16",
        "--max-seq-length",
        str(max_length),
        "--adapter-path",
        str(adapter_dir),
        "--mask-prompt",
        "--seed",
        str(seed),
        "--steps-per-report",
        "1",
        "--save-every",
        str(iterations),
    ]
    log_path = output_dir / "training.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)
            log_file.flush()
        exit_code = process.wait()
    if exit_code != 0:
        raise subprocess.CalledProcessError(exit_code, command)

    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "framework": "mlx-lm",
        "method": "QLoRA because the loaded base checkpoint is 4-bit quantized",
        "canonical_base_model": canonical_base_model or _canonical_base(model_id),
        "quantized_base_model": model_id,
        "train_set": str(train_set),
        "training_examples": len(rows),
        "epochs": epochs,
        "iterations": iterations,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "max_length": max_length,
        "seed": seed,
        "num_lora_layers": 16,
        "mask_prompt": True,
        "qwen_thinking_disabled": True,
        "adapter_path": str(adapter_dir),
        "training_log": str(log_path),
        "command": command,
    }
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    (output_dir / "flowjudge_training_manifest.json").write_text(
        manifest_text, encoding="utf-8"
    )
    # Keep the same provenance in the upload-ready adapter directory.
    (adapter_dir / "flowjudge_training_manifest.json").write_text(
        manifest_text, encoding="utf-8"
    )


def _disable_qwen_thinking(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    copied = [dict(message) for message in messages]
    for message in copied:
        if message.get("role") == "user":
            message["content"] = message["content"].rstrip() + "\n/no_think"
            break
    return copied


def _canonical_base(model_id: str) -> str:
    name = model_id.rsplit("/", 1)[-1].removesuffix("-4bit")
    return f"Qwen/{name}" if name.startswith("Qwen3-") else model_id


if __name__ == "__main__":
    main()

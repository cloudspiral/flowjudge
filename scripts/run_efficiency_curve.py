#!/usr/bin/env python3
from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from pathlib import Path

SIZES = (12, 24, 48, 96)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the four FlowJudge data-efficiency points")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--data-dir", type=Path, default=Path("data/training"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/efficiency"))
    parser.add_argument("--eval-set", type=Path, default=Path("data/eval/own_eval.jsonl"))
    parser.add_argument("--eval-output-dir", type=Path, default=Path("results/efficiency"))
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--backend", choices=("auto", "cuda", "mlx"), default="auto")
    parser.add_argument("--judge-model")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    backend = args.backend
    if backend == "auto":
        backend = "mlx" if platform.system() == "Darwin" and platform.machine() == "arm64" else "cuda"
    if backend == "mlx" and args.model == "Qwen/Qwen3-0.6B":
        args.model = "mlx-community/Qwen3-0.6B-4bit"

    for size in SIZES:
        train_set = args.data_dir / f"v1_n{size}.jsonl"
        checkpoint = args.output_dir / f"n{size}"
        training_script = "scripts/train_mlx_qlora.py" if backend == "mlx" else "scripts/train_qlora.py"
        command = [
            sys.executable,
            training_script,
            "--model",
            args.model,
            "--train-set",
            str(train_set),
            "--output-dir",
            str(checkpoint),
            "--epochs",
            str(args.epochs),
        ]
        subprocess.run(command, check=True)
        if args.skip_eval:
            continue
        model_path = checkpoint / "adapter" if backend == "mlx" else checkpoint
        eval_command = [
            sys.executable,
            "eval.py",
            "--model",
            str(model_path),
            "--eval-set",
            str(args.eval_set),
            "--output-dir",
            str(args.eval_output_dir / f"n{size}"),
            "--backend",
            "mlx" if backend == "mlx" else "transformers",
        ]
        if args.judge_model:
            eval_command.extend(["--judge-model", args.judge_model])
        subprocess.run(eval_command, check=True)


if __name__ == "__main__":
    main()

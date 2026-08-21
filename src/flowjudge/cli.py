from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import load_benchmark
from .review import DEFAULT_REVIEW_PATH, generate_review_page
from .runner import dry_run_manifest, run_experiment
from .scorer import score_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FlowJudge benchmark tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("validate", help="validate and join benchmark JSONL files")

    review_parser = subparsers.add_parser("build-review", help="regenerate the compact benchmark review page")
    review_parser.add_argument("--output", type=Path, default=DEFAULT_REVIEW_PATH)

    subparsers.add_parser("dry-run", help="validate the 192-assignment matrix without API calls or keys")

    run_parser = subparsers.add_parser("run", help="run the approved online experiment")
    run_parser.add_argument("--approval", required=True, help="exact formal benchmark approval phrase")
    run_parser.add_argument("--openai-model", help="override OPENAI_MODEL without editing .env")
    run_parser.add_argument("--anthropic-model", help="override ANTHROPIC_MODEL without editing .env")
    run_parser.add_argument("--judge-model", help="override JUDGE_MODEL without editing .env")

    score_parser = subparsers.add_parser("score", help="recompute deterministic metrics for a run")
    score_parser.add_argument("run_directory", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "validate":
        cases = load_benchmark()
        print(json.dumps({"valid": True, "benchmark_scenarios": len(cases)}, indent=2))
    elif args.command == "build-review":
        print(generate_review_page(args.output))
    elif args.command == "dry-run":
        print(json.dumps(dry_run_manifest(), indent=2, sort_keys=True))
    elif args.command == "run":
        print(
            run_experiment(
                args.approval,
                openai_model=args.openai_model,
                anthropic_model=args.anthropic_model,
                judge_model=args.judge_model,
            )
        )
    elif args.command == "score":
        print(json.dumps(score_run(args.run_directory, load_benchmark()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

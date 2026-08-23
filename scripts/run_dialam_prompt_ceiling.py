from __future__ import annotations

import argparse
import json

from flowjudge.patch_ceiling import (
    APPROVAL_PHRASE,
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_OPENAI_MODEL,
    PREFLIGHT_APPROVAL_PHRASE,
    patch_ceiling_dry_run,
    run_patch_ceiling,
    run_patch_ceiling_preflight,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the DialAM prompt-ceiling gate")
    parser.add_argument(
        "--approval",
        help=(
            "exact phrase required for billable calls: "
            f"{PREFLIGHT_APPROVAL_PHRASE} for --preflight or {APPROVAL_PHRASE} for the full gate"
        ),
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="run one candidate and one blinded-judge call for each of the six model/prompt cells",
    )
    parser.add_argument(
        "--preflight-provider",
        action="append",
        choices=("openai", "anthropic"),
        help="limit a preflight to one provider; repeat to select both",
    )
    parser.add_argument("--openai-model", default=DEFAULT_OPENAI_MODEL)
    parser.add_argument("--anthropic-model", default=DEFAULT_ANTHROPIC_MODEL)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    args = parser.parse_args()

    model_args = {
        "openai_model": args.openai_model,
        "anthropic_model": args.anthropic_model,
        "judge_model": args.judge_model,
    }
    if args.approval is None:
        print(json.dumps(patch_ceiling_dry_run(**model_args), indent=2, sort_keys=True))
        return
    if args.preflight:
        providers = (
            tuple(dict.fromkeys(args.preflight_provider))
            if args.preflight_provider
            else ("openai", "anthropic")
        )
        print(
            run_patch_ceiling_preflight(
                args.approval,
                providers=providers,
                **model_args,
            )
        )
        return
    print(run_patch_ceiling(args.approval, **model_args))


if __name__ == "__main__":
    main()

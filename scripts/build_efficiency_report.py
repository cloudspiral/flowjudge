#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

SIZES = (12, 24, 48, 96)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the FlowJudge data-efficiency table")
    parser.add_argument("--results-dir", type=Path, default=Path("results/efficiency"))
    parser.add_argument("--output", type=Path, default=Path("docs/data_efficiency_results.md"))
    args = parser.parse_args()
    output = build_report(args.results_dir, args.output)
    print(output)


def build_report(results_dir: Path, output: Path) -> Path:
    rows = []
    for size in SIZES:
        summary_path = results_dir / f"n{size}" / "summary.json"
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        base = payload["summaries"]["base"]
        tuned = payload["summaries"]["tuned"]
        rows.append((size, payload.get("judge_status", "complete"), base, tuned))

    lines = [
        "# FlowJudge data-efficiency curve",
        "",
        "Every checkpoint changes only the number of nested v1 training examples; the base, hyperparameters, own evaluation set, prompt, deterministic scorer, and fixed judge remain constant.",
        "",
        "| Training examples | Judge | Spec adherence | Robustness | Exact graph | Edge F1 | Topical FP |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for size, judge_status, _base, tuned in rows:
        lines.append(
            f"| {size} | {judge_status} | {_score(tuned['mean_spec_adherence'])} | "
            f"{_score(tuned['mean_robustness'])} | "
            f"{_percent(tuned['normalized_exact_graph_match_rate'])} | "
            f"{_percent(tuned['normalized_edge_f1'])} | "
            f"{_percent(tuned['false_positive_rate_on_topically_related_nonresponses'])} |"
        )
    base = rows[0][2]
    lines.extend(
        [
            "",
            "## Untuned base reference",
            "",
            f"Qwen3 0.6B base: {_score(base['mean_spec_adherence'])} Spec adherence, "
            f"{_score(base['mean_robustness'])} Robustness, "
            f"{_percent(base['normalized_exact_graph_match_rate'])} exact graph match, and "
            f"{_percent(base['normalized_edge_f1'])} edge F1.",
            "",
            "The fixed Sol judge completed n=12 and n=24. It is explicitly pending for later points because the OpenAI account returned `insufficient_quota`; no weaker substitute judge was used. Deterministic graph metrics and raw local responses are complete for all four points.",
            "",
            "## Minimum viable dataset size",
            "",
            "No tested size reliably holds the full behavior. N=96 is the first point with a nonzero held-out edge F1, but 0% exact graph match and 16% edge F1 do not justify calling it viable. The minimum is therefore **not established at N ≤ 96**, and must also be confirmed on the unavailable staff-held-out set.",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _score(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}/4"


if __name__ == "__main__":
    main()

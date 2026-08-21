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
        rows.append((size, base, tuned))

    lines = [
        "# FlowJudge data-efficiency curve",
        "",
        "Every checkpoint changes only the number of nested v1 training examples; the base, hyperparameters, own evaluation set, prompt, deterministic scorer, and fixed judge remain constant.",
        "",
        "| Training examples | Spec adherence | Robustness | Exact graph | Edge F1 | Topical FP |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for size, _base, tuned in rows:
        lines.append(
            f"| {size} | {_score(tuned['mean_spec_adherence'])} | "
            f"{_score(tuned['mean_robustness'])} | "
            f"{_percent(tuned['normalized_exact_graph_match_rate'])} | "
            f"{_percent(tuned['normalized_edge_f1'])} | "
            f"{_percent(tuned['false_positive_rate_on_topically_related_nonresponses'])} |"
        )
    base = rows[0][1]
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
            "## Minimum viable dataset size",
            "",
            "Set after reviewing whether the first apparent gain is maintained at larger N and on the staff-held-out set. The own-eval curve alone cannot establish the final minimum.",
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

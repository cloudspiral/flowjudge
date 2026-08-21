from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .data import PROJECT_ROOT

DEFAULT_PROMPT_CEILING_REPORT = PROJECT_ROOT / "docs" / "prompt_ceiling_results.md"
PROMPT_ORDER = ("zero_shot", "few_shot", "structured_checklist")


def generate_prompt_ceiling_report(
    run_directory: Path,
    output_path: Path = DEFAULT_PROMPT_CEILING_REPORT,
) -> Path:
    run_directory = run_directory.resolve()
    output_path = output_path.resolve()
    manifest = json.loads((run_directory / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((run_directory / "summary.json").read_text(encoding="utf-8"))
    combinations = summary["by_model_prompt"]
    if not combinations:
        raise ValueError("run summary has no model/prompt combinations")
    if any(metrics.get("mean_spec_adherence") is None for metrics in combinations.values()):
        raise ValueError("run does not contain the assignment rubric scores")

    parsed = [_parse_combination(key, metrics) for key, metrics in combinations.items()]
    best = max(
        parsed,
        key=lambda row: (
            row["metrics"]["mean_spec_adherence"],
            row["metrics"]["mean_robustness"],
            row["metrics"]["normalized_exact_graph_match_rate"],
        ),
    )
    best_spec_rate = best["metrics"]["mean_spec_adherence"] / 4
    best_robustness_rate = best["metrics"]["mean_robustness"] / 4
    clears_gate = best_spec_rate >= 0.95 and best_robustness_rate >= 0.90
    verdict = "KILL OR HARDEN THE BEHAVIOR" if clears_gate else "SURVIVES PROMPT-CEILING GATE"

    lines = [
        "# FlowJudge prompt-ceiling ablation",
        "",
        f"**Verdict: {verdict}.**",
        "",
        (
            f"Best combination: `{best['provider']}:{best['model']} + {best['prompt']}` "
            f"with {best_spec_rate:.1%} mean Spec adherence, {best_robustness_rate:.1%} "
            f"mean Robustness, and {best['metrics']['normalized_exact_graph_match_rate']:.1%} "
            "deterministic exact graph match."
        ),
        "",
        "## Required 2 × 3 table",
        "",
        "Each cell is **mean Spec adherence / mean Robustness**, with each 0–4 judge score normalized to a percentage.",
        "",
        "| Candidate model | Zero-shot | Few-shot | Structured checklist |",
        "| --- | ---: | ---: | ---: |",
    ]
    by_model: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for row in parsed:
        by_model.setdefault((row["provider"], row["model"]), {})[row["prompt"]] = row["metrics"]
    for (provider, model), prompt_metrics in sorted(by_model.items()):
        cells = [
            _rubric_cell(prompt_metrics.get(prompt_name))
            for prompt_name in PROMPT_ORDER
        ]
        lines.append(f"| `{provider}:{model}` | {' | '.join(cells)} |")

    lines.extend(
        [
            "",
            "Gate used for the assignment decision: mean Spec adherence ≥95% and mean Robustness ≥90% in one model/prompt cell.",
            "",
            "## Deterministic cross-check",
            "",
            "| Candidate model | Prompt | Valid JSON | Exact graph | Edge precision | Edge recall | Edge F1 | Topical-nonresponse FP |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(parsed, key=lambda item: (item["provider"], item["model"], item["prompt"])):
        metrics = row["metrics"]
        lines.append(
            f"| `{row['provider']}:{row['model']}` | {row['prompt']} | "
            f"{_percent(metrics['normalized_valid_json_rate'])} | "
            f"{_percent(metrics['normalized_exact_graph_match_rate'])} | "
            f"{_percent(metrics['normalized_edge_precision'])} | "
            f"{_percent(metrics['normalized_edge_recall'])} | "
            f"{_percent(metrics['normalized_edge_f1'])} | "
            f"{_percent(metrics['false_positive_rate_on_topically_related_nonresponses'])} |"
        )

    best_prefix = f"{best['provider']}__{best['prompt']}__"
    best_failures = [
        failure
        for failure in summary["failure_cases"]
        if str(failure.get("assignment_id", "")).startswith(best_prefix)
        and failure.get("model") == best["model"]
    ]
    missed_count = sum(len(failure["missed_edges"]) for failure in best_failures)
    spurious_count = sum(len(failure["spurious_edges"]) for failure in best_failures)
    phenomenon_counts = Counter(
        phenomenon for failure in best_failures for phenomenon in failure["phenomena"]
    )
    recurring = phenomenon_counts.most_common(1)[0][0] if phenomenon_counts else "none"
    dominant_error = "missed direct responses" if missed_count >= spurious_count else "spurious response edges"
    lines.extend(
        [
            "",
            "## Failure that survives the best prompt",
            "",
            (
                f"The best cell still produced {len(best_failures)} non-exact scenarios, with "
                f"{missed_count} missed and {spurious_count} spurious edges. Its dominant recurring "
                f"failure was **{dominant_error}**, concentrated most often in scenarios tagged "
                f"`{recurring}`. This is the specific behavior the distilled dataset and error-driven "
                "v2 examples should target."
            ),
            "",
            "## Representative failures",
            "",
            "| Scenario | Phenomena | Missed edges | Spurious edges |",
            "| --- | --- | --- | --- |",
        ]
    )
    for failure in best_failures[:6]:
        lines.append(
            f"| `{failure['scenario_id']}` | {', '.join(failure['phenomena'])} | "
            f"`{json.dumps(failure['missed_edges'], separators=(',', ':'))}` | "
            f"`{json.dumps(failure['spurious_edges'], separators=(',', ':'))}` |"
        )

    lines.extend(
        [
            "",
            "## Reproduction record",
            "",
            f"- Run directory: `{run_directory.relative_to(PROJECT_ROOT)}`",
            f"- Candidate calls: {manifest['primary_call_count']}",
            f"- Fixed-judge calls: {manifest['judge_call_count']}",
            f"- Judge model: `{manifest['models']['judge']}`",
            f"- Fixed-judge valid rubric rate: {_percent(summary['fixed_judge_rubric_score_rate'])}",
            "- Raw candidate text, judge text, and complete SDK envelopes are preserved under the run directory.",
            "",
            "This ablation does not train or fine-tune any model.",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _parse_combination(key: str, metrics: dict[str, Any]) -> dict[str, Any]:
    provider, model, prompt = key.split(":", 2)
    return {"provider": provider, "model": model, "prompt": prompt, "metrics": metrics}


def _rubric_cell(metrics: dict[str, Any] | None) -> str:
    if metrics is None:
        return "n/a"
    return f"{metrics['mean_spec_adherence'] / 4:.1%} / {metrics['mean_robustness'] / 4:.1%}"


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"

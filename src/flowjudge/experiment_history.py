from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
from typing import Any

from .patch_data import PROJECT_ROOT


REPORTS_DIR = PROJECT_ROOT / "reports"
DEFAULT_LEDGER_PATH = REPORTS_DIR / "dialam_experiment_history.json"
DEFAULT_MARKDOWN_PATH = PROJECT_ROOT / "docs" / "dialam_experiment_history.md"
DEFAULT_CHART_PATH = REPORTS_DIR / "dialam_experiment_history.svg"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_payload(value: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    deterministic = value.get("deterministic_metrics", value)
    judge = value.get("judge_metrics", {})
    return deterministic, judge


def _point(
    *,
    run_id: str,
    label: str,
    evaluation_scope: str,
    n: int | None,
    intervention: str,
    status: str,
    value: dict[str, Any],
    source_artifact: str,
) -> dict[str, Any]:
    deterministic, judge = _metric_payload(value)
    none = deterministic.get("none_diagnostics", {})
    relation_metrics = deterministic.get("relation_metrics", {})
    return {
        "run_id": run_id,
        "label": label,
        "evaluation_scope": evaluation_scope,
        "n": n,
        "intervention": intervention,
        "status": status,
        "metrics": {
            "exact_patch_accuracy": deterministic.get("exact_patch_accuracy"),
            "edge_precision": deterministic.get("edge_precision"),
            "edge_recall": deterministic.get("edge_recall"),
            "edge_f1": deterministic.get("edge_f1"),
            "relation_macro_f1": deterministic.get("relation_macro_f1"),
            "false_edges_per_update": deterministic.get("false_edges_per_update"),
            "none_scenarios_with_false_edges": none.get("scenarios_with_false_edges"),
            "support_f1": relation_metrics.get("SUPPORT", {}).get("f1"),
            "attack_f1": relation_metrics.get("ATTACK", {}).get("f1"),
            "rephrase_f1": relation_metrics.get("REPHRASE", {}).get("f1"),
            "json_validity_rate": deterministic.get("json_validity_rate"),
            "schema_validity_rate": deterministic.get("schema_validity_rate"),
            "judge_spec_adherence_mean": judge.get("mean_spec_adherence"),
            "judge_robustness_mean": judge.get("mean_robustness"),
        },
        "source_artifact": source_artifact,
    }


def build_experiment_ledger(reports_dir: Path = REPORTS_DIR) -> dict[str, Any]:
    combined_path = reports_dir / "dialam_v1_v2_v3.json"
    curve_path = reports_dir / "dialam_v1_efficiency_curve.json"
    v4_path = reports_dir / "dialam_v4_class_balanced_rehearsal.json"
    combined = _load(combined_path)
    curve = _load(curve_path)
    v4 = _load(v4_path)

    points = [
        _point(
            run_id="base-generative-frozen",
            label="Base",
            evaluation_scope="frozen_30",
            n=None,
            intervention="Untouched Qwen3-0.6B with block-level free generation",
            status="baseline",
            value=combined["base"],
            source_artifact="reports/dialam_v1_v2_v3.json",
        )
    ]
    for row in curve["curve"]:
        n = int(row["n"])
        points.append(
            _point(
                run_id=f"v1-n{n}-frozen",
                label=f"v1/{n}",
                evaluation_scope="frozen_30",
                n=n,
                intervention="Fixed block-level QLoRA data-efficiency curve",
                status="v1 smoke" if n == 256 else "v1 curve",
                value=row,
                source_artifact="reports/dialam_v1_efficiency_curve.json",
            )
        )
    points.extend(
        [
            _point(
                run_id="v2-n2048-frozen",
                label="v2/2048",
                evaluation_scope="frozen_30",
                n=2048,
                intervention="More difficult NONE blocks",
                status="rejected: false edges worsened",
                value=combined["v2_n2048"],
                source_artifact="reports/dialam_v1_v2_v3.json",
            ),
            _point(
                run_id="v3-n4096-frozen",
                label="v3/4096",
                evaluation_scope="frozen_30",
                n=4096,
                intervention="Same-update contrast pairs plus per-example loss",
                status="selected public model",
                value=combined["v3_frozen_n4096"],
                source_artifact="reports/dialam_v1_v2_v3.json",
            ),
            _point(
                run_id="v3-n4096-development",
                label="v3/4096",
                evaluation_scope="episode_disjoint_development_30",
                n=4096,
                intervention="Same-update contrast pairs plus per-example loss",
                status="development baseline",
                value=v4["v3_development_n4096"],
                source_artifact="reports/dialam_v4_class_balanced_rehearsal.json",
            ),
            _point(
                run_id="v4-n8192-development",
                label="v4/8192",
                evaluation_scope="episode_disjoint_development_30",
                n=8192,
                intervention="Class-balanced block-level rehearsal",
                status="failed development NONE-calibration gate",
                value=v4["v4_development_n8192"],
                source_artifact="reports/dialam_v4_class_balanced_rehearsal.json",
            ),
        ]
    )

    v5_path = reports_dir / "dialam_v5_pairwise_classification.json"
    sources = [combined_path, curve_path, v4_path]
    if v5_path.exists():
        v5 = _load(v5_path)
        sources.append(v5_path)
        for key, run_id, label, status in (
            (
                "base_development_n8192",
                "base-pairwise-development",
                "Pairwise base",
                "v5 formulation baseline",
            ),
            (
                "v5_development_n8192",
                "v5-n8192-development",
                "v5/8192",
                v5.get("development_decision", "development result"),
            ),
        ):
            if key in v5:
                points.append(
                    _point(
                        run_id=run_id,
                        label=label,
                        evaluation_scope="episode_disjoint_development_30",
                        n=None if key.startswith("base") else 8192,
                        intervention="Complete-block pairwise four-label classification",
                        status=status,
                        value=v5[key],
                        source_artifact="reports/dialam_v5_pairwise_classification.json",
                    )
                )
        if "v5_frozen_n8192" in v5:
            points.append(
                _point(
                    run_id="v5-n8192-frozen",
                    label="v5/8192",
                    evaluation_scope="frozen_30",
                    n=8192,
                    intervention="Complete-block pairwise four-label classification",
                    status=v5.get("promotion_decision", "frozen result"),
                    value=v5["v5_frozen_n8192"],
                    source_artifact="reports/dialam_v5_pairwise_classification.json",
                )
            )

    v5_1_path = reports_dir / "dialam_v5_1_calibration.json"
    if v5_1_path.exists():
        v5_1 = _load(v5_1_path)
        sources.append(v5_1_path)
        selected = v5_1["selected"]
        points.append(
            _point(
                run_id="v5-1-n8192-development",
                label="v5.1/8192",
                evaluation_scope="episode_disjoint_development_30",
                n=8192,
                intervention=(
                    "Preregistered fixed-grid NONE-margin calibration over v5 scores"
                ),
                status=v5_1["development_decision"],
                value={
                    **selected["metrics"],
                    "none_diagnostics": selected["none_diagnostics"],
                },
                source_artifact="reports/dialam_v5_1_calibration.json",
            )
        )

    v5_1_result_path = reports_dir / "dialam_v5_1_result.json"
    if v5_1_result_path.exists():
        v5_1_result = _load(v5_1_result_path)
        sources.append(v5_1_result_path)
        if v5_1_result.get("promotion_passed"):
            for point in points:
                if point["run_id"] == "v3-n4096-frozen":
                    point["status"] = "previous selected model"
        points.append(
            _point(
                run_id="v5-1-n8192-frozen",
                label="v5.1/8192",
                evaluation_scope="frozen_30",
                n=8192,
                intervention=(
                    "Pairwise four-label scoring plus preregistered fixed NONE margin"
                ),
                status=v5_1_result["selection_decision"],
                value=v5_1_result["v5_1_frozen_n8192"],
                source_artifact="reports/dialam_v5_1_result.json",
            )
        )

    v6_path = reports_dir / "dialam_v6_calibration.json"
    if v6_path.exists():
        v6 = _load(v6_path)
        sources.append(v6_path)
        selected = v6["selected"]
        points.append(
            _point(
                run_id="v6-1-n12288-development",
                label="v6.1/12288",
                evaluation_scope="episode_disjoint_development_30",
                n=12288,
                intervention=(
                    "VivesDebate pairwise warm-up, exact QT30 v5 target stage, "
                    "and development-only NONE calibration"
                ),
                status=v6["development_decision"],
                value={
                    **selected["metrics"],
                    "none_diagnostics": selected["none_diagnostics"],
                },
                source_artifact="reports/dialam_v6_calibration.json",
            )
        )

    v6_result_path = reports_dir / "dialam_v6_result.json"
    if v6_result_path.exists():
        v6_result = _load(v6_result_path)
        sources.append(v6_result_path)
        if "v6_1_frozen_n12288" in v6_result:
            if v6_result.get("promotion_passed"):
                for point in points:
                    if point["run_id"] == "v5-1-n8192-frozen":
                        point["status"] = "previous selected model"
            points.append(
                _point(
                    run_id="v6-1-n12288-frozen",
                    label="v6.1/12288",
                    evaluation_scope="frozen_30",
                    n=12288,
                    intervention=(
                        "VivesDebate pairwise warm-up, exact QT30 v5 target stage, "
                        "and locked NONE margin"
                    ),
                    status=v6_result["selection_decision"],
                    value=v6_result["v6_1_frozen_n12288"],
                    source_artifact="reports/dialam_v6_result.json",
                )
            )

    v7_path = reports_dir / "dialam_v7_calibration.json"
    if v7_path.exists():
        v7 = _load(v7_path)
        sources.append(v7_path)
        selected = v7["selected"]
        points.append(
            _point(
                run_id="v7-1-n8192-development",
                label="v7.1/8192",
                evaluation_scope="episode_disjoint_development_30",
                n=8192,
                intervention=(
                    "QT30-only reciprocal candidate-preference continuation from v5 "
                    "plus development-only NONE calibration"
                ),
                status=v7["development_decision"],
                value={
                    **selected["metrics"],
                    "none_diagnostics": selected["none_diagnostics"],
                },
                source_artifact="reports/dialam_v7_calibration.json",
            )
        )

    v7_result_path = reports_dir / "dialam_v7_result.json"
    if v7_result_path.exists():
        v7_result = _load(v7_result_path)
        sources.append(v7_result_path)
        if "v7_1_frozen_n8192" in v7_result:
            if v7_result.get("promotion_passed"):
                for point in points:
                    if point["run_id"] == "v5-1-n8192-frozen":
                        point["status"] = "previous selected model"
            points.append(
                _point(
                    run_id="v7-1-n8192-frozen",
                    label="v7.1/8192",
                    evaluation_scope="frozen_30",
                    n=8192,
                    intervention=(
                        "QT30-only reciprocal candidate-preference continuation from v5 "
                        "plus locked NONE margin"
                    ),
                    status=v7_result["selection_decision"],
                    value=v7_result["v7_1_frozen_n8192"],
                    source_artifact="reports/dialam_v7_result.json",
                )
            )

    v7_2_path = reports_dir / "dialam_v7_2_calibration.json"
    if v7_2_path.exists():
        v7_2 = _load(v7_2_path)
        sources.append(v7_2_path)
        selected = v7_2["selected"]
        points.append(
            _point(
                run_id="v7-2-n8192-development",
                label="v7.2/8192",
                evaluation_scope="episode_disjoint_development_30",
                n=8192,
                intervention=(
                    "Transition-complete score-only NONE calibration over fixed v7 "
                    "development scores"
                ),
                status=v7_2["development_decision"],
                value={
                    **selected["metrics"],
                    "none_diagnostics": selected["none_diagnostics"],
                },
                source_artifact="reports/dialam_v7_2_calibration.json",
            )
        )

    v8_path = reports_dir / "dialam_v8_calibration.json"
    if v8_path.exists():
        v8 = _load(v8_path)
        sources.append(v8_path)
        selected = v8["selected"]
        points.append(
            _point(
                run_id="v8-1-n12288-development",
                label="v8.1/12288",
                evaluation_scope="episode_disjoint_development_30",
                n=12288,
                intervention=(
                    "QT30 prior-aware one-positive/two-hard-NONE groups with "
                    "restricted four-label score cross-entropy"
                ),
                status=v8["development_decision"],
                value={
                    **selected["metrics"],
                    "none_diagnostics": selected["none_diagnostics"],
                },
                source_artifact="reports/dialam_v8_calibration.json",
            )
        )

    v8_result_path = reports_dir / "dialam_v8_result.json"
    if v8_result_path.exists():
        v8_result = _load(v8_result_path)
        sources.append(v8_result_path)
        if "v8_1_frozen_n12288" in v8_result:
            if v8_result.get("promotion_passed"):
                for point in points:
                    if point["run_id"] == "v5-1-n8192-frozen":
                        point["status"] = "previous selected model"
            points.append(
                _point(
                    run_id="v8-1-n12288-frozen",
                    label="v8.1/12288",
                    evaluation_scope="frozen_30",
                    n=12288,
                    intervention=(
                        "QT30 prior-aware restricted-label listwise training plus "
                        "locked NONE margin"
                    ),
                    status=v8_result["selection_decision"],
                    value=v8_result["v8_1_frozen_n12288"],
                    source_artifact="reports/dialam_v8_result.json",
                )
            )

    v9_path = reports_dir / "dialam_v9_calibration.json"
    if v9_path.exists():
        v9 = _load(v9_path)
        sources.append(v9_path)
        selected = v9["selected"]
        points.append(
            _point(
                run_id="v9-1-n8192-development",
                label="v9.1/8192",
                evaluation_scope="episode_disjoint_development_30",
                n=8192,
                intervention=(
                    "QT30 selected-model hard-NONE corrective continuation from v5.1 "
                    "with restricted four-label score cross-entropy"
                ),
                status=v9["development_decision"],
                value={
                    **selected["metrics"],
                    "none_diagnostics": selected["none_diagnostics"],
                },
                source_artifact="reports/dialam_v9_calibration.json",
            )
        )

    v9_result_path = reports_dir / "dialam_v9_result.json"
    if v9_result_path.exists():
        v9_result = _load(v9_result_path)
        sources.append(v9_result_path)
        if "v9_1_frozen_n8192" in v9_result:
            if v9_result.get("promotion_passed"):
                for point in points:
                    if point["run_id"] == "v5-1-n8192-frozen":
                        point["status"] = "previous selected model"
            points.append(
                _point(
                    run_id="v9-1-n8192-frozen",
                    label="v9.1/8192",
                    evaluation_scope="frozen_30",
                    n=8192,
                    intervention=(
                        "QT30 selected-model hard-negative corrective continuation "
                        "plus locked NONE margin"
                    ),
                    status=v9_result["selection_decision"],
                    value=v9_result["v9_1_frozen_n8192"],
                    source_artifact="reports/dialam_v9_result.json",
                )
            )

    return {
        "schema_version": "dialam_experiment_history_v1",
        "scope_warning": (
            "Compare points only within one evaluation_scope. Development-only results "
            "are not frozen-benchmark claims."
        ),
        "points": points,
        "source_artifacts": [
            {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "sha256": _sha256(path),
            }
            for path in sources
        ],
    }


def _format_metric(value: Any, *, percent: bool = False) -> str:
    if value is None:
        return "—"
    if percent:
        return f"{100 * float(value):.1f}%"
    return f"{float(value):.3f}"


def render_experiment_markdown(ledger: dict[str, Any]) -> str:
    lines = [
        "# DialAM experiment history",
        "",
        "This ledger is generated from persisted aggregate reports. Development and frozen",
        "results are deliberately separated; they are not interchangeable evaluation sets.",
        "",
        "![DialAM improvement chart](../reports/dialam_experiment_history.svg)",
        "",
    ]
    for scope, title in (
        ("frozen_30", "Frozen 30-scenario benchmark"),
        ("episode_disjoint_development_30", "Episode-disjoint 30-scenario development set"),
    ):
        lines.extend(
            [
                f"## {title}",
                "",
                "| Run | N | Exact patch | Edge F1 | Macro-F1 | False edges/update | NONE FP cases | Status |",
                "|---|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for point in ledger["points"]:
            if point["evaluation_scope"] != scope:
                continue
            metrics = point["metrics"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        point["label"],
                        str(point["n"]) if point["n"] is not None else "—",
                        _format_metric(metrics["exact_patch_accuracy"], percent=True),
                        _format_metric(metrics["edge_f1"], percent=True),
                        _format_metric(metrics["relation_macro_f1"], percent=True),
                        _format_metric(metrics["false_edges_per_update"]),
                        str(metrics["none_scenarios_with_false_edges"])
                        if metrics["none_scenarios_with_false_edges"] is not None
                        else "—",
                        point["status"],
                    ]
                )
                + " |"
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "- The frozen series records the real base-to-selected-model improvement.",
            "- The development series records later formulation experiments without presenting them as frozen gains.",
            "- Failed gates remain in the ledger because they establish what changed and why it was rejected.",
            "",
            "Regenerate with:",
            "",
            "```bash",
            ".venv/bin/python scripts/build_dialam_experiment_history.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _svg_panel(
    points: list[dict[str, Any]],
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    title: str,
) -> list[str]:
    chart_left = x + 58
    chart_top = y + 42
    chart_width = width - 78
    chart_height = height - 105
    elements = [
        f'<text x="{x + width / 2}" y="{y + 22}" text-anchor="middle" class="panel-title">{html.escape(title)}</text>',
        f'<rect x="{chart_left}" y="{chart_top}" width="{chart_width}" height="{chart_height}" class="plot"/>',
    ]
    for tick in range(0, 101, 25):
        tick_y = chart_top + chart_height - chart_height * tick / 100
        elements.extend(
            [
                f'<line x1="{chart_left}" y1="{tick_y}" x2="{chart_left + chart_width}" y2="{tick_y}" class="grid"/>',
                f'<text x="{chart_left - 8}" y="{tick_y + 4}" text-anchor="end" class="tick">{tick}%</text>',
            ]
        )
    if not points:
        return elements
    x_step = chart_width / max(1, len(points) - 1)
    series = (
        ("exact_patch_accuracy", "Exact patch", "#2563eb"),
        ("edge_f1", "Edge F1", "#059669"),
        ("relation_macro_f1", "Macro-F1", "#d97706"),
    )
    for metric, _, color in series:
        coordinates = []
        for index, point in enumerate(points):
            value = point["metrics"].get(metric)
            if value is None:
                continue
            point_x = chart_left + index * x_step
            point_y = chart_top + chart_height * (1 - float(value))
            coordinates.append((point_x, point_y))
        if coordinates:
            path = " ".join(
                ("M" if index == 0 else "L") + f" {point_x:.1f} {point_y:.1f}"
                for index, (point_x, point_y) in enumerate(coordinates)
            )
            elements.append(f'<path d="{path}" fill="none" stroke="{color}" class="series"/>')
            elements.extend(
                f'<circle cx="{point_x:.1f}" cy="{point_y:.1f}" r="4" fill="{color}"/>'
                for point_x, point_y in coordinates
            )
    for index, point in enumerate(points):
        point_x = chart_left + index * x_step
        elements.append(
            f'<text x="{point_x:.1f}" y="{chart_top + chart_height + 20}" text-anchor="middle" class="x-label">{html.escape(point["label"])}</text>'
        )
    return elements


def render_experiment_svg(ledger: dict[str, Any]) -> str:
    frozen = [item for item in ledger["points"] if item["evaluation_scope"] == "frozen_30"]
    development = [
        item
        for item in ledger["points"]
        if item["evaluation_scope"] == "episode_disjoint_development_30"
    ]
    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="720" viewBox="0 0 1200 720">',
        "<style>",
        "text{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;fill:#172033}",
        ".title{font-size:24px;font-weight:700}.subtitle{font-size:13px;fill:#596579}",
        ".panel-title{font-size:16px;font-weight:650}.plot{fill:#fbfcfe;stroke:#cbd5e1}",
        ".grid{stroke:#e2e8f0;stroke-width:1}.tick,.x-label{font-size:11px;fill:#5d687a}",
        ".series{stroke-width:3;stroke-linejoin:round;stroke-linecap:round}",
        ".legend{font-size:12px;font-weight:600}",
        "</style>",
        '<rect width="1200" height="720" fill="#f5f7fb"/>',
        '<text x="600" y="34" text-anchor="middle" class="title">DialAM improvement history</text>',
        '<text x="600" y="56" text-anchor="middle" class="subtitle">Separate panels prevent development-only results from being presented as frozen-benchmark gains.</text>',
    ]
    elements.extend(
        _svg_panel(
            frozen,
            x=30,
            y=82,
            width=740,
            height=540,
            title="Frozen 30-scenario benchmark",
        )
    )
    elements.extend(
        _svg_panel(
            development,
            x=790,
            y=82,
            width=380,
            height=540,
            title="Episode-disjoint development",
        )
    )
    legend = (
        ("Exact patch", "#2563eb"),
        ("Edge F1", "#059669"),
        ("Macro-F1", "#d97706"),
    )
    for index, (label, color) in enumerate(legend):
        legend_x = 420 + index * 150
        elements.extend(
            [
                f'<line x1="{legend_x}" y1="670" x2="{legend_x + 26}" y2="670" stroke="{color}" class="series"/>',
                f'<text x="{legend_x + 34}" y="674" class="legend">{label}</text>',
            ]
        )
    elements.append("</svg>")
    return "\n".join(elements) + "\n"


def write_experiment_history(
    *,
    ledger_path: Path = DEFAULT_LEDGER_PATH,
    markdown_path: Path = DEFAULT_MARKDOWN_PATH,
    chart_path: Path = DEFAULT_CHART_PATH,
) -> dict[str, Path]:
    ledger = build_experiment_ledger(ledger_path.parent)
    ledger_path.write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_experiment_markdown(ledger), encoding="utf-8")
    chart_path.write_text(render_experiment_svg(ledger), encoding="utf-8")
    return {"ledger": ledger_path, "markdown": markdown_path, "chart": chart_path}

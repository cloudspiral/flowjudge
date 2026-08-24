from __future__ import annotations

import json
from pathlib import Path

import pytest

from flowjudge.experiment_history import (
    build_experiment_ledger,
    render_experiment_markdown,
    render_experiment_svg,
)


def test_experiment_history_separates_frozen_and_development_points() -> None:
    ledger = build_experiment_ledger()
    by_id = {item["run_id"]: item for item in ledger["points"]}

    assert by_id["base-generative-frozen"]["metrics"]["edge_f1"] == 0.0
    assert by_id["v3-n4096-frozen"]["metrics"]["edge_f1"] == pytest.approx(0.35)
    assert by_id["v3-n4096-frozen"]["status"] == "previous selected model"
    assert by_id["v5-1-n8192-frozen"]["metrics"]["edge_f1"] == pytest.approx(
        0.4761904761904762
    )
    assert by_id["v5-1-n8192-frozen"]["status"] in {
        "PROMOTE_V5_1_AS_FINAL_DIRECTION",
        "previous selected model",
    }
    assert by_id["v4-n8192-development"]["evaluation_scope"] == (
        "episode_disjoint_development_30"
    )
    assert "failed development" in by_id["v4-n8192-development"]["status"]

    if Path("reports/dialam_v6_calibration.json").exists():
        assert by_id["v6-1-n12288-development"]["evaluation_scope"] == (
            "episode_disjoint_development_30"
        )
    v6_result_path = Path("reports/dialam_v6_result.json")
    if v6_result_path.exists() and "v6_1_frozen_n12288" in json.loads(
        v6_result_path.read_text(encoding="utf-8")
    ):
        assert by_id["v6-1-n12288-frozen"]["evaluation_scope"] == "frozen_30"

    v8_result_path = Path("reports/dialam_v8_result.json")
    if v8_result_path.exists():
        v8 = by_id["v8-1-n12288-development"]
        assert v8["evaluation_scope"] == "episode_disjoint_development_30"
        assert v8["metrics"]["exact_patch_accuracy"] == 0.5
        assert v8["metrics"]["edge_f1"] == pytest.approx(0.52)
        assert v8["metrics"]["none_scenarios_with_false_edges"] == 3
        assert v8["status"] == "RETAIN_V5_1_V8_FAILED_DEVELOPMENT_GATE"


def test_experiment_history_renderers_include_scope_warning_and_series() -> None:
    ledger = build_experiment_ledger()
    markdown = render_experiment_markdown(ledger)
    svg = render_experiment_svg(ledger)

    assert "Frozen 30-scenario benchmark" in markdown
    assert "Episode-disjoint 30-scenario development set" in markdown
    assert "v3/4096" in markdown
    assert "v5.1/8192" in markdown
    if Path("reports/dialam_v8_result.json").exists():
        assert "v8.1/12288" in markdown
        assert "v8.1/12288" in svg
    assert "DialAM improvement history" in svg
    assert "Edge F1" in svg

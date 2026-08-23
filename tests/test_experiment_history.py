from __future__ import annotations

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
    assert by_id["v5-1-n8192-frozen"]["status"] == (
        "PROMOTE_V5_1_AS_FINAL_DIRECTION"
    )
    assert by_id["v4-n8192-development"]["evaluation_scope"] == (
        "episode_disjoint_development_30"
    )
    assert "failed development" in by_id["v4-n8192-development"]["status"]


def test_experiment_history_renderers_include_scope_warning_and_series() -> None:
    ledger = build_experiment_ledger()
    markdown = render_experiment_markdown(ledger)
    svg = render_experiment_svg(ledger)

    assert "Frozen 30-scenario benchmark" in markdown
    assert "Episode-disjoint 30-scenario development set" in markdown
    assert "v3/4096" in markdown
    assert "v5.1/8192" in markdown
    assert "DialAM improvement history" in svg
    assert "Edge F1" in svg

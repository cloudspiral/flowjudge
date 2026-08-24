from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT = PROJECT_ROOT / "reports" / "dialam_v7_2_result.json"


def test_v7_2_recovered_v7_score_scale_but_failed_development_gate() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    v5_1 = report["v5_1_development_n8192"]
    v7_1 = report["v7_1_development_n8192"]
    v7_2 = report["v7_2_development_n8192"]

    assert report["selection_decision"] == (
        "RETAIN_V5_1_V7_2_FAILED_DEVELOPMENT_GATE"
    )
    assert report["development_gate_passed"] is False
    assert v7_2["exact_patch_accuracy"] == 0.5333333333333333
    assert v7_2["exact_patch_accuracy"] > v7_1["exact_patch_accuracy"]
    assert v7_2["exact_patch_accuracy"] == v5_1["exact_patch_accuracy"]
    assert v7_2["edge_f1"] == 0.5217391304347826
    assert v7_2["edge_f1"] > v7_1["edge_f1"]
    assert v7_2["edge_f1"] < v5_1["edge_f1"]
    assert v7_2["false_edges_per_update"] == 1 / 3
    assert v7_2["none_diagnostics"]["scenarios_with_false_edges"] == 2
    assert set(report["failed_development_conditions"]) == {
        "edge_f1",
        "false_edges_per_update",
        "relation_macro_f1",
    }


def test_v7_2_was_score_only_and_kept_frozen_evaluation_sealed() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    calibration = report["calibration"]

    assert calibration["candidate_margin_count"] == 31
    assert calibration["minimum_margin"] == 3.0
    assert calibration["maximum_margin"] == 11.88692478928715
    assert calibration["selected_margin"] == 5.505193236283958
    assert calibration["new_training"] is False
    assert calibration["new_model_calls"] is False
    assert report["frozen_status"] == (
        "NOT_RUN_PREREGISTERED_DEVELOPMENT_GATE_FAILED"
    )
    assert report["frozen_candidate_calls"] == 0
    assert report["frozen_judge_calls"] == 0
    assert report["promotion_passed"] is False

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT = PROJECT_ROOT / "reports" / "dialam_v7_result.json"


def test_v7_failed_development_gate_and_kept_frozen_eval_sealed() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["selection_decision"] == "RETAIN_V5_1_V7_FAILED_DEVELOPMENT_GATE"
    assert report["development_gate_passed"] is False
    assert report["frozen_status"] == "NOT_RUN_PREREGISTERED_DEVELOPMENT_GATE_FAILED"
    assert report["frozen_candidate_calls"] == 0
    assert report["frozen_judge_calls"] == 0
    assert "v7_1_frozen_n8192" not in report
    assert set(report["failed_development_conditions"]) == {
        "edge_f1",
        "exact_patch_accuracy",
        "false_edges_per_update",
        "none_scenarios_with_false_edges",
        "relation_macro_f1",
    }


def test_v7_result_records_regression_diagnosis_and_resumable_checkpoint() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    baseline = report["v5_1_development_n8192"]
    v7 = report["v7_1_development_n8192"]
    training = report["training"]

    assert v7["exact_patch_accuracy"] == 0.4 < baseline["exact_patch_accuracy"]
    assert v7["edge_f1"] == 0.4444444444444445 < baseline["edge_f1"]
    assert v7["false_edges_per_update"] == 0.6 > baseline["false_edges_per_update"]
    assert v7["none_diagnostics"]["scenarios_with_false_edges"] == 5
    assert report["failure_diagnosis"] == {
        "false_negative_edges": 12,
        "false_positive_edges": 18,
        "none_scenarios_with_false_edges": 5,
        "support_false_positive_edges": 12,
    }
    assert training["global_step"] == 1024
    assert training["reload_verified"] is True
    assert training["resumability"]["full_trainer_state"] is True
    assert training["resumability"]["volume_commit_on_save"] is True
    assert training["idempotent_reuse_check"]["returned_without_training"] is True
    assert training["idempotent_reuse_check"]["global_step"] == 1024
    assert training["idempotent_reuse_check"]["adapter_tree_sha256"] == training[
        "adapter_tree_sha256"
    ]

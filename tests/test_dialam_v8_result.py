from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT = PROJECT_ROOT / "reports" / "dialam_v8_result.json"


def test_v8_failed_development_gate_and_kept_frozen_eval_sealed() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["selection_decision"] == "RETAIN_V5_1_V8_FAILED_DEVELOPMENT_GATE"
    assert report["development_gate_passed"] is False
    assert report["promotion_passed"] is False
    assert report["frozen_status"] == "NOT_RUN_PREREGISTERED_DEVELOPMENT_GATE_FAILED"
    assert report["frozen_candidate_calls"] == 0
    assert report["frozen_judge_calls"] == 0
    assert "v8_1_frozen_n12288" not in report
    assert set(report["failed_development_conditions"]) == {
        "edge_f1",
        "exact_patch_accuracy",
        "false_edges_per_update",
        "none_scenarios_with_false_edges",
        "relation_macro_f1",
    }


def test_v8_result_records_metrics_and_support_overprediction() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    baseline = report["v5_1_development_n8192"]
    v8 = report["v8_1_development_n12288"]
    diagnosis = report["development_diagnosis"]

    assert v8["exact_patch_accuracy"] == 0.5 < baseline["exact_patch_accuracy"]
    assert v8["edge_f1"] == 0.52 < baseline["edge_f1"]
    assert v8["relation_macro_f1"] == 0.5317460317460317
    assert v8["false_edges_per_update"] == 13 / 30
    assert v8["none_diagnostics"]["scenarios_with_false_edges"] == 3
    assert diagnosis["dominant_false_positive_relation"] == "SUPPORT"
    assert diagnosis["dominant_false_positive_count"] == 10
    assert diagnosis["false_positive_edges"] == 13
    assert diagnosis["false_negative_edges"] == 11


def test_v8_result_proves_training_and_eval_idempotence() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    training = report["training"]
    eval_resume = report["development_evaluation_resume"]

    assert training["global_step"] == 1536
    assert training["reload_verified"] is True
    assert training["resumability"]["full_trainer_state"] is True
    assert training["resumability"]["volume_commit_on_save"] is True
    assert training["resumability"]["save_steps"] == 100
    assert training["idempotent_reuse_check"]["returned_without_training"] is True
    assert training["idempotent_reuse_check"]["global_step"] == 1536
    assert training["idempotent_reuse_check"]["adapter_tree_sha256"] == training[
        "adapter_tree_sha256"
    ]
    assert eval_resume["idempotent_reuse"] is True
    assert eval_resume["newly_completed_scenarios"] == 0
    assert eval_resume["resumed_completed_scenarios"] == 30
    assert eval_resume["total_scenarios"] == 30

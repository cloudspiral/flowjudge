from __future__ import annotations

import json
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT = PROJECT_ROOT / "reports" / "dialam_v9_result.json"


def _report() -> dict:
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_v9_failed_development_gate_and_kept_frozen_eval_sealed() -> None:
    report = _report()

    assert report["selection_decision"] == "RETAIN_V5_1_V9_FAILED_DEVELOPMENT_GATE"
    assert report["development_gate_passed"] is False
    assert report["promotion_passed"] is False
    assert report["frozen_status"] == "NOT_RUN_PREREGISTERED_DEVELOPMENT_GATE_FAILED"
    assert report["frozen_candidate_calls"] == 0
    assert report["frozen_judge_calls"] == 0
    assert "v9_1_frozen_n8192" not in report
    assert set(report["failed_development_conditions"]) == {
        "edge_f1",
        "exact_patch_accuracy",
        "relation_macro_f1",
    }


def test_v9_records_hard_negative_overcorrection() -> None:
    report = _report()
    baseline = report["v5_1_development_n8192"]
    v9 = report["v9_1_development_n8192"]
    diagnosis = report["development_diagnosis"]

    assert v9["exact_patch_accuracy"] == pytest.approx(0.3)
    assert v9["exact_patch_accuracy"] < baseline["exact_patch_accuracy"]
    assert v9["edge_f1"] == pytest.approx(2 / 9)
    assert v9["relation_macro_f1"] == pytest.approx(2 / 11)
    assert v9["false_edges_per_update"] == 0
    assert v9["none_diagnostics"]["scenarios_with_false_edges"] == 0
    assert diagnosis == {
        "dominant_false_positive_count": 0,
        "dominant_false_positive_relation": None,
        "failure_mode": "overcorrected_to_none_with_positive_recall_collapse",
        "false_negative_edges": 21,
        "false_positive_edges": 0,
        "gold_edge_count": 24,
        "none_scenarios_with_false_edges": 0,
        "relation_false_positive_counts": {
            "ATTACK": 0,
            "REPHRASE": 0,
            "SUPPORT": 0,
        },
        "rephrase_true_positive_edges": 0,
        "support_true_positive_edges": 0,
        "true_positive_edges": 3,
    }


def test_v9_proves_training_and_development_eval_idempotence() -> None:
    report = _report()
    training = report["training"]
    evaluation = report["development_evaluation_resume"]

    assert training["global_step"] == 1024
    assert training["reload_verified"] is True
    assert training["resumability"]["full_trainer_state"] is True
    assert training["resumability"]["volume_commit_on_save"] is True
    assert training["resumability"]["save_steps"] == 100
    assert training["idempotent_reuse_check"]["returned_without_training"] is True
    assert training["idempotent_reuse_check"]["global_step"] == 1024
    assert training["idempotent_reuse_check"]["adapter_tree_sha256"] == training[
        "adapter_tree_sha256"
    ]

    first = evaluation["first_run"]
    replay = evaluation["idempotent_replay"]
    assert first["newly_completed_scenarios"] == 30
    assert first["resumed_completed_scenarios"] == 0
    assert replay["idempotent_reuse"] is True
    assert replay["newly_completed_scenarios"] == 0
    assert replay["resumed_completed_scenarios"] == 30
    assert first["identity_sha256"] == replay["identity_sha256"]
    assert first["predictions_sha256"] == replay["predictions_sha256"]
    assert evaluation["prediction_bytes_identical"] is True


def test_v9_corpus_is_qt30_only_episode_disjoint_and_hard() -> None:
    corpus = _report()["corpus"]

    assert corpus["dialogue_leakage"] is False
    assert corpus["new_external_data"] is False
    assert corpus["row_role_counts"] == {
        "MINED_NONE": 4096,
        "POSITIVE_REHEARSAL": 4096,
    }
    assert corpus["row_label_counts"] == {
        "ATTACK": 1366,
        "NONE": 4096,
        "REPHRASE": 1365,
        "SUPPORT": 1365,
    }
    assert corpus["selected_v5_1_margin_false_predictions"] == 2127

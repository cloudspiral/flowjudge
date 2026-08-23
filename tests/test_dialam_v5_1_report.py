from __future__ import annotations

import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT = PROJECT_ROOT / "reports" / "dialam_v5_1_result.json"


def test_v5_1_result_promotes_only_after_every_frozen_check_passes() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["selection_decision"] == "PROMOTE_V5_1_AS_FINAL_DIRECTION"
    assert report["promotion_passed"] is True
    assert all(report["promotion_checks"].values())
    assert report["calibration"]["margin"] == 3.0
    assert report["calibration"]["checkpoint_or_training_changed"] is False
    assert report["evaluation_accounting"]["frozen_candidate_scenario_calls"] == 30
    assert report["evaluation_accounting"]["frozen_judge_calls"] == 30
    assert report["evaluation_accounting"]["judge_identity_blinded"] is True


def test_v5_1_result_records_real_improvement_without_overclaiming_reliability() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    v3 = report["v3_frozen_n4096"]
    v5_1 = report["v5_1_frozen_n8192"]

    assert v5_1["exact_patch_accuracy"] > v3["exact_patch_accuracy"]
    assert v5_1["edge_f1"] > v3["edge_f1"]
    assert v5_1["relation_macro_f1"] > v3["relation_macro_f1"]
    assert v5_1["false_edges_per_update"] < v3["false_edges_per_update"]
    assert v5_1["none_diagnostics"]["scenarios_with_false_edges"] == 0
    assert report["clears_original_reliability_bar"] is False
    assert report["failed_original_reliability_conditions"]


def test_v5_1_result_hashes_every_private_evidence_artifact() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    for artifact in report["artifacts"].values():
        path = PROJECT_ROOT / artifact["path"]
        assert artifact["bytes"] > 0
        assert len(artifact["sha256"]) == 64
        if path.is_file():
            assert path.stat().st_size == artifact["bytes"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]

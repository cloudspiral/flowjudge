from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MINING_REPORT = PROJECT_ROOT / "reports" / "dialam_v9_mining_resume.json"
TRAINING_MANIFEST = (
    PROJECT_ROOT / "data" / "dialam" / "training" / "training_v9_manifest.json"
)


def test_v9_mining_proves_interruption_resume_and_idempotence() -> None:
    report = json.loads(MINING_REPORT.read_text(encoding="utf-8"))

    assert report["decision"] == "PASS_BUILD_V9_CORPUS"
    assert report["all_checks_passed"] is True
    assert all(report["checks"].values())
    assert report["intentional_interruption"]["committed_candidates"] == 128
    assert report["resume_completion"]["resumed_completed_chunks"] == 1
    assert report["resume_completion"]["newly_completed_chunks"] == 95
    assert report["resume_completion"]["total_candidates"] == 12288
    assert report["completed_replay"]["idempotent_reuse"] is True
    assert report["completed_replay"]["newly_completed_chunks"] == 0
    assert report["completed_replay"]["resumed_completed_chunks"] == 96


def test_v9_corpus_is_balanced_and_uses_only_training_episodes() -> None:
    manifest = json.loads(TRAINING_MANIFEST.read_text(encoding="utf-8"))

    assert manifest["size"] == 8192
    assert manifest["smoke_size"] == 256
    assert manifest["split_unit"] == "original_parent_episode"
    assert manifest["dialogue_leakage"] is False
    assert manifest["new_external_data"] is False
    assert manifest["row_role_counts"] == {
        "MINED_NONE": 4096,
        "POSITIVE_REHEARSAL": 4096,
    }
    assert manifest["row_label_counts"] == {
        "ATTACK": 1366,
        "NONE": 4096,
        "REPHRASE": 1365,
        "SUPPORT": 1365,
    }
    assert manifest["selected_v5_1_margin_false_predictions"] == 2127

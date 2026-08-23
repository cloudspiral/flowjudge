from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT = PROJECT_ROOT / "reports" / "dialam_v7_resume_smoke.json"


def test_v7_resume_smoke_proves_exact_recovery_and_idempotence() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["decision"] == "PASS"
    assert report["all_checks_passed"] is True
    assert all(report["checks"].values())
    assert report["intentional_interruption"]["committed_global_step"] == 8
    assert report["resume_completion"]["resumed_from_step"] == 8
    assert report["resume_completion"]["completed_global_step"] == 32
    assert report["idempotent_reuse"]["returned_without_training"] is True
    assert report["frozen_eval_touched"] is False
    assert report["judge_calls"] == 0

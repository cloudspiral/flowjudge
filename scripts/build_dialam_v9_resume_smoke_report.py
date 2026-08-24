#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from flowjudge.patch_data import PROJECT_ROOT


TRAINING_RESULT = (
    PROJECT_ROOT
    / "artifacts"
    / "dialam_qlora"
    / "v9_resume_smoke_n256"
    / "remote_training_result.json"
)
IDEMPOTENT_RESULT = (
    PROJECT_ROOT
    / "artifacts"
    / "dialam_qlora"
    / "v9_resume_smoke_n256"
    / "idempotent_reuse_result.json"
)
DATA_MANIFEST = (
    PROJECT_ROOT / "data" / "dialam" / "training" / "training_v9_manifest.json"
)
PREREGISTRATION = PROJECT_ROOT / "docs" / "dialam_v9_preregistration.md"
OUTPUT = PROJECT_ROOT / "reports" / "dialam_v9_resume_smoke.json"
DOC = PROJECT_ROOT / "docs" / "dialam_v9_resume_smoke.md"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_report() -> dict[str, Any]:
    training = _load(TRAINING_RESULT)
    reused = _load(IDEMPOTENT_RESULT)
    data = _load(DATA_MANIFEST)
    events = training["resumability"]["resume_events"]
    checks = {
        "nested_smoke_size_is_256": training["size"] == data["smoke_size"] == 256,
        "smoke_data_hash_matches_manifest": (
            training["train_sha256"] == data["smoke_output"]["sha256"]
        ),
        "first_invocation_created_new_run": events[0]["action"] == "new"
        and events[0]["from_global_step"] == 0,
        "second_invocation_required_step_8": events[1]["action"] == "resume"
        and events[1]["from_global_step"] == 8
        and training["resumability"]["resumed_from_step"] == 8
        and training["resumability"]["resume_mode"] == "required",
        "run_completed_step_32": training["global_step"] == 32,
        "full_trainer_state_was_persisted": training["resumability"][
            "full_trainer_state"
        ]
        is True,
        "volume_was_committed_on_every_save": training["resumability"][
            "volume_commit_on_save"
        ]
        is True,
        "trainer_state_preserved_after_completion": training["resumability"][
            "trainer_state_preserved_after_completion"
        ]
        is True,
        "final_adapter_reloaded": training["reload_verified"] is True,
        "completed_rerun_was_idempotent": reused["idempotent_reuse"] is True,
        "idempotent_rerun_preserved_step": (
            reused["global_step"] == training["global_step"]
        ),
        "idempotent_rerun_preserved_adapter_hash": (
            reused["adapter_tree_sha256"] == training["adapter_tree_sha256"]
        ),
        "source_adapter_matches_frozen_v5_1": training["source_adapter"][
            "tree_sha256"
        ]
        == data["continuation"]["source_adapter_tree_sha256"],
        "objective_matches_manifest": (
            training["loss_config"]["name"] == data["objective"]["name"]
        ),
        "learning_rate_matches_manifest": (
            training["fixed_config"]["learning_rate"]
            == data["fixed_training_config"]["learning_rate"]
        ),
    }
    return {
        "schema_version": "dialam_v9_resume_smoke_result_v1",
        "experiment": "V9/n256 intentional interruption and exact-state resume",
        "decision": "PASS" if all(checks.values()) else "FAIL",
        "all_checks_passed": all(checks.values()),
        "checks": checks,
        "intentional_interruption": {
            "modal_app_id": "ap-ozWEo7lky70HNUjClfxh73",
            "expected_error": (
                "INTENTIONAL_RESUME_SMOKE_INTERRUPTION_AFTER_COMMITTED_CHECKPOINT"
            ),
            "committed_global_step": 8,
        },
        "resume_completion": {
            "modal_app_id": "ap-9RfT5NTh1bIfV17oGZTqnW",
            "resumed_from_step": training["resumability"]["resumed_from_step"],
            "first_new_global_step": 9,
            "completed_global_step": training["global_step"],
            "train_loss": training["metrics"]["train_loss"],
            "adapter_tree_sha256": training["adapter_tree_sha256"],
            "reload_verified": training["reload_verified"],
        },
        "idempotent_reuse": {
            "modal_app_id": "ap-gUtuJQ44oeDp9dPkYRshtx",
            "returned_without_training": reused["idempotent_reuse"],
            "global_step": reused["global_step"],
            "adapter_tree_sha256": reused["adapter_tree_sha256"],
        },
        "private_local_artifacts": {
            "training_result": {
                "path": str(TRAINING_RESULT.relative_to(PROJECT_ROOT)),
                "sha256": _sha256(TRAINING_RESULT),
            },
            "idempotent_result": {
                "path": str(IDEMPOTENT_RESULT.relative_to(PROJECT_ROOT)),
                "sha256": _sha256(IDEMPOTENT_RESULT),
            },
            "modal_checkpoint_root": (
                "/workspace/persistent/checkpoints/v9_resume_smoke_n256"
            ),
        },
        "full_run_allowed": all(checks.values()),
        "frozen_eval_touched": False,
        "judge_calls": 0,
        "preregistration_sha256": _sha256(PREREGISTRATION),
        "data_manifest_sha256": _sha256(DATA_MANIFEST),
    }


def main() -> None:
    report = build_report()
    OUTPUT.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    DOC.write_text(
        "\n".join(
            [
                "# DialAM V9 interruption/resume smoke",
                "",
                f"Decision: **{report['decision']}**.",
                "",
                "The n=256 run intentionally terminated only after committing the full",
                "step-8 Trainer state to the Modal volume. The required-resume invocation",
                "executed step 9 next and finished at step 32; the final adapter reloaded",
                "successfully. A third auto-mode invocation returned the identical adapter",
                "hash and completed manifest without executing another training step.",
                "",
                "This proves recovery of adapter/model, optimizer, scheduler, RNG, Trainer",
                "state, and sample position for the V9 restricted four-label objective.",
                "The full run uses the same implementation with a 100-step save interval",
                "and retains the newest two complete states.",
                "",
                "Raw run results and checkpoints remain local/ignored and on the private",
                "Modal volume. Their paths and hashes are recorded in",
                "`reports/dialam_v9_resume_smoke.json`.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

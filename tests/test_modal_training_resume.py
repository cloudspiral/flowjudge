from __future__ import annotations

import json
from pathlib import Path

import pytest

from flowjudge.modal_training_resume import (
    checkpoint_validation_errors,
    object_sha256,
    prepare_resumable_run,
    write_json_atomic,
)


IDENTITY = {
    "schema_version": "test_run_identity_v1",
    "dataset_sha256": "data-hash",
    "source_adapter_sha256": "adapter-hash",
    "config": {"seed": 7, "learning_rate": 5e-5},
}


def _write_valid_checkpoint(trainer_dir: Path, step: int) -> Path:
    checkpoint = trainer_dir / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
    (checkpoint / "optimizer.pt").write_bytes(b"optimizer")
    (checkpoint / "scheduler.pt").write_bytes(b"scheduler")
    (checkpoint / "rng_state.pth").write_bytes(b"rng")
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": step}), encoding="utf-8"
    )
    return checkpoint


def test_new_run_writes_hashed_identity_and_required_mode_rejects_absence(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="resume required"):
        prepare_resumable_run(tmp_path / "missing", IDENTITY, resume_mode="required")

    run_dir = tmp_path / "run"
    decision = prepare_resumable_run(run_dir, IDENTITY, resume_mode="auto")

    assert decision.action == "new"
    stored = json.loads((run_dir / "run_identity.json").read_text())
    assert stored["identity_sha256"] == object_sha256(IDENTITY)


def test_resume_selects_latest_valid_checkpoint_and_ignores_newer_partial(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    prepare_resumable_run(run_dir, IDENTITY, resume_mode="auto")
    valid = _write_valid_checkpoint(run_dir / "trainer", 8)
    partial = run_dir / "trainer" / "checkpoint-16"
    partial.mkdir()
    (partial / "trainer_state.json").write_text(
        json.dumps({"global_step": 16}), encoding="utf-8"
    )

    decision = prepare_resumable_run(run_dir, IDENTITY, resume_mode="required")

    assert decision.action == "resume"
    assert decision.checkpoint == valid
    assert decision.global_step == 8
    assert decision.ignored_incomplete_checkpoints


def test_resume_rejects_identity_drift_and_missing_full_state(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    prepare_resumable_run(run_dir, IDENTITY, resume_mode="auto")
    partial = run_dir / "trainer" / "checkpoint-8"
    partial.mkdir(parents=True)
    (partial / "trainer_state.json").write_text(
        json.dumps({"global_step": 8}), encoding="utf-8"
    )

    assert "optimizer.pt" in checkpoint_validation_errors(partial)
    with pytest.raises(RuntimeError, match="no resumable checkpoint"):
        prepare_resumable_run(run_dir, IDENTITY, resume_mode="auto")
    with pytest.raises(RuntimeError, match="identity mismatch"):
        prepare_resumable_run(
            run_dir,
            {**IDENTITY, "dataset_sha256": "changed"},
            resume_mode="auto",
        )


def test_completed_run_is_idempotent_but_never_mode_refuses(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    prepare_resumable_run(run_dir, IDENTITY, resume_mode="auto")
    (run_dir / "adapter").mkdir()
    (run_dir / "adapter" / "adapter_model.safetensors").write_bytes(b"done")
    write_json_atomic(
        run_dir / "training_manifest.json",
        {"run_identity_sha256": object_sha256(IDENTITY), "global_step": 32},
    )

    decision = prepare_resumable_run(run_dir, IDENTITY, resume_mode="auto")
    assert decision.action == "complete"
    assert decision.global_step == 32

    with pytest.raises(FileExistsError, match="completed run"):
        prepare_resumable_run(run_dir, IDENTITY, resume_mode="never")

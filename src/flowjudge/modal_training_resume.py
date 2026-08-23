from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


ResumeMode = Literal["never", "auto", "required"]
ResumeAction = Literal["new", "resume", "complete"]

RUN_IDENTITY_FILENAME = "run_identity.json"
TRAINING_MANIFEST_FILENAME = "training_manifest.json"
PROGRESS_FILENAME = "training_progress.json"
CHECKPOINT_PATTERN = re.compile(r"^checkpoint-(\d+)$")
REQUIRED_TRAINER_STATE_FILES = (
    "trainer_state.json",
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
)
MODEL_STATE_FILENAMES = (
    "adapter_model.safetensors",
    "model.safetensors",
    "pytorch_model.bin",
)


@dataclass(frozen=True)
class ResumeDecision:
    action: ResumeAction
    checkpoint: Path | None
    global_step: int
    ignored_incomplete_checkpoints: tuple[str, ...] = ()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_manifest(path: Path) -> tuple[str, list[dict[str, Any]]]:
    digest = hashlib.sha256()
    files: list[dict[str, Any]] = []
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        relative = str(item.relative_to(path))
        item_hash = file_sha256(item)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(item_hash.encode("ascii"))
        digest.update(b"\n")
        files.append(
            {"path": relative, "sha256": item_hash, "bytes": item.stat().st_size}
        )
    if not files:
        raise FileNotFoundError(f"tree is empty or missing: {path}")
    return digest.hexdigest(), files


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def checkpoint_step(path: Path) -> int | None:
    match = CHECKPOINT_PATTERN.fullmatch(path.name)
    return int(match.group(1)) if match else None


def checkpoint_validation_errors(path: Path) -> list[str]:
    step = checkpoint_step(path)
    if step is None or not path.is_dir():
        return ["not a checkpoint-N directory"]
    errors = [name for name in REQUIRED_TRAINER_STATE_FILES if not (path / name).is_file()]
    if not any((path / name).is_file() for name in MODEL_STATE_FILENAMES):
        errors.append("missing model or adapter state")
    trainer_state_path = path / "trainer_state.json"
    if trainer_state_path.is_file():
        try:
            state = _load_json(trainer_state_path)
            if int(state.get("global_step", -1)) != step:
                errors.append("trainer_state global_step differs from directory")
        except (json.JSONDecodeError, TypeError, ValueError):
            errors.append("trainer_state.json is not valid JSON state")
    return errors


def latest_valid_checkpoint(
    trainer_dir: Path,
) -> tuple[Path | None, tuple[str, ...]]:
    candidates = sorted(
        (
            (step, path)
            for path in trainer_dir.glob("checkpoint-*")
            if (step := checkpoint_step(path)) is not None
        ),
        reverse=True,
    )
    incomplete: list[str] = []
    for _, path in candidates:
        errors = checkpoint_validation_errors(path)
        if not errors:
            return path, tuple(incomplete)
        incomplete.append(f"{path.name}: {', '.join(errors)}")
    return None, tuple(incomplete)


def prepare_resumable_run(
    run_dir: Path,
    expected_identity: dict[str, Any],
    *,
    resume_mode: ResumeMode,
) -> ResumeDecision:
    if resume_mode not in {"never", "auto", "required"}:
        raise ValueError("resume_mode must be never, auto, or required")

    identity_path = run_dir / RUN_IDENTITY_FILENAME
    completed_path = run_dir / TRAINING_MANIFEST_FILENAME
    expected_hash = object_sha256(expected_identity)

    if not run_dir.exists():
        if resume_mode == "required":
            raise FileNotFoundError(f"resume required but run directory is absent: {run_dir}")
        run_dir.mkdir(parents=True)
        write_json_atomic(
            identity_path,
            {**expected_identity, "identity_sha256": expected_hash},
        )
        return ResumeDecision(action="new", checkpoint=None, global_step=0)

    if not identity_path.is_file():
        raise RuntimeError(
            f"existing run has no identity manifest and will not be overwritten: {run_dir}"
        )
    observed_identity = _load_json(identity_path)
    observed_hash = observed_identity.pop("identity_sha256", None)
    if observed_hash != object_sha256(observed_identity):
        raise RuntimeError(f"stored run identity hash is invalid: {identity_path}")
    if observed_identity != expected_identity or observed_hash != expected_hash:
        raise RuntimeError(
            "resume identity mismatch; dataset, source adapter, config, seed, or code changed"
        )

    if completed_path.is_file():
        if resume_mode == "never":
            raise FileExistsError(f"completed run already exists: {run_dir}")
        completed = _load_json(completed_path)
        if completed.get("run_identity_sha256") != expected_hash:
            raise RuntimeError("completed manifest does not match the run identity")
        adapter_dir = run_dir / "adapter"
        if not adapter_dir.is_dir() or not any(adapter_dir.iterdir()):
            raise RuntimeError("completed run is missing its final adapter")
        return ResumeDecision(
            action="complete",
            checkpoint=None,
            global_step=int(completed.get("global_step", 0)),
        )

    if resume_mode == "never":
        raise FileExistsError(
            f"incomplete run already exists; use --resume-mode auto or required: {run_dir}"
        )

    checkpoint, incomplete = latest_valid_checkpoint(run_dir / "trainer")
    if checkpoint is None:
        detail = "; ".join(incomplete) if incomplete else "no checkpoint-N directories"
        raise RuntimeError(f"run exists but has no resumable checkpoint: {detail}")
    return ResumeDecision(
        action="resume",
        checkpoint=checkpoint,
        global_step=checkpoint_step(checkpoint) or 0,
        ignored_incomplete_checkpoints=incomplete,
    )

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any

from dotenv import load_dotenv
from pydantic import ValidationError

from .data import PROJECT_ROOT
from .training_data import (
    DATASET_SIZES,
    DEFAULT_CANDIDATES_PATH,
    QualityAssessment,
    TeacherRewrite,
    TrainingCandidate,
    TrainingExample,
    load_training_candidates,
    make_training_example,
    render_candidate_for_teacher,
    write_dataset_slices,
)

PROMPTS_DIR = PROJECT_ROOT / "prompts"


@dataclass(frozen=True)
class DistillationResult:
    index: int
    candidate: TrainingCandidate
    example: TrainingExample | None
    record: dict[str, Any]
    teacher_envelope: str | None
    filter_envelope: str | None


def distill_training_data(
    candidates_path: Path = DEFAULT_CANDIDATES_PATH,
    *,
    model: str | None = None,
    max_workers: int = 4,
    required_examples: int = DATASET_SIZES[-1],
) -> Path:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    teacher_model = (model or os.getenv("JUDGE_MODEL", "")).strip()
    if not api_key or not teacher_model:
        raise RuntimeError("OPENAI_API_KEY and a teacher model are required")
    if max_workers < 1 or max_workers > 8:
        raise ValueError("max_workers must be between 1 and 8")

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    candidates = load_training_candidates(candidates_path)
    if len(candidates) < required_examples:
        raise ValueError(
            f"candidate pool has {len(candidates)} rows but {required_examples} accepted rows are required"
        )

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir = PROJECT_ROOT / "results" / "data_generation" / stamp
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True)
    records_path = run_dir / "records.jsonl"
    write_lock = Lock()

    def process(index: int, candidate: TrainingCandidate) -> DistillationResult:
        return _distill_one(client, teacher_model, index, candidate)

    results: list[DistillationResult] = []
    with records_path.open("a", encoding="utf-8") as records_file:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process, index, candidate): (index, candidate)
                for index, candidate in enumerate(candidates)
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                with write_lock:
                    example_dir = raw_dir / result.candidate.example_id
                    example_dir.mkdir(parents=True, exist_ok=True)
                    if result.teacher_envelope is not None:
                        (example_dir / "teacher_envelope.json").write_text(
                            result.teacher_envelope, encoding="utf-8"
                        )
                    if result.filter_envelope is not None:
                        (example_dir / "filter_envelope.json").write_text(
                            result.filter_envelope, encoding="utf-8"
                        )
                    records_file.write(json.dumps(result.record, ensure_ascii=False) + "\n")
                    records_file.flush()
                print(
                    f"distilled {len(results)}/{len(candidates)} "
                    f"{result.candidate.example_id} accepted={result.example is not None}",
                    flush=True,
                )

    accepted = [
        result.example
        for result in sorted(results, key=lambda item: item.index)
        if result.example is not None
    ]
    if len(accepted) < required_examples:
        raise RuntimeError(
            f"quality filter accepted {len(accepted)}/{len(candidates)}; need {required_examples}"
        )
    accepted = accepted[:required_examples]
    dataset_paths = write_dataset_slices(accepted)
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "teacher_model": teacher_model,
        "filter_model": teacher_model,
        "candidate_count": len(candidates),
        "accepted_count_before_cap": sum(result.example is not None for result in results),
        "published_training_count": len(accepted),
        "dataset_sizes": [int(path.stem.removeprefix("v1_n")) for path in dataset_paths],
        "dataset_paths": [str(path.relative_to(PROJECT_ROOT)) for path in dataset_paths],
        "records_path": str(records_path.relative_to(PROJECT_ROOT)),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (PROJECT_ROOT / "data" / "training" / "v1_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return run_dir


def materialize_distillation_run(
    run_dir: Path,
    candidates_path: Path = DEFAULT_CANDIDATES_PATH,
    *,
    required_examples: int = DATASET_SIZES[-1],
) -> Path:
    """Rebuild dataset slices from preserved accepted responses using current prompt code."""
    candidates = load_training_candidates(candidates_path)
    existing_manifest_path = run_dir / "manifest.json"
    manifest = (
        json.loads(existing_manifest_path.read_text(encoding="utf-8"))
        if existing_manifest_path.exists()
        else {}
    )
    teacher_model = str(manifest.get("teacher_model", "preserved-teacher"))
    filter_model = str(manifest.get("filter_model", "preserved-filter"))
    records = {
        record["example_id"]: record
        for record in (
            json.loads(line)
            for line in (run_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        )
    }
    examples: list[TrainingExample] = []
    for candidate in candidates:
        record = records.get(candidate.example_id)
        if not record or not record.get("accepted"):
            continue
        rewrite = TeacherRewrite.model_validate_json(record["teacher_response"])
        quality = QualityAssessment.model_validate_json(record["filter_response"])
        examples.append(
            make_training_example(
                candidate,
                rewrite,
                quality,
                teacher_model=teacher_model,
                filter_model=filter_model,
            )
        )
    if len(examples) < required_examples:
        raise RuntimeError(
            f"preserved run contains {len(examples)} accepted examples; need {required_examples}"
        )
    examples = examples[:required_examples]
    dataset_paths = write_dataset_slices(examples)
    manifest.update(
        {
            "rematerialized_at": datetime.now(UTC).isoformat(),
            "published_training_count": len(examples),
            "dataset_sizes": [int(path.stem.removeprefix("v1_n")) for path in dataset_paths],
            "dataset_paths": [str(path.relative_to(PROJECT_ROOT)) for path in dataset_paths],
            "response_edge_label_leak_check": "training prompts omit source anchor-pair titles",
        }
    )
    existing_manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (PROJECT_ROOT / "data" / "training" / "v1_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return existing_manifest_path


def _distill_one(
    client: Any,
    model: str,
    index: int,
    candidate: TrainingCandidate,
) -> DistillationResult:
    base_teacher_prompt = _render_template(
        "data_teacher",
        {"{{CANDIDATE}}": render_candidate_for_teacher(candidate)},
    )
    teacher_prompt = base_teacher_prompt
    attempts: list[dict[str, Any]] = []
    teacher_envelopes: list[dict[str, Any]] = []
    filter_envelopes: list[dict[str, Any]] = []
    example: TrainingExample | None = None
    last_error = "teacher/filter attempts exhausted"
    for attempt_number in range(1, 4):
        teacher_text: str | None = None
        filter_text: str | None = None
        try:
            teacher_text, teacher_envelope = _call_openai(client, model, teacher_prompt)
            teacher_envelopes.append(json.loads(teacher_envelope))
            rewrite = TeacherRewrite.model_validate_json(teacher_text)
            expected = [(unit.id, unit.side) for unit in candidate.units]
            actual = [(unit.id, unit.side) for unit in rewrite.units]
            if actual != expected:
                raise ValueError("teacher changed unit IDs, order, or sides")

            filter_prompt = _render_template(
                "data_filter",
                {
                    "{{SOURCE}}": json.dumps(
                        [unit.model_dump() for unit in candidate.units],
                        ensure_ascii=False,
                        indent=2,
                    ),
                    "{{REWRITE}}": rewrite.model_dump_json(indent=2),
                    "{{GOLD_GRAPH}}": candidate.gold_graph.model_dump_json(indent=2),
                },
            )
            filter_text, filter_envelope = _call_openai(client, model, filter_prompt)
            filter_envelopes.append(json.loads(filter_envelope))
            quality = QualityAssessment.model_validate_json(filter_text)
            example = make_training_example(candidate, rewrite, quality, model, model)
            last_error = ""
        except (ValueError, ValidationError) as exc:
            last_error = str(exc)
        attempts.append(
            {
                "attempt": attempt_number,
                "teacher_response": teacher_text,
                "filter_response": filter_text,
                "error": last_error or None,
            }
        )
        if example is not None:
            break
        teacher_prompt = (
            base_teacher_prompt
            + "\n\nREVISION REQUIRED\n"
            + "The previous rewrite failed the independent filter. Correct the cited issue while "
            + "still obeying every original constraint. Do not defend the prior version.\n"
            + f"PREVIOUS REWRITE\n{teacher_text or 'No valid rewrite was produced.'}\n\n"
            + f"FILTER OR VALIDATION FEEDBACK\n{filter_text or last_error}"
        )

    record = {
        "example_id": candidate.example_id,
        "accepted": example is not None,
        "attempt_count": len(attempts),
        "attempts": attempts,
        "teacher_response": attempts[-1]["teacher_response"],
        "filter_response": attempts[-1]["filter_response"],
        "error": None if example is not None else last_error,
    }
    return DistillationResult(
        index=index,
        candidate=candidate,
        example=example,
        record=record,
        teacher_envelope=(
            json.dumps(teacher_envelopes, ensure_ascii=False, indent=2)
            if teacher_envelopes
            else None
        ),
        filter_envelope=(
            json.dumps(filter_envelopes, ensure_ascii=False, indent=2)
            if filter_envelopes
            else None
        ),
    )


def _call_openai(client: Any, model: str, prompt: str) -> tuple[str, str]:
    response = client.responses.create(model=model, input=prompt, max_output_tokens=4096)
    return response.output_text, response.model_dump_json(indent=2)


def _render_template(name: str, replacements: dict[str, str]) -> str:
    template = (PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8")
    for marker, value in replacements.items():
        if template.count(marker) != 1:
            raise ValueError(f"{name}.txt must contain exactly one {marker} marker")
        template = template.replace(marker, value)
    return template

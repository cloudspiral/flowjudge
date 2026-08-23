from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import Field

from .dialam import DEFAULT_DIALOGUES_PATH, DEFAULT_SOURCE_DIR, load_canonical_maps
from .patch_ceiling import build_patch_prompt
from .patch_data import (
    DEFAULT_DIAGNOSTIC_EVAL_EXAMPLES_PATH,
    DEFAULT_EVAL_SUMMARY_PATH,
    PROJECT_ROOT,
    PatchExample,
    PatchSplit,
    _materialize_candidate,
    build_update_candidates,
    load_jsonl,
)
from .schemas import StrictModel


TRAINING_SIZES = (256, 512, 1024, 2048)
TRAINING_SEED = "dialam-qt30-qlora-v1"
DEFAULT_TRAINING_DIR = PROJECT_ROOT / "data" / "dialam" / "training"
CATEGORY_WEIGHTS = {
    "NONE": 0.40,
    "SUPPORT": 0.20,
    "ATTACK": 0.15,
    "REPHRASE": 0.20,
    "MIXED": 0.05,
}


class TrainingMessage(StrictModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class DialAMTrainingRow(StrictModel):
    schema_version: Literal["dialam_qlora_chat_v1"]
    example_id: str
    update_id: str
    dialogue_id: str
    category: Literal["NONE", "SUPPORT", "ATTACK", "REPHRASE", "MIXED"]
    messages: list[TrainingMessage] = Field(min_length=2, max_length=2)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selection_key(example: PatchExample) -> str:
    return hashlib.sha256(f"{TRAINING_SEED}|{example.example_id}".encode()).hexdigest()


def _category(example: PatchExample) -> str:
    labels = {edge.type.value for edge in example.gold_patch.relations}
    if not labels:
        return "NONE"
    if len(labels) == 1:
        return next(iter(labels))
    return "MIXED"


def _weighted_nested_order(examples: list[PatchExample], target: int) -> list[PatchExample]:
    queues: dict[str, list[PatchExample]] = defaultdict(list)
    for example in examples:
        queues[_category(example)].append(example)
    for queue in queues.values():
        queue.sort(key=_selection_key)
    if len(examples) < target:
        raise ValueError(f"only {len(examples)} train blocks are available; need {target}")

    indexes = {category: 0 for category in CATEGORY_WEIGHTS}
    counts = Counter()
    selected: list[PatchExample] = []
    while len(selected) < target:
        available = [
            category
            for category in CATEGORY_WEIGHTS
            if indexes[category] < len(queues.get(category, []))
        ]
        if not available:
            raise ValueError(f"training pool exhausted at {len(selected)} examples")
        next_size = len(selected) + 1
        category = max(
            available,
            key=lambda item: (
                CATEGORY_WEIGHTS[item] * next_size - counts[item],
                CATEGORY_WEIGHTS[item],
                item,
            ),
        )
        selected.append(queues[category][indexes[category]])
        indexes[category] += 1
        counts[category] += 1
    return selected


def _training_row(example: PatchExample) -> DialAMTrainingRow:
    return DialAMTrainingRow(
        schema_version="dialam_qlora_chat_v1",
        example_id=example.example_id,
        update_id=example.update_id,
        dialogue_id=example.dialogue_id,
        category=_category(example),
        messages=[
            TrainingMessage(
                role="user",
                content=build_patch_prompt("zero_shot", example, []),
            ),
            TrainingMessage(
                role="assistant",
                content=example.gold_patch.model_dump_json(),
            ),
        ],
    )


def _write_jsonl(path: Path, rows: list[DialAMTrainingRow]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row.model_dump_json() + "\n")


def _slice_stats(rows: list[DialAMTrainingRow]) -> dict[str, object]:
    relation_counts = Counter()
    for row in rows:
        output = json.loads(row.messages[1].content)
        relation_counts.update(edge["type"] for edge in output["relations"])
    return {
        "examples": len(rows),
        "updates": len({row.update_id for row in rows}),
        "parent_episodes": len({row.dialogue_id for row in rows}),
        "category_counts": dict(sorted(Counter(row.category for row in rows).items())),
        "relation_counts": dict(sorted(relation_counts.items())),
        "all_negative_examples": sum(row.category == "NONE" for row in rows),
    }


def build_dialam_training_corpus(
    *,
    source_dir: Path = DEFAULT_SOURCE_DIR,
    dialogues_path: Path = DEFAULT_DIALOGUES_PATH,
    eval_summary_path: Path = DEFAULT_EVAL_SUMMARY_PATH,
    output_dir: Path = DEFAULT_TRAINING_DIR,
) -> dict[str, object]:
    eval_summary = json.loads(eval_summary_path.read_text(encoding="utf-8"))
    heldout_dialogues = set(eval_summary["heldout_dialogue_ids"])
    maps = load_canonical_maps(source_dir, dialogues_path)
    candidates = [
        candidate
        for candidate in build_update_candidates(maps)
        if candidate.canonical_map.dialogue_id not in heldout_dialogues
    ]
    examples = [
        example
        for candidate in candidates
        for example in _materialize_candidate(candidate, PatchSplit.TRAIN)
    ]
    if {example.dialogue_id for example in examples} & heldout_dialogues:
        raise ValueError("held-out parent episode leaked into the training pool")
    if len({example.example_id for example in examples}) != len(examples):
        raise ValueError("duplicate training block IDs found")

    ordered = _weighted_nested_order(examples, max(TRAINING_SIZES))
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, dict[str, object]] = {}
    previous_ids: list[str] = []
    for size in TRAINING_SIZES:
        rows = [_training_row(example) for example in ordered[:size]]
        ids = [row.example_id for row in rows]
        if ids[: len(previous_ids)] != previous_ids:
            raise ValueError(f"n{size} is not a strict prefix of the prior slice")
        previous_ids = ids
        path = output_dir / f"dialam_n{size}.jsonl"
        _write_jsonl(path, rows)
        stats = _slice_stats(rows)
        category_counts = stats["category_counts"]
        assert isinstance(category_counts, dict)
        for required in ("NONE", "SUPPORT", "ATTACK", "REPHRASE"):
            if int(category_counts.get(required, 0)) < max(16, size // 12):
                raise ValueError(f"n{size} lacks meaningful {required} coverage")
        if int(stats["all_negative_examples"]) < size * 0.30:
            raise ValueError(f"n{size} lacks substantial all-negative coverage")
        outputs[str(size)] = {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "sha256": file_sha256(path),
            **stats,
        }

    schema_path = output_dir / "training_example.schema.json"
    schema_path.write_text(
        json.dumps(DialAMTrainingRow.model_json_schema(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    frozen_eval_inputs_path = output_dir / "frozen_eval_inputs.jsonl"
    frozen_eval_examples = load_jsonl(
        DEFAULT_DIAGNOSTIC_EVAL_EXAMPLES_PATH,
        PatchExample,
    )
    with frozen_eval_inputs_path.open("w", encoding="utf-8") as handle:
        for example in frozen_eval_examples:
            handle.write(
                json.dumps(
                    {
                        "example_id": example.example_id,
                        "update_id": example.update_id,
                        "prompt": build_patch_prompt("zero_shot", example, []),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    manifest: dict[str, object] = {
        "schema_version": "dialam_training_manifest_v1",
        "created_at": datetime.now(UTC).isoformat(),
        "selection_seed": TRAINING_SEED,
        "nested_sizes": list(TRAINING_SIZES),
        "max_examples": max(TRAINING_SIZES),
        "base_model": "Qwen/Qwen3-0.6B",
        "prompt": "prompts/dialam/zero_shot.txt",
        "split_unit": "original_parent_episode",
        "heldout_parent_episode_ids": sorted(heldout_dialogues),
        "training_parent_episode_ids": sorted({item.dialogue_id for item in ordered}),
        "dialogue_leakage": False,
        "category_target_weights": CATEGORY_WEIGHTS,
        "selection_policy": (
            "deterministic weighted interleave without replacement; each smaller slice is "
            "the exact prefix of n2048"
        ),
        "source": {
            "archive_sha256": file_sha256(PROJECT_ROOT / "data/source/dialam_qt30/dataset.zip"),
            "dialogues_sha256": file_sha256(dialogues_path),
            "eval_summary_sha256": file_sha256(eval_summary_path),
        },
        "available_training_blocks": len(examples),
        "available_category_counts": dict(sorted(Counter(_category(item) for item in examples).items())),
        "slices": outputs,
        "schema": {
            "path": str(schema_path.relative_to(PROJECT_ROOT)),
            "sha256": file_sha256(schema_path),
        },
        "frozen_eval_inputs": {
            "path": str(frozen_eval_inputs_path.relative_to(PROJECT_ROOT)),
            "examples": len(frozen_eval_examples),
            "sha256": file_sha256(frozen_eval_inputs_path),
            "source_eval_sha256": file_sha256(DEFAULT_DIAGNOSTIC_EVAL_EXAMPLES_PATH),
        },
        "privacy": (
            "The JSONL slices contain transformed QT30 text and remain private/local. "
            "This text-free manifest and schema may be published."
        ),
    }
    manifest_path = output_dir / "training_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest

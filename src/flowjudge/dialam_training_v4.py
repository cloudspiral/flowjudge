from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Literal

from pydantic import Field

from .dialam import DEFAULT_DIALOGUES_PATH, DEFAULT_SOURCE_DIR, load_canonical_maps
from .dialam_training import DEFAULT_TRAINING_DIR, TrainingMessage, _category, file_sha256
from .dialam_training_v3 import (
    V3_DEV_EXAMPLES_FILENAME,
    V3_DEV_INPUTS_FILENAME,
    V3_MANIFEST_FILENAME,
    V3_OUTPUT_FILENAME,
    PairMetadata,
    SelectedPair,
    _dialogue_split,
    _negative_metadata,
    _seeded_hash,
    select_paired_examples,
)
from .patch_ceiling import FROZEN_JUDGE_RUBRIC_SHA256, build_patch_prompt
from .patch_data import (
    DEFAULT_DIAGNOSTIC_EVAL_EXAMPLES_PATH,
    DEFAULT_EVAL_SUMMARY_PATH,
    PROJECT_ROOT,
    PatchExample,
    PatchSplit,
    _materialize_candidate,
    build_update_candidates,
)
from .schemas import StrictModel


V4_TRAINING_SIZE = 8192
V4_FOUNDATION_SIZE = 4096
V4_SUPPLEMENT_SIZE = V4_TRAINING_SIZE - V4_FOUNDATION_SIZE
V4_TRAINING_SEED = "dialam-qt30-qlora-v4-class-balanced-rehearsal"
V4_FINAL_CATEGORY_COUNTS = {
    "NONE": 4096,
    "SUPPORT": 1344,
    "ATTACK": 1344,
    "REPHRASE": 1344,
    "MIXED": 64,
}
V4_SUPPLEMENTAL_POSITIVE_COUNTS = {
    "SUPPORT": 430,
    "ATTACK": 1155,
    "REPHRASE": 429,
    "MIXED": 34,
}
V4_SUPPLEMENTAL_NONE_COUNT = 2048
V4_OUTPUT_FILENAME = "dialam_v4_n8192.jsonl"
V4_MANIFEST_FILENAME = "training_v4_manifest.json"
V4_SCHEMA_FILENAME = "training_example_v4.schema.json"


class V4NegativeMetadata(StrictModel):
    kind: Literal["direct_edge_elsewhere", "all_negative_update"]
    max_shared_content_tokens: int = Field(ge=0)
    max_overlap_coefficient: float = Field(ge=0.0, le=1.0)
    max_jaccard_similarity: float = Field(ge=0.0, le=1.0)
    closest_turn_gap: int = Field(gt=0)


class DialAMTrainingRowV4(StrictModel):
    schema_version: Literal["dialam_qlora_chat_v4"]
    training_row_id: str
    example_id: str
    update_id: str
    dialogue_id: str
    category: Literal["NONE", "SUPPORT", "ATTACK", "REPHRASE", "MIXED"]
    source_stage: Literal["V3_FOUNDATION", "V4_CLASS_BALANCE"]
    sampling_role: Literal["POSITIVE", "NONE"]
    repetition_index: int = Field(ge=0)
    is_repeated_copy: bool
    source_pair_id: str | None = None
    negative_metadata: V4NegativeMetadata | None = None
    assistant_loss_weight: Literal[1.0]
    messages: list[TrainingMessage] = Field(min_length=2, max_length=2)


@dataclass(frozen=True)
class SelectedV4Example:
    example: PatchExample
    source_stage: Literal["V3_FOUNDATION", "V4_CLASS_BALANCE"]
    sampling_role: Literal["POSITIVE", "NONE"]
    source_pair_id: str | None = None
    negative_metadata: V4NegativeMetadata | None = None


def _v4_negative_metadata(
    example: PatchExample,
    *,
    direct_edge_elsewhere: bool,
) -> V4NegativeMetadata:
    base: PairMetadata = _negative_metadata(example)
    return V4NegativeMetadata(
        kind=("direct_edge_elsewhere" if direct_edge_elsewhere else "all_negative_update"),
        max_shared_content_tokens=base.max_shared_content_tokens,
        max_overlap_coefficient=base.max_overlap_coefficient,
        max_jaccard_similarity=base.max_jaccard_similarity,
        closest_turn_gap=base.closest_turn_gap,
    )


def _positive_rank(example: PatchExample) -> tuple[str, str]:
    return (_seeded_hash(V4_TRAINING_SEED, example.example_id), example.example_id)


def _negative_rank(
    example: PatchExample,
    metadata: V4NegativeMetadata,
) -> tuple[object, ...]:
    return (
        metadata.kind != "direct_edge_elsewhere",
        -int(metadata.max_shared_content_tokens >= 2),
        -metadata.max_overlap_coefficient,
        -metadata.max_shared_content_tokens,
        -metadata.max_jaccard_similarity,
        metadata.closest_turn_gap,
        _seeded_hash(V4_TRAINING_SEED, example.example_id),
    )


def select_v4_supplement(
    examples: list[PatchExample],
    foundation_pairs: list[SelectedPair],
    *,
    positive_counts: dict[str, int] = V4_SUPPLEMENTAL_POSITIVE_COUNTS,
    none_count: int = V4_SUPPLEMENTAL_NONE_COUNT,
) -> list[SelectedV4Example]:
    foundation_positive_ids = {pair.positive.example_id for pair in foundation_pairs}
    foundation_none_ids = {pair.negative.example_id for pair in foundation_pairs}
    selected: list[SelectedV4Example] = []

    by_category: dict[str, list[PatchExample]] = defaultdict(list)
    for example in examples:
        by_category[_category(example)].append(example)
    for category in by_category:
        by_category[category] = sorted(by_category[category], key=_positive_rank)

    for category in ("ATTACK", "MIXED", "REPHRASE", "SUPPORT"):
        target = positive_counts.get(category, 0)
        pool = by_category.get(category, [])
        if target and not pool:
            raise ValueError(f"v4 has no {category} examples for a target of {target}")
        unseen = [item for item in pool if item.example_id not in foundation_positive_ids]
        chosen = unseen[:target]
        cycle_index = 0
        while len(chosen) < target:
            chosen.append(pool[cycle_index % len(pool)])
            cycle_index += 1
        selected.extend(
            SelectedV4Example(
                example=item,
                source_stage="V4_CLASS_BALANCE",
                sampling_role="POSITIVE",
            )
            for item in chosen
        )

    positive_updates = {
        item.update_id for item in examples if _category(item) != "NONE"
    }
    negative_pool: list[tuple[PatchExample, V4NegativeMetadata]] = []
    for example in by_category.get("NONE", []):
        if example.example_id in foundation_none_ids:
            continue
        metadata = _v4_negative_metadata(
            example,
            direct_edge_elsewhere=example.update_id in positive_updates,
        )
        negative_pool.append((example, metadata))
    negative_pool.sort(key=lambda item: _negative_rank(item[0], item[1]))
    if len(negative_pool) < none_count:
        raise ValueError(
            f"v4 has only {len(negative_pool)} unused NONE examples; need {none_count}"
        )
    selected.extend(
        SelectedV4Example(
            example=example,
            source_stage="V4_CLASS_BALANCE",
            sampling_role="NONE",
            negative_metadata=metadata,
        )
        for example, metadata in negative_pool[:none_count]
    )

    actual_positive = Counter(
        _category(item.example)
        for item in selected
        if item.sampling_role == "POSITIVE"
    )
    if actual_positive != Counter(positive_counts):
        raise ValueError(f"v4 supplemental positive mix is invalid: {dict(actual_positive)}")
    if sum(item.sampling_role == "NONE" for item in selected) != none_count:
        raise ValueError("v4 supplemental NONE count is invalid")
    return selected


def _foundation_examples(pairs: list[SelectedPair]) -> list[SelectedV4Example]:
    return [
        row
        for pair in pairs
        for row in (
            SelectedV4Example(
                example=pair.positive,
                source_stage="V3_FOUNDATION",
                sampling_role="POSITIVE",
                source_pair_id=pair.pair_id,
            ),
            SelectedV4Example(
                example=pair.negative,
                source_stage="V3_FOUNDATION",
                sampling_role="NONE",
                source_pair_id=pair.pair_id,
                negative_metadata=V4NegativeMetadata(
                    kind="direct_edge_elsewhere",
                    max_shared_content_tokens=pair.metadata.max_shared_content_tokens,
                    max_overlap_coefficient=pair.metadata.max_overlap_coefficient,
                    max_jaccard_similarity=pair.metadata.max_jaccard_similarity,
                    closest_turn_gap=pair.metadata.closest_turn_gap,
                ),
            ),
        )
    ]


def materialize_v4_rows(
    selections: list[SelectedV4Example],
) -> list[DialAMTrainingRowV4]:
    occurrences: Counter[str] = Counter()
    rows: list[DialAMTrainingRowV4] = []
    for selection in selections:
        example = selection.example
        repetition_index = occurrences[example.example_id]
        occurrences[example.example_id] += 1
        training_row_id = "v4row-" + _seeded_hash(
            V4_TRAINING_SEED,
            f"{example.example_id}|{selection.source_stage}|{repetition_index}",
        )[:24]
        rows.append(
            DialAMTrainingRowV4(
                schema_version="dialam_qlora_chat_v4",
                training_row_id=training_row_id,
                example_id=example.example_id,
                update_id=example.update_id,
                dialogue_id=example.dialogue_id,
                category=_category(example),
                source_stage=selection.source_stage,
                sampling_role=selection.sampling_role,
                repetition_index=repetition_index,
                is_repeated_copy=repetition_index > 0,
                source_pair_id=selection.source_pair_id,
                negative_metadata=selection.negative_metadata,
                assistant_loss_weight=1.0,
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
        )
    return sorted(
        rows,
        key=lambda item: _seeded_hash(V4_TRAINING_SEED, item.training_row_id),
    )


def _write_jsonl(path: Path, rows: list[StrictModel]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row.model_dump_json() + "\n")


def _relation_counts(rows: list[DialAMTrainingRowV4]) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        payload = json.loads(row.messages[1].content)
        counts.update(relation["type"] for relation in payload["relations"])
    return dict(sorted(counts.items()))


def _distribution(values: list[int | float]) -> dict[str, int | float]:
    return {"minimum": min(values), "mean": mean(values), "maximum": max(values)}


def build_dialam_v4_training_corpus(
    *,
    source_dir: Path = DEFAULT_SOURCE_DIR,
    dialogues_path: Path = DEFAULT_DIALOGUES_PATH,
    eval_summary_path: Path = DEFAULT_EVAL_SUMMARY_PATH,
    output_dir: Path = DEFAULT_TRAINING_DIR,
) -> dict[str, object]:
    eval_summary = json.loads(eval_summary_path.read_text(encoding="utf-8"))
    frozen_dialogues = set(eval_summary["heldout_dialogue_ids"])
    maps = load_canonical_maps(source_dir, dialogues_path)
    candidates = build_update_candidates(maps)
    nonfrozen_candidates = [
        item for item in candidates if item.canonical_map.dialogue_id not in frozen_dialogues
    ]
    available_dialogues = {
        item.canonical_map.dialogue_id for item in nonfrozen_candidates
    }
    training_dialogues, development_dialogues = _dialogue_split(available_dialogues)
    if training_dialogues & development_dialogues or training_dialogues & frozen_dialogues:
        raise ValueError("v4 parent-episode splits overlap")

    training_examples = [
        example
        for candidate in nonfrozen_candidates
        if candidate.canonical_map.dialogue_id in training_dialogues
        for example in _materialize_candidate(candidate, PatchSplit.TRAIN)
    ]
    foundation_pairs = select_paired_examples(training_examples)
    foundation = _foundation_examples(foundation_pairs)
    supplement = select_v4_supplement(training_examples, foundation_pairs)
    rows = materialize_v4_rows([*foundation, *supplement])

    if len(rows) != V4_TRAINING_SIZE:
        raise ValueError(f"v4 must contain {V4_TRAINING_SIZE} rows")
    if len({row.training_row_id for row in rows}) != len(rows):
        raise ValueError("v4 training-row IDs are not unique")
    if Counter(row.category for row in rows) != Counter(V4_FINAL_CATEGORY_COUNTS):
        raise ValueError("v4 final category mix is invalid")
    if Counter(row.source_stage for row in rows) != Counter(
        {"V3_FOUNDATION": V4_FOUNDATION_SIZE, "V4_CLASS_BALANCE": V4_SUPPLEMENT_SIZE}
    ):
        raise ValueError("v4 source-stage mix is invalid")
    if Counter(row.sampling_role for row in rows) != Counter(
        {"POSITIVE": V4_TRAINING_SIZE // 2, "NONE": V4_TRAINING_SIZE // 2}
    ):
        raise ValueError("v4 positive/NONE balance is invalid")
    if {row.dialogue_id for row in rows} - training_dialogues:
        raise ValueError("v4 contains a development or frozen parent episode")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / V4_OUTPUT_FILENAME
    _write_jsonl(output_path, rows)
    schema_path = output_dir / V4_SCHEMA_FILENAME
    schema_path.write_text(
        json.dumps(DialAMTrainingRowV4.model_json_schema(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    example_occurrences = Counter(row.example_id for row in rows)
    category_unique_examples = {
        category: len({row.example_id for row in rows if row.category == category})
        for category in V4_FINAL_CATEGORY_COUNTS
    }
    supplemental_none = [
        row
        for row in rows
        if row.source_stage == "V4_CLASS_BALANCE" and row.sampling_role == "NONE"
    ]
    hard_kinds = Counter(
        row.negative_metadata.kind
        for row in supplemental_none
        if row.negative_metadata is not None
    )
    shared_tokens = [
        row.negative_metadata.max_shared_content_tokens
        for row in supplemental_none
        if row.negative_metadata is not None
    ]
    positive_chars = sum(
        len(row.messages[1].content) for row in rows if row.sampling_role == "POSITIVE"
    )
    none_chars = sum(
        len(row.messages[1].content) for row in rows if row.sampling_role == "NONE"
    )
    manifest: dict[str, object] = {
        "schema_version": "dialam_training_manifest_v4",
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_version": "v4-class-balanced-rehearsal",
        "base_model": "Qwen/Qwen3-0.6B",
        "size": V4_TRAINING_SIZE,
        "selection_seed": V4_TRAINING_SEED,
        "split_unit": "original_parent_episode",
        "training_parent_episode_count": len(training_dialogues),
        "development_parent_episode_count": len(development_dialogues),
        "frozen_parent_episode_count": len(frozen_dialogues),
        "dialogue_leakage": False,
        "category_counts": dict(sorted(Counter(row.category for row in rows).items())),
        "category_unique_example_counts": dict(sorted(category_unique_examples.items())),
        "relation_counts": _relation_counts(rows),
        "source_stage_counts": dict(
            sorted(Counter(row.source_stage for row in rows).items())
        ),
        "sampling_role_counts": dict(
            sorted(Counter(row.sampling_role for row in rows).items())
        ),
        "repetition_policy": {
            "description": (
                "Retain every v3 foundation row; add every unused ATTACK block before "
                "deterministically cycling ATTACK examples to class balance. Other "
                "supplemental positive classes and every NONE supplement remain unique."
            ),
            "repeated_copy_count": sum(row.is_repeated_copy for row in rows),
            "unique_example_count": len(example_occurrences),
            "maximum_occurrences_of_one_example": max(example_occurrences.values()),
            "categories_with_repeated_examples": sorted(
                {
                    row.category
                    for row in rows
                    if example_occurrences[row.example_id] > 1
                }
            ),
        },
        "supplement_policy": {
            "positive_target_counts": V4_SUPPLEMENTAL_POSITIVE_COUNTS,
            "none_count": V4_SUPPLEMENTAL_NONE_COUNT,
            "none_kind_counts": dict(sorted(hard_kinds.items())),
            "none_shared_content_tokens": _distribution(shared_tokens),
            "semantic_candidate_retrieval": False,
            "description": (
                "Use all available positive-class diversity before deterministic ATTACK "
                "repetition. Exhaust unused no-edge blocks whose update has a direct edge "
                "elsewhere, then fill with the hardest unused all-negative blocks ranked "
                "only by within-block lexical overlap and chronology."
            ),
        },
        "loss_policy": {
            "name": "per_example_assistant_token_mean_then_batch_mean",
            "assistant_loss_weight_per_row": 1.0,
            "positive_to_none_row_ratio": 1.0,
            "unbalanced_positive_to_none_supervised_character_ratio": (
                positive_chars / none_chars
            ),
        },
        "development_gate": {
            "evaluation": "existing episode-disjoint v3 development set",
            "eval_examples_sha256": file_sha256(output_dir / V3_DEV_EXAMPLES_FILENAME),
            "eval_inputs_sha256": file_sha256(output_dir / V3_DEV_INPUTS_FILENAME),
            "v3_baseline": {
                "exact_patch_accuracy": 0.26666666666666666,
                "edge_f1": 0.2105263157894737,
                "attack_f1": 0.0,
                "false_edges_per_update": 0.3333333333333333,
                "none_scenarios_with_false_edges": 2,
            },
            "requirements_to_run_reused_frozen_benchmark": {
                "minimum_exact_patch_accuracy": 0.26666666666666666,
                "minimum_edge_f1": 0.25,
                "minimum_attack_f1": 0.1,
                "maximum_false_edges_per_update": 0.3333333333333333,
                "maximum_none_scenarios_with_false_edges": 2,
                "json_and_schema_validity": 1.0,
            },
        },
        "reused_frozen_benchmark": {
            "examples": 30,
            "eval_sha256": file_sha256(DEFAULT_DIAGNOSTIC_EVAL_EXAMPLES_PATH),
            "judge_rubric_sha256": FROZEN_JUDGE_RUBRIC_SHA256,
            "disclosure": (
                "The v3 frozen errors informed v4, so this set is now a reused comparison "
                "benchmark rather than an untouched model-selection holdout. The external "
                "staff-held-out set remains the final unbiased evaluation."
            ),
            "promotion_requirements": {
                "minimum_exact_patch_accuracy": 0.43333333333333335,
                "minimum_edge_f1": 0.4,
                "minimum_relation_macro_f1": 0.39,
                "minimum_attack_f1": 0.3,
                "maximum_false_edges_per_update": 0.3,
                "maximum_none_scenarios_with_false_edges": 1,
                "json_and_schema_validity": 1.0,
            },
        },
        "source": {
            "archive_sha256": file_sha256(PROJECT_ROOT / "data/source/dialam_qt30/dataset.zip"),
            "dialogues_sha256": file_sha256(dialogues_path),
            "eval_summary_sha256": file_sha256(eval_summary_path),
            "v3_training_jsonl_sha256": file_sha256(output_dir / V3_OUTPUT_FILENAME),
            "v3_training_manifest_sha256": file_sha256(output_dir / V3_MANIFEST_FILENAME),
        },
        "output": {
            "path": str(output_path.relative_to(PROJECT_ROOT)),
            "sha256": file_sha256(output_path),
        },
        "schema": {
            "path": str(schema_path.relative_to(PROJECT_ROOT)),
            "sha256": file_sha256(schema_path),
        },
        "privacy": (
            "The v4 JSONL contains transformed QT30 text and remains private/local. "
            "Only this text-free aggregate manifest, schema, and deterministic "
            "reconstruction code may be published."
        ),
    }
    manifest_path = output_dir / V4_MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Literal

from pydantic import Field

from .dialam import DEFAULT_DIALOGUES_PATH, DEFAULT_SOURCE_DIR, load_canonical_maps
from .dialam_training import (
    DEFAULT_TRAINING_DIR,
    TrainingMessage,
    _category,
    file_sha256,
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


V2_TRAINING_SIZE = 2048
V2_TRAINING_SEED = "dialam-qt30-qlora-v2-hard-negative"
V2_TARGET_COUNTS = {
    "NONE": 1024,
    "SUPPORT": 358,
    "ATTACK": 256,
    "REPHRASE": 307,
    "MIXED": 103,
}
V2_OUTPUT_FILENAME = "dialam_v2_n2048.jsonl"
V2_MANIFEST_FILENAME = "training_v2_manifest.json"
V2_SCHEMA_FILENAME = "training_example_v2.schema.json"

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_STOP_WORDS = {
    "about",
    "also",
    "and",
    "any",
    "are",
    "because",
    "been",
    "being",
    "but",
    "can",
    "could",
    "did",
    "does",
    "doing",
    "for",
    "from",
    "get",
    "got",
    "had",
    "has",
    "have",
    "here",
    "how",
    "into",
    "its",
    "just",
    "know",
    "like",
    "made",
    "make",
    "many",
    "may",
    "might",
    "more",
    "most",
    "much",
    "not",
    "one",
    "our",
    "really",
    "said",
    "say",
    "says",
    "should",
    "some",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "they",
    "think",
    "this",
    "two",
    "very",
    "was",
    "well",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "would",
    "yeah",
    "yes",
    "you",
    "your",
}


class HardNegativeMetadata(StrictModel):
    kind: Literal["positive_sibling_lexical_no_edge"]
    positive_sibling_block: Literal[True]
    max_shared_content_tokens: int = Field(ge=1)
    max_overlap_coefficient: float = Field(ge=0.0, le=1.0)
    max_jaccard_similarity: float = Field(ge=0.0, le=1.0)
    closest_turn_gap: int = Field(gt=0)


class DialAMTrainingRowV2(StrictModel):
    schema_version: Literal["dialam_qlora_chat_v2"]
    example_id: str
    update_id: str
    dialogue_id: str
    category: Literal["NONE", "SUPPORT", "ATTACK", "REPHRASE", "MIXED"]
    hard_negative: HardNegativeMetadata | None = None
    messages: list[TrainingMessage] = Field(min_length=2, max_length=2)


def _content_tokens(text: str) -> set[str]:
    return {
        token
        for token in _TOKEN_PATTERN.findall(text.lower())
        if len(token) >= 3 and token not in _STOP_WORDS
    }


def _hard_negative_metadata(
    example: PatchExample,
    *,
    positive_sibling_block: bool,
) -> HardNegativeMetadata | None:
    if _category(example) != "NONE" or not positive_sibling_block:
        return None
    new_tokens = _content_tokens(example.new_proposition.text)
    comparisons: list[tuple[int, float, float]] = []
    for earlier in example.earlier_propositions:
        earlier_tokens = _content_tokens(earlier.text)
        shared = len(new_tokens & earlier_tokens)
        overlap = shared / max(1, min(len(new_tokens), len(earlier_tokens)))
        jaccard = shared / max(1, len(new_tokens | earlier_tokens))
        comparisons.append((shared, overlap, jaccard))
    shared, overlap, jaccard = max(
        comparisons,
        key=lambda item: (item[0] >= 2, item[1], item[0], item[2]),
        default=(0, 0.0, 0.0),
    )
    if shared < 1:
        return None
    return HardNegativeMetadata(
        kind="positive_sibling_lexical_no_edge",
        positive_sibling_block=True,
        max_shared_content_tokens=shared,
        max_overlap_coefficient=overlap,
        max_jaccard_similarity=jaccard,
        closest_turn_gap=min(
            example.new_proposition.chronological_turn - earlier.chronological_turn
            for earlier in example.earlier_propositions
        ),
    )


def _selection_hash(example: PatchExample) -> str:
    return hashlib.sha256(
        f"{V2_TRAINING_SEED}|{example.example_id}".encode()
    ).hexdigest()


def _select_examples(examples: list[PatchExample]) -> list[tuple[PatchExample, HardNegativeMetadata | None]]:
    by_update: dict[str, list[PatchExample]] = defaultdict(list)
    for example in examples:
        by_update[example.update_id].append(example)

    queues: dict[str, list[tuple[PatchExample, HardNegativeMetadata | None]]] = defaultdict(list)
    for example in examples:
        category = _category(example)
        positive_sibling = any(
            _category(sibling) != "NONE" for sibling in by_update[example.update_id]
        )
        metadata = _hard_negative_metadata(
            example,
            positive_sibling_block=positive_sibling,
        )
        if category == "NONE":
            if metadata is not None:
                queues[category].append((example, metadata))
        else:
            queues[category].append((example, None))

    queues["NONE"].sort(
        key=lambda item: (
            -int(item[1].max_shared_content_tokens >= 2),  # type: ignore[union-attr]
            -item[1].max_overlap_coefficient,  # type: ignore[union-attr]
            -item[1].max_shared_content_tokens,  # type: ignore[union-attr]
            -item[1].max_jaccard_similarity,  # type: ignore[union-attr]
            item[1].closest_turn_gap,  # type: ignore[union-attr]
            _selection_hash(item[0]),
        )
    )
    for category in V2_TARGET_COUNTS:
        if category != "NONE":
            queues[category].sort(key=lambda item: _selection_hash(item[0]))
        if len(queues[category]) < V2_TARGET_COUNTS[category]:
            raise ValueError(
                f"only {len(queues[category])} eligible {category} examples; "
                f"need {V2_TARGET_COUNTS[category]}"
            )

    indexes = Counter()
    counts = Counter()
    selected: list[tuple[PatchExample, HardNegativeMetadata | None]] = []
    while len(selected) < V2_TRAINING_SIZE:
        next_size = len(selected) + 1
        available = [
            category
            for category, target in V2_TARGET_COUNTS.items()
            if counts[category] < target
        ]
        category = max(
            available,
            key=lambda item: (
                V2_TARGET_COUNTS[item] * next_size / V2_TRAINING_SIZE - counts[item],
                V2_TARGET_COUNTS[item],
                item,
            ),
        )
        selected.append(queues[category][indexes[category]])
        indexes[category] += 1
        counts[category] += 1
    if dict(counts) != V2_TARGET_COUNTS:
        raise ValueError(f"v2 target mix was not met: {dict(counts)}")
    return selected


def _training_row(
    example: PatchExample,
    hard_negative: HardNegativeMetadata | None,
) -> DialAMTrainingRowV2:
    return DialAMTrainingRowV2(
        schema_version="dialam_qlora_chat_v2",
        example_id=example.example_id,
        update_id=example.update_id,
        dialogue_id=example.dialogue_id,
        category=_category(example),
        hard_negative=hard_negative,
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


def _sha_or_missing(path: Path) -> str | None:
    return file_sha256(path) if path.exists() else None


def build_dialam_v2_training_corpus(
    *,
    source_dir: Path = DEFAULT_SOURCE_DIR,
    dialogues_path: Path = DEFAULT_DIALOGUES_PATH,
    eval_summary_path: Path = DEFAULT_EVAL_SUMMARY_PATH,
    output_dir: Path = DEFAULT_TRAINING_DIR,
) -> dict[str, object]:
    eval_summary = json.loads(eval_summary_path.read_text(encoding="utf-8"))
    heldout_dialogues = set(eval_summary["heldout_dialogue_ids"])
    maps = load_canonical_maps(source_dir, dialogues_path)
    examples = [
        example
        for candidate in build_update_candidates(maps)
        if candidate.canonical_map.dialogue_id not in heldout_dialogues
        for example in _materialize_candidate(candidate, PatchSplit.TRAIN)
    ]
    if {example.dialogue_id for example in examples} & heldout_dialogues:
        raise ValueError("held-out parent episode leaked into the v2 training pool")
    if len({example.example_id for example in examples}) != len(examples):
        raise ValueError("duplicate v2 training block IDs found")

    selected = _select_examples(examples)
    rows = [_training_row(example, metadata) for example, metadata in selected]
    selected_ids = [row.example_id for row in rows]
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("v2 selection duplicated a training block")
    selected_dialogues = {row.dialogue_id for row in rows}
    if selected_dialogues & heldout_dialogues:
        raise ValueError("held-out parent episode leaked into the v2 slice")

    hard_negatives = [row.hard_negative for row in rows if row.hard_negative is not None]
    if len(hard_negatives) != V2_TARGET_COUNTS["NONE"]:
        raise ValueError("every selected NONE row must be an eligible hard negative")
    if any(not item.positive_sibling_block for item in hard_negatives):
        raise ValueError("v2 contains a NONE row without a positive sibling block")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / V2_OUTPUT_FILENAME
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row.model_dump_json() + "\n")
    schema_path = output_dir / V2_SCHEMA_FILENAME
    schema_path.write_text(
        json.dumps(DialAMTrainingRowV2.model_json_schema(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    relation_counts = Counter()
    for row in rows:
        relation_counts.update(
            edge["type"] for edge in json.loads(row.messages[1].content)["relations"]
        )
    overlap_values = [item.max_overlap_coefficient for item in hard_negatives]
    shared_values = [item.max_shared_content_tokens for item in hard_negatives]
    v1_manifest_path = output_dir / "training_manifest.json"
    v1_manifest = json.loads(v1_manifest_path.read_text(encoding="utf-8"))
    v1_n2048_none_count = int(
        v1_manifest["slices"]["2048"]["category_counts"]["NONE"]
    )
    frozen_eval_path = DEFAULT_DIAGNOSTIC_EVAL_EXAMPLES_PATH
    manifest: dict[str, object] = {
        "schema_version": "dialam_training_manifest_v2",
        "dataset_version": "v2-hard-negative",
        "created_at": datetime.now(UTC).isoformat(),
        "selection_seed": V2_TRAINING_SEED,
        "size": V2_TRAINING_SIZE,
        "base_model": "Qwen/Qwen3-0.6B",
        "prompt": "prompts/dialam/zero_shot.txt",
        "split_unit": "original_parent_episode",
        "heldout_parent_episode_count": len(heldout_dialogues),
        "training_parent_episode_count": len(selected_dialogues),
        "dialogue_leakage": False,
        "target_category_counts": V2_TARGET_COUNTS,
        "actual_category_counts": dict(sorted(Counter(row.category for row in rows).items())),
        "relation_counts": dict(sorted(relation_counts.items())),
        "v1_n2048_none_count": v1_n2048_none_count,
        "v2_n2048_none_count": V2_TARGET_COUNTS["NONE"],
        "hard_negative_policy": {
            "kind": "positive_sibling_lexical_no_edge",
            "description": (
                "Select an all-negative fixed block only when the same update has a direct "
                "gold edge in a different block, then rank without embeddings by content-word "
                "overlap, shared-token count, Jaccard similarity, recency, and seeded hash."
            ),
            "semantic_candidate_retrieval": False,
            "duplicates": False,
            "selected_count": len(hard_negatives),
            "positive_sibling_count": sum(item.positive_sibling_block for item in hard_negatives),
            "with_two_or_more_shared_content_tokens": sum(value >= 2 for value in shared_values),
            "overlap_coefficient": {
                "minimum": min(overlap_values),
                "mean": mean(overlap_values),
                "maximum": max(overlap_values),
            },
            "shared_content_tokens": {
                "minimum": min(shared_values),
                "mean": mean(shared_values),
                "maximum": max(shared_values),
            },
        },
        "selection_policy": (
            "deterministic weighted interleave without replacement; positives use a seeded "
            "hash and NONE uses the declared hard-negative ranking"
        ),
        "source": {
            "archive_sha256": file_sha256(PROJECT_ROOT / "data/source/dialam_qt30/dataset.zip"),
            "dialogues_sha256": file_sha256(dialogues_path),
            "eval_summary_sha256": file_sha256(eval_summary_path),
            "v1_training_manifest_sha256": _sha_or_missing(v1_manifest_path),
        },
        "frozen_evaluation": {
            "examples": 30,
            "eval_sha256": file_sha256(frozen_eval_path),
            "judge_rubric_sha256": FROZEN_JUDGE_RUBRIC_SHA256,
            "unchanged_from_v1": True,
        },
        "preregistered_material_improvement": {
            "baseline": "v1/n2048",
            "minimum_absolute_edge_f1_gain": 0.05,
            "false_edges_per_update_must_not_increase": True,
            "json_and_schema_validity_must_remain": 1.0,
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
            "The v2 JSONL contains transformed QT30 text and remains private/local. This "
            "text-free manifest, aggregate statistics, schema, and reconstruction code may be published."
        ),
    }
    manifest_path = output_dir / V2_MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest

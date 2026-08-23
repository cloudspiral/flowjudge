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

from .dialam import DEFAULT_DIALOGUES_PATH, DEFAULT_SOURCE_DIR, NormalizedRelationLabel, load_canonical_maps
from .dialam_training import DEFAULT_TRAINING_DIR, TrainingMessage, _category, file_sha256
from .dialam_training_v2 import _content_tokens
from .patch_ceiling import FROZEN_JUDGE_RUBRIC_SHA256, build_patch_prompt
from .patch_data import (
    DEFAULT_DIAGNOSTIC_EVAL_EXAMPLES_PATH,
    DEFAULT_EVAL_SUMMARY_PATH,
    PROJECT_ROOT,
    PatchExample,
    PatchSplit,
    UpdateCandidate,
    _materialize_candidate,
    build_update_candidates,
)
from .schemas import StrictModel


V3_TRAINING_SIZE = 4096
V3_PAIR_COUNT = V3_TRAINING_SIZE // 2
V3_TRAINING_SEED = "dialam-qt30-qlora-v3-paired-balanced"
V3_DEV_SEED = "dialam-qt30-qlora-v3-development"
V3_DEV_EPISODE_COUNT = 4
V3_DEV_SCENARIO_COUNT = 30
V3_POSITIVE_TARGET_COUNTS = {
    "SUPPORT": 914,
    "ATTACK": 189,
    "REPHRASE": 915,
    "MIXED": 30,
}
V3_OUTPUT_FILENAME = "dialam_v3_n4096.jsonl"
V3_MANIFEST_FILENAME = "training_v3_manifest.json"
V3_SCHEMA_FILENAME = "training_example_v3.schema.json"
V3_DEV_EXAMPLES_FILENAME = "v3_dev_eval_examples.jsonl"
V3_DEV_INPUTS_FILENAME = "v3_dev_eval_inputs.jsonl"


class PairMetadata(StrictModel):
    kind: Literal["same_update_direct_edge_elsewhere"]
    max_shared_content_tokens: int = Field(ge=0)
    max_overlap_coefficient: float = Field(ge=0.0, le=1.0)
    max_jaccard_similarity: float = Field(ge=0.0, le=1.0)
    closest_turn_gap: int = Field(gt=0)


class DialAMTrainingRowV3(StrictModel):
    schema_version: Literal["dialam_qlora_chat_v3"]
    example_id: str
    update_id: str
    dialogue_id: str
    category: Literal["NONE", "SUPPORT", "ATTACK", "REPHRASE", "MIXED"]
    pair_id: str
    pair_role: Literal["POSITIVE", "NONE"]
    paired_example_id: str
    assistant_loss_weight: Literal[1.0]
    pair_metadata: PairMetadata
    messages: list[TrainingMessage] = Field(min_length=2, max_length=2)


@dataclass(frozen=True)
class SelectedPair:
    pair_id: str
    positive: PatchExample
    negative: PatchExample
    metadata: PairMetadata


@dataclass(frozen=True)
class _PairCandidate:
    positive: PatchExample
    negative: PatchExample
    metadata: PairMetadata


def _seeded_hash(seed: str, value: str) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode()).hexdigest()


def _dialogue_split(
    dialogue_ids: set[str],
) -> tuple[set[str], set[str]]:
    if len(dialogue_ids) <= V3_DEV_EPISODE_COUNT:
        raise ValueError("not enough parent episodes for a v3 train/development split")
    ordered = sorted(
        dialogue_ids,
        key=lambda item: _seeded_hash(V3_DEV_SEED, item),
    )
    development = set(ordered[:V3_DEV_EPISODE_COUNT])
    return dialogue_ids - development, development


def _negative_metadata(example: PatchExample) -> PairMetadata:
    if _category(example) != "NONE":
        raise ValueError("paired negative metadata requires a NONE block")
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
    return PairMetadata(
        kind="same_update_direct_edge_elsewhere",
        max_shared_content_tokens=shared,
        max_overlap_coefficient=overlap,
        max_jaccard_similarity=jaccard,
        closest_turn_gap=min(
            example.new_proposition.chronological_turn - earlier.chronological_turn
            for earlier in example.earlier_propositions
        ),
    )


def _negative_rank(example: PatchExample, metadata: PairMetadata) -> tuple[object, ...]:
    return (
        -int(metadata.max_shared_content_tokens >= 2),
        -metadata.max_overlap_coefficient,
        -metadata.max_shared_content_tokens,
        -metadata.max_jaccard_similarity,
        metadata.closest_turn_gap,
        _seeded_hash(V3_TRAINING_SEED, example.example_id),
    )


def _candidate_rank(candidate: _PairCandidate) -> tuple[object, ...]:
    return (
        *_negative_rank(candidate.negative, candidate.metadata),
        _seeded_hash(
            V3_TRAINING_SEED,
            f"{candidate.positive.example_id}|{candidate.negative.example_id}",
        ),
    )


def _pair_candidates(examples: list[PatchExample]) -> dict[str, list[_PairCandidate]]:
    by_update: dict[str, list[PatchExample]] = defaultdict(list)
    for example in examples:
        by_update[example.update_id].append(example)

    queues: dict[str, list[_PairCandidate]] = defaultdict(list)
    for update_examples in by_update.values():
        positives = [item for item in update_examples if _category(item) != "NONE"]
        negatives = [item for item in update_examples if _category(item) == "NONE"]
        if not positives or not negatives:
            continue
        ranked_negatives = sorted(
            ((item, _negative_metadata(item)) for item in negatives),
            key=lambda item: _negative_rank(item[0], item[1]),
        )
        negative, metadata = ranked_negatives[0]
        by_category: dict[str, list[PatchExample]] = defaultdict(list)
        for positive in positives:
            by_category[_category(positive)].append(positive)
        for category, category_positives in by_category.items():
            positive = min(
                category_positives,
                key=lambda item: _seeded_hash(V3_TRAINING_SEED, item.example_id),
            )
            queues[category].append(
                _PairCandidate(
                    positive=positive,
                    negative=negative,
                    metadata=metadata,
                )
            )
    for queue in queues.values():
        queue.sort(key=_candidate_rank)
    return queues


def select_paired_examples(
    examples: list[PatchExample],
    *,
    target_counts: dict[str, int] = V3_POSITIVE_TARGET_COUNTS,
) -> list[SelectedPair]:
    if sum(target_counts.values()) * 2 != V3_TRAINING_SIZE and target_counts is V3_POSITIVE_TARGET_COUNTS:
        raise ValueError("v3 target counts do not sum to the registered pair count")
    queues = _pair_candidates(examples)
    selected_candidates: list[_PairCandidate] = []
    used_updates: set[str] = set()
    # Rare labels are exhausted first. The remaining SUPPORT/REPHRASE queues
    # have ample slack, so multi-label updates cannot crowd out ATTACK/MIXED.
    selection_order = ["ATTACK", "MIXED", "REPHRASE", "SUPPORT"]
    for category in selection_order:
        target = target_counts.get(category, 0)
        eligible = [
            item for item in queues.get(category, []) if item.positive.update_id not in used_updates
        ]
        if len(eligible) < target:
            raise ValueError(
                f"only {len(eligible)} distinct-update paired {category} examples; need {target}"
            )
        chosen = eligible[:target]
        selected_candidates.extend(chosen)
        used_updates.update(item.positive.update_id for item in chosen)

    actual = Counter(_category(item.positive) for item in selected_candidates)
    if actual != Counter(target_counts):
        raise ValueError(f"v3 positive target mix was not met: {dict(actual)}")
    if len(selected_candidates) != len(used_updates):
        raise ValueError("v3 selected more than one pair from an update")

    pairs = [
        SelectedPair(
            pair_id="pair-"
            + _seeded_hash(
                V3_TRAINING_SEED,
                f"{item.positive.example_id}|{item.negative.example_id}",
            )[:20],
            positive=item.positive,
            negative=item.negative,
            metadata=item.metadata,
        )
        for item in selected_candidates
    ]
    return sorted(pairs, key=lambda item: item.pair_id)


def _training_row(pair: SelectedPair, role: Literal["POSITIVE", "NONE"]) -> DialAMTrainingRowV3:
    example = pair.positive if role == "POSITIVE" else pair.negative
    paired = pair.negative if role == "POSITIVE" else pair.positive
    return DialAMTrainingRowV3(
        schema_version="dialam_qlora_chat_v3",
        example_id=example.example_id,
        update_id=example.update_id,
        dialogue_id=example.dialogue_id,
        category=_category(example),
        pair_id=pair.pair_id,
        pair_role=role,
        paired_example_id=paired.example_id,
        assistant_loss_weight=1.0,
        pair_metadata=pair.metadata,
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


def _select_dev_candidates(
    candidates: list[UpdateCandidate],
    development_dialogues: set[str],
) -> list[UpdateCandidate]:
    eligible = [
        item
        for item in candidates
        if item.canonical_map.dialogue_id in development_dialogues and item.block_count == 1
    ]
    selected: list[UpdateCandidate] = []
    used: set[str] = set()

    negatives_by_dialogue: dict[str, list[UpdateCandidate]] = defaultdict(list)
    for candidate in eligible:
        if not candidate.is_positive:
            negatives_by_dialogue[candidate.canonical_map.dialogue_id].append(candidate)
    for dialogue_id in sorted(development_dialogues, key=lambda item: _seeded_hash(V3_DEV_SEED, item)):
        queue = sorted(
            negatives_by_dialogue.get(dialogue_id, []),
            key=lambda item: _seeded_hash(V3_DEV_SEED, item.update_id),
        )
        if not queue:
            raise ValueError("a v3 development episode has no one-block NONE scenario")
        selected.append(queue[0])
        used.add(queue[0].update_id)

    remaining_negatives = sorted(
        (item for item in eligible if not item.is_positive and item.update_id not in used),
        key=lambda item: _seeded_hash(V3_DEV_SEED, item.update_id),
    )
    selected.extend(remaining_negatives[: 6 - len(selected)])
    used.update(item.update_id for item in selected)

    for label in (
        NormalizedRelationLabel.SUPPORT,
        NormalizedRelationLabel.ATTACK,
        NormalizedRelationLabel.REPHRASE,
    ):
        queue = sorted(
            (
                item
                for item in eligible
                if item.update_id not in used
                and len(item.gold_relations) == 1
                and item.labels == {label}
            ),
            key=lambda item: _seeded_hash(V3_DEV_SEED, item.update_id),
        )
        if len(queue) < 8:
            raise ValueError(f"not enough one-block {label.value} v3 development scenarios")
        selected.extend(queue[:8])
        used.update(item.update_id for item in queue[:8])

    if len(selected) != V3_DEV_SCENARIO_COUNT or len(used) != V3_DEV_SCENARIO_COUNT:
        raise ValueError("v3 development selection is not 30 unique scenarios")
    return sorted(selected, key=lambda item: _seeded_hash(V3_DEV_SEED, item.update_id))


def _write_jsonl(path: Path, rows: list[StrictModel]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row.model_dump_json() + "\n")


def _relation_counts(rows: list[DialAMTrainingRowV3]) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        payload = json.loads(row.messages[1].content)
        counts.update(relation["type"] for relation in payload["relations"])
    return dict(sorted(counts.items()))


def _distribution(values: list[float | int]) -> dict[str, float | int]:
    return {
        "minimum": min(values),
        "mean": mean(values),
        "maximum": max(values),
    }


def build_dialam_v3_training_corpus(
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
        raise ValueError("v3 parent-episode splits overlap")

    all_nonfrozen_examples = [
        example
        for candidate in nonfrozen_candidates
        for example in _materialize_candidate(candidate, PatchSplit.TRAIN)
    ]
    training_examples = [
        item for item in all_nonfrozen_examples if item.dialogue_id in training_dialogues
    ]
    pairs = select_paired_examples(training_examples)
    rows = [
        row
        for pair in pairs
        for row in (_training_row(pair, "POSITIVE"), _training_row(pair, "NONE"))
    ]

    ids = [row.example_id for row in rows]
    if len(rows) != V3_TRAINING_SIZE or len(ids) != len(set(ids)):
        raise ValueError("v3 must contain 4096 unique blocks")
    if len({row.pair_id for row in rows}) != V3_PAIR_COUNT:
        raise ValueError("v3 pair count is invalid")
    pair_rows: dict[str, list[DialAMTrainingRowV3]] = defaultdict(list)
    for row in rows:
        pair_rows[row.pair_id].append(row)
    for pair_id, members in pair_rows.items():
        if len(members) != 2:
            raise ValueError(f"{pair_id} does not contain exactly two rows")
        if {item.pair_role for item in members} != {"POSITIVE", "NONE"}:
            raise ValueError(f"{pair_id} lacks one positive and one NONE row")
        if len({item.update_id for item in members}) != 1:
            raise ValueError(f"{pair_id} crosses update boundaries")
        if members[0].paired_example_id != members[1].example_id or members[1].paired_example_id != members[0].example_id:
            raise ValueError(f"{pair_id} reciprocal example IDs are invalid")

    dev_candidates = _select_dev_candidates(nonfrozen_candidates, development_dialogues)
    dev_examples = [
        _materialize_candidate(candidate, PatchSplit.EVAL)[0]
        for candidate in dev_candidates
    ]
    if {item.dialogue_id for item in dev_examples} & {
        item.dialogue_id for item in training_examples
    }:
        raise ValueError("v3 development episodes leaked into training")
    if {item.dialogue_id for item in dev_examples} & frozen_dialogues:
        raise ValueError("frozen held-out episodes leaked into v3 development")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / V3_OUTPUT_FILENAME
    _write_jsonl(output_path, rows)
    schema_path = output_dir / V3_SCHEMA_FILENAME
    schema_path.write_text(
        json.dumps(DialAMTrainingRowV3.model_json_schema(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    dev_examples_path = output_dir / V3_DEV_EXAMPLES_FILENAME
    _write_jsonl(dev_examples_path, dev_examples)
    dev_inputs_path = output_dir / V3_DEV_INPUTS_FILENAME
    with dev_inputs_path.open("w", encoding="utf-8") as handle:
        for example in dev_examples:
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

    category_counts = Counter(row.category for row in rows)
    role_counts = Counter(row.pair_role for row in rows)
    positive_chars = sum(
        len(row.messages[1].content) for row in rows if row.pair_role == "POSITIVE"
    )
    none_chars = sum(
        len(row.messages[1].content) for row in rows if row.pair_role == "NONE"
    )
    lexical_shared = [pair.metadata.max_shared_content_tokens for pair in pairs]
    overlaps = [pair.metadata.max_overlap_coefficient for pair in pairs]
    dev_categories = Counter(_category(item) for item in dev_examples)
    manifest: dict[str, object] = {
        "schema_version": "dialam_training_manifest_v3",
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_version": "v3-paired-loss-balanced",
        "base_model": "Qwen/Qwen3-0.6B",
        "size": V3_TRAINING_SIZE,
        "pair_count": V3_PAIR_COUNT,
        "selection_seed": V3_TRAINING_SEED,
        "split_unit": "original_parent_episode",
        "training_parent_episode_count": len(training_dialogues),
        "development_parent_episode_count": len(development_dialogues),
        "frozen_parent_episode_count": len(frozen_dialogues),
        "development_parent_episode_set_sha256": hashlib.sha256(
            "\n".join(sorted(development_dialogues)).encode()
        ).hexdigest(),
        "dialogue_leakage": False,
        "category_counts": dict(sorted(category_counts.items())),
        "pair_role_counts": dict(sorted(role_counts.items())),
        "relation_counts": _relation_counts(rows),
        "pair_policy": {
            "description": (
                "Select one positive block and one all-negative block from the exact same "
                "update, use each update at most once, exhaust paired ATTACK/MIXED coverage, "
                "and rank the NONE sibling by deterministic lexical hardness."
            ),
            "same_update_for_every_pair": True,
            "one_pair_per_update": True,
            "duplicates": False,
            "semantic_candidate_retrieval": False,
            "positive_target_counts": V3_POSITIVE_TARGET_COUNTS,
            "pairs_with_one_or_more_shared_content_tokens": sum(value >= 1 for value in lexical_shared),
            "pairs_with_two_or_more_shared_content_tokens": sum(value >= 2 for value in lexical_shared),
            "shared_content_tokens": _distribution(lexical_shared),
            "overlap_coefficient": _distribution(overlaps),
        },
        "loss_policy": {
            "name": "per_example_assistant_token_mean_then_batch_mean",
            "description": (
                "Mask every prompt token, average causal cross-entropy over each row's "
                "assistant tokens, then average rows. This gives a short NONE JSON answer "
                "the same configured example weight as its longer positive sibling."
            ),
            "assistant_loss_weight_per_row": 1.0,
            "positive_to_none_row_ratio": 1.0,
            "unbalanced_positive_to_none_supervised_character_ratio": positive_chars / none_chars,
        },
        "development_evaluation": {
            "policy": "30 one-block scenarios from four parent episodes excluded from v3 training and the frozen gate",
            "examples": len(dev_examples),
            "category_counts": dict(sorted(dev_categories.items())),
            "examples_path": str(dev_examples_path.relative_to(PROJECT_ROOT)),
            "examples_sha256": file_sha256(dev_examples_path),
            "inputs_path": str(dev_inputs_path.relative_to(PROJECT_ROOT)),
            "inputs_sha256": file_sha256(dev_inputs_path),
        },
        "frozen_evaluation": {
            "examples": 30,
            "eval_sha256": file_sha256(DEFAULT_DIAGNOSTIC_EVAL_EXAMPLES_PATH),
            "judge_rubric_sha256": FROZEN_JUDGE_RUBRIC_SHA256,
            "unchanged_from_v1_v2": True,
            "policy": "evaluate once after the development check; never select or tune v3 from frozen responses",
        },
        "preregistered_material_improvement": {
            "semantic_baseline": "v2/n2048 edge_f1=0.21428571428571427",
            "false_edge_baseline": "v1/n2048 false_edges_per_update=0.6666666666666666",
            "minimum_edge_f1": 0.2642857142857143,
            "maximum_false_edges_per_update": 0.6666666666666666,
            "maximum_none_scenarios_with_false_edges": 2,
            "json_and_schema_validity_must_remain": 1.0,
        },
        "source": {
            "archive_sha256": file_sha256(PROJECT_ROOT / "data/source/dialam_qt30/dataset.zip"),
            "dialogues_sha256": file_sha256(dialogues_path),
            "eval_summary_sha256": file_sha256(eval_summary_path),
            "v2_training_manifest_sha256": file_sha256(
                output_dir / "training_v2_manifest.json"
            ),
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
            "The v3 training and development JSONL files contain transformed QT30 text and "
            "remain private/local. This text-free manifest, aggregate statistics, schema, "
            "and deterministic reconstruction code may be published."
        ),
    }
    manifest_path = output_dir / V3_MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest

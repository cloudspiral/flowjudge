from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Literal

from pydantic import Field, model_validator

from .dialam import (
    DEFAULT_DIALOGUES_PATH,
    DEFAULT_SOURCE_DIR,
    SOURCE_ARCHIVE_SHA256,
    load_canonical_maps,
)
from .dialam_training import DEFAULT_TRAINING_DIR, TrainingMessage, file_sha256
from .dialam_training_v3 import V3_DEV_EXAMPLES_FILENAME, _dialogue_split, _seeded_hash
from .dialam_training_v5 import (
    CandidateHardness,
    PairwiseDecision,
    PairwiseLabel,
    POSITIVE_LABELS,
    V5_OUTPUT_FILENAME,
    V5_POSITIVE_TARGET_COUNTS,
    _candidate_hardness,
    _contrast_pool,
    build_pairwise_prompt,
    candidate_label_distribution,
)
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


V8_GROUP_COUNT = 4096
V8_TRAINING_SIZE = V8_GROUP_COUNT * 3
V8_SMOKE_SIZE = 252
V8_SELECTION_SEED = "dialam-qt30-qlora-v8-prior-aware-listwise"
V8_SOURCE_V5_SHA256 = "23f4b8d552a6bca16cdb79507d89ec9675bb16a8d2f5b7b573c6a0981a42d8db"
V8_OUTPUT_FILENAME = "dialam_v8_n12288.jsonl"
V8_SMOKE_OUTPUT_FILENAME = "dialam_v8_resume_smoke_n252.jsonl"
V8_MANIFEST_FILENAME = "training_v8_manifest.json"
V8_SCHEMA_FILENAME = "training_example_v8.schema.json"
V8_PREREGISTRATION_PATH = PROJECT_ROOT / "docs" / "dialam_v8_preregistration.md"


class DialAMListwiseRowV8(StrictModel):
    schema_version: Literal["dialam_qlora_prior_aware_listwise_v8"]
    training_row_id: str
    group_id: str
    group_role: Literal["POSITIVE", "NONE_PRIMARY", "NONE_SECONDARY"]
    group_position: int = Field(ge=0, le=2)
    source_positive_key: str
    source_example_id: str
    update_id: str
    dialogue_id: str
    block_index: int = Field(ge=0)
    candidate_target_id: str
    label: PairwiseLabel
    positive_label: Literal["SUPPORT", "ATTACK", "REPHRASE"]
    repetition_index: int = Field(ge=0)
    is_repeated_positive: bool
    candidate_hardness: CandidateHardness
    messages: list[TrainingMessage] = Field(min_length=2, max_length=2)

    @model_validator(mode="after")
    def validate_listwise_row(self) -> "DialAMListwiseRowV8":
        expected_label = self.positive_label if self.group_role == "POSITIVE" else "NONE"
        if self.label != expected_label:
            raise ValueError("v8 group role and label disagree")
        expected_position = {
            "POSITIVE": 0,
            "NONE_PRIMARY": 1,
            "NONE_SECONDARY": 2,
        }[self.group_role]
        if self.group_position != expected_position:
            raise ValueError("v8 group role and position disagree")
        if self.messages[1].role != "assistant" or self.messages[1].content != self.label:
            raise ValueError("assistant target must be the exact listwise label")
        return self


@dataclass(frozen=True)
class SelectedListwiseGroupV8:
    group_id: str
    positive: PairwiseDecision
    primary_negative: PairwiseDecision
    secondary_negative: PairwiseDecision
    repetition_index: int


def _positive_selection_rank(decision: PairwiseDecision) -> tuple[str, str]:
    return (
        _seeded_hash(V8_SELECTION_SEED, decision.key),
        decision.key,
    )


def select_v8_groups(
    examples: list[PatchExample],
    *,
    target_counts: dict[str, int] = V5_POSITIVE_TARGET_COUNTS,
) -> list[SelectedListwiseGroupV8]:
    if target_counts is V5_POSITIVE_TARGET_COUNTS and sum(target_counts.values()) != V8_GROUP_COUNT:
        raise ValueError("v8 positive targets must sum to the registered group count")
    pools, _ = _contrast_pool(examples)
    groups: list[SelectedListwiseGroupV8] = []
    for label in POSITIVE_LABELS:
        target = target_counts.get(label, 0)
        eligible = sorted(
            (item for item in pools.get(label, []) if len(item.negatives) >= 2),
            key=lambda item: _positive_selection_rank(item.positive),
        )
        if target and not eligible:
            raise ValueError(f"v8 has no eligible {label} decisions with two negatives")
        occurrences: Counter[str] = Counter()
        for index in range(target):
            item = eligible[index % len(eligible)]
            repetition_index = occurrences[item.positive.key]
            occurrences[item.positive.key] += 1
            first_index = (2 * repetition_index) % len(item.negatives)
            second_index = (first_index + 1) % len(item.negatives)
            primary = item.negatives[first_index]
            secondary = item.negatives[second_index]
            if primary.candidate_target_id == secondary.candidate_target_id:
                raise ValueError("v8 hard negatives must be distinct within a group")
            group_id = "v8group-" + _seeded_hash(
                V8_SELECTION_SEED,
                f"{item.positive.key}|{repetition_index}",
            )[:24]
            groups.append(
                SelectedListwiseGroupV8(
                    group_id=group_id,
                    positive=item.positive,
                    primary_negative=primary,
                    secondary_negative=secondary,
                    repetition_index=repetition_index,
                )
            )
    if Counter(group.positive.label for group in groups) != Counter(target_counts):
        raise ValueError("v8 positive target mix differs from the registered counts")
    if len({group.group_id for group in groups}) != len(groups):
        raise ValueError("v8 group IDs are not unique")
    return sorted(
        groups,
        key=lambda group: _seeded_hash(V8_SELECTION_SEED, group.group_id),
    )


def materialize_v8_rows(
    groups: list[SelectedListwiseGroupV8],
) -> list[DialAMListwiseRowV8]:
    rows: list[DialAMListwiseRowV8] = []
    for group in groups:
        decisions = (
            ("POSITIVE", 0, group.positive),
            ("NONE_PRIMARY", 1, group.primary_negative),
            ("NONE_SECONDARY", 2, group.secondary_negative),
        )
        for role, position, decision in decisions:
            label: PairwiseLabel = (
                group.positive.label if role == "POSITIVE" else "NONE"
            )
            rows.append(
                DialAMListwiseRowV8(
                    schema_version="dialam_qlora_prior_aware_listwise_v8",
                    training_row_id="v8row-"
                    + _seeded_hash(V8_SELECTION_SEED, f"{group.group_id}|{role}")[:24],
                    group_id=group.group_id,
                    group_role=role,
                    group_position=position,
                    source_positive_key=group.positive.key,
                    source_example_id=decision.example.example_id,
                    update_id=decision.example.update_id,
                    dialogue_id=decision.example.dialogue_id,
                    block_index=decision.example.block_index,
                    candidate_target_id=decision.candidate_target_id,
                    label=label,
                    positive_label=group.positive.label,  # type: ignore[arg-type]
                    repetition_index=group.repetition_index,
                    is_repeated_positive=group.repetition_index > 0,
                    candidate_hardness=_candidate_hardness(decision),
                    messages=[
                        TrainingMessage(
                            role="user",
                            content=build_pairwise_prompt(
                                decision.example,
                                decision.candidate_target_id,
                            ),
                        ),
                        TrainingMessage(role="assistant", content=label),
                    ],
                )
            )
    return rows


def _validate_groups(rows: list[DialAMListwiseRowV8]) -> None:
    if len(rows) % 3:
        raise ValueError("v8 rows must contain complete adjacent triples")
    for index in range(0, len(rows), 3):
        group = rows[index : index + 3]
        if [row.group_position for row in group] != [0, 1, 2]:
            raise ValueError(f"v8 rows {index}:{index + 3} are not ordered triples")
        if len({row.group_id for row in group}) != 1:
            raise ValueError(f"v8 rows {index}:{index + 3} cross group IDs")
        if len({row.source_example_id for row in group}) != 1:
            raise ValueError(f"v8 rows {index}:{index + 3} cross source blocks")
        if len({row.candidate_target_id for row in group}) != 3:
            raise ValueError(f"v8 rows {index}:{index + 3} repeat a candidate")
        if [row.label for row in group[1:]] != ["NONE", "NONE"]:
            raise ValueError(f"v8 rows {index}:{index + 3} lack two NONE labels")


def _write_jsonl(path: Path, rows: list[DialAMListwiseRowV8]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row.model_dump_json() + "\n")


def _distribution(values: list[int | float]) -> dict[str, int | float]:
    return {"minimum": min(values), "mean": mean(values), "maximum": max(values)}


def build_dialam_v8_listwise_corpus(
    *,
    source_dir: Path = DEFAULT_SOURCE_DIR,
    dialogues_path: Path = DEFAULT_DIALOGUES_PATH,
    eval_summary_path: Path = DEFAULT_EVAL_SUMMARY_PATH,
    frozen_examples_path: Path = DEFAULT_DIAGNOSTIC_EVAL_EXAMPLES_PATH,
    output_dir: Path = DEFAULT_TRAINING_DIR,
) -> dict[str, object]:
    if not V8_PREREGISTRATION_PATH.is_file():
        raise FileNotFoundError("v8 preregistration must exist before corpus construction")
    source_v5_path = output_dir / V5_OUTPUT_FILENAME
    if file_sha256(source_v5_path) != V8_SOURCE_V5_SHA256:
        raise ValueError("v8 provenance v5 corpus differs from the frozen source bytes")

    eval_summary = json.loads(eval_summary_path.read_text(encoding="utf-8"))
    frozen_dialogues = set(eval_summary["heldout_dialogue_ids"])
    maps = load_canonical_maps(source_dir, dialogues_path)
    candidates = build_update_candidates(maps)
    nonfrozen_candidates = [
        item
        for item in candidates
        if item.canonical_map.dialogue_id not in frozen_dialogues
    ]
    training_dialogues, development_dialogues = _dialogue_split(
        {item.canonical_map.dialogue_id for item in nonfrozen_candidates}
    )
    if (
        training_dialogues & development_dialogues
        or training_dialogues & frozen_dialogues
        or development_dialogues & frozen_dialogues
    ):
        raise ValueError("v8 parent-episode splits overlap")

    training_examples = [
        example
        for candidate in nonfrozen_candidates
        if candidate.canonical_map.dialogue_id in training_dialogues
        for example in _materialize_candidate(candidate, PatchSplit.TRAIN)
    ]
    available_counts, ambiguous_count = candidate_label_distribution(training_examples)
    pools, pool_ambiguous_count = _contrast_pool(training_examples)
    if ambiguous_count != pool_ambiguous_count:
        raise ValueError("v8 ambiguity accounting differs across passes")
    eligible_counts = {
        label: sum(len(item.negatives) >= 2 for item in pools.get(label, []))
        for label in POSITIVE_LABELS
    }
    groups = select_v8_groups(training_examples)
    rows = materialize_v8_rows(groups)
    _validate_groups(rows)
    if len(rows) != V8_TRAINING_SIZE:
        raise ValueError(f"v8 must contain exactly {V8_TRAINING_SIZE} rows")
    if len({row.training_row_id for row in rows}) != len(rows):
        raise ValueError("v8 training-row IDs are not unique")
    if {row.dialogue_id for row in rows} - training_dialogues:
        raise ValueError("v8 contains a development or frozen parent episode")

    smoke_rows = rows[:V8_SMOKE_SIZE]
    _validate_groups(smoke_rows)
    if len({row.group_id for row in smoke_rows}) != V8_SMOKE_SIZE // 3:
        raise ValueError("v8 smoke prefix does not preserve complete triples")
    if set(row.label for row in smoke_rows) != {"NONE", *POSITIVE_LABELS}:
        raise ValueError("v8 smoke prefix does not cover all four labels")

    dev_examples_path = output_dir / V3_DEV_EXAMPLES_FILENAME
    dev_examples = load_jsonl(dev_examples_path, PatchExample)
    if {item.dialogue_id for item in dev_examples} != development_dialogues:
        raise ValueError("v8 development episodes differ from the frozen v3 split")
    frozen_examples = load_jsonl(frozen_examples_path, PatchExample)
    if {item.dialogue_id for item in frozen_examples} != frozen_dialogues:
        raise ValueError("v8 frozen episodes differ from the immutable evaluation")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / V8_OUTPUT_FILENAME
    smoke_path = output_dir / V8_SMOKE_OUTPUT_FILENAME
    schema_path = output_dir / V8_SCHEMA_FILENAME
    _write_jsonl(output_path, rows)
    _write_jsonl(smoke_path, smoke_rows)
    schema_path.write_text(
        json.dumps(DialAMListwiseRowV8.model_json_schema(), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    positive_occurrences = Counter(group.positive.key for group in groups)
    negative_hardness = [
        _candidate_hardness(decision)
        for group in groups
        for decision in (group.primary_negative, group.secondary_negative)
    ]
    manifest: dict[str, object] = {
        "schema_version": "dialam_training_manifest_v8",
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_version": "v8-qt30-prior-aware-listwise",
        "size": len(rows),
        "group_count": len(groups),
        "smoke_size": len(smoke_rows),
        "selection_seed": V8_SELECTION_SEED,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "source_v5_sha256": file_sha256(source_v5_path),
        "split_unit": "original_parent_episode",
        "training_parent_episode_count": len(training_dialogues),
        "development_parent_episode_count": len(development_dialogues),
        "frozen_parent_episode_count": len(frozen_dialogues),
        "dialogue_leakage": False,
        "new_external_data": False,
        "available_candidate_label_counts": dict(sorted(available_counts.items())),
        "available_candidate_positive_rate": (
            sum(count for label, count in available_counts.items() if label != "NONE")
            / sum(available_counts.values())
        ),
        "available_unique_positive_decisions": {
            label: len(pools.get(label, [])) for label in POSITIVE_LABELS
        },
        "eligible_two_negative_positive_decisions": eligible_counts,
        "insufficient_negative_positive_decisions_excluded": {
            label: len(pools.get(label, [])) - eligible_counts[label]
            for label in POSITIVE_LABELS
        },
        "ambiguous_same_pair_decisions_excluded": ambiguous_count,
        "row_label_counts": dict(sorted(Counter(row.label for row in rows).items())),
        "group_role_counts": dict(
            sorted(Counter(row.group_role for row in rows).items())
        ),
        "positive_target_counts": dict(
            sorted(Counter(group.positive.label for group in groups).items())
        ),
        "group_policy": {
            "same_complete_block": True,
            "one_positive_two_none": True,
            "distinct_candidate_targets_within_group": True,
            "unique_positives_before_repetition": True,
            "unique_positive_decisions": len(positive_occurrences),
            "repeated_positive_groups": sum(
                count - 1 for count in positive_occurrences.values()
            ),
            "maximum_positive_occurrences": max(positive_occurrences.values()),
            "semantic_candidate_retrieval": False,
            "negative_shared_content_tokens": _distribution(
                [item.shared_content_tokens for item in negative_hardness]
            ),
            "negative_overlap_coefficient": _distribution(
                [item.overlap_coefficient for item in negative_hardness]
            ),
        },
        "objective": {
            "name": "restricted_four_label_score_cross_entropy",
            "labels": ["NONE", "SUPPORT", "ATTACK", "REPHRASE"],
            "sequence_score": "mean label-token log probability",
            "loss": "cross_entropy(stack(four_label_scores), gold_label_index)",
            "generative_token_nll": False,
            "preference_loss": False,
            "class_weights": False,
            "label_smoothing": False,
        },
        "continuation": {
            "source_adapter": "v5_n8192",
            "source_adapter_tree_sha256": (
                "cbd9a2a6cae8ab988d78b7694ac9f9829bf9262bbb7a6fab838894a5080f122a"
            ),
            "optimizer_state_reset_between_v5_and_v8": True,
        },
        "fixed_training_config": {
            "epochs": 1.0,
            "learning_rate": 2e-5,
            "effective_batch_size": 8,
            "seed": 20260823,
            "save_steps": {"v8-smoke": 8, "v8": 100},
            "save_total_limit": 2,
        },
        "resumability": {
            "smoke_subset_is_nested_prefix": True,
            "periodic_full_trainer_state": True,
            "persistent_modal_volume_commit_on_every_save": True,
            "identity_mismatch_refuses_resume": True,
            "completed_run_is_idempotent_in_auto_mode": True,
        },
        "development_gate": {
            "exact_patch_accuracy_min": 0.5333333333333333,
            "edge_f1_min": 0.56,
            "relation_macro_f1_min": 0.5427350427350427,
            "false_edges_per_update_max": 0.3,
            "none_scenarios_with_false_edges_max": 2,
            "json_validity_rate_min": 1.0,
            "schema_validity_rate_min": 1.0,
        },
        "frozen_promotion_gate": {
            "exact_patch_accuracy_min": 0.5333333333333333,
            "edge_f1_min": 0.5061904761904762,
            "relation_macro_f1_min": 0.4743589743589744,
            "attack_f1_min": 0.3076923076923077,
            "false_edges_per_update_max": 0.26666666666666666,
            "none_scenarios_with_false_edges_max": 0,
            "json_validity_rate_min": 1.0,
            "schema_validity_rate_min": 1.0,
        },
        "evaluation_hashes": {
            "development_examples_sha256": file_sha256(dev_examples_path),
            "frozen_examples_sha256": file_sha256(frozen_examples_path),
        },
        "preregistration": {
            "path": str(V8_PREREGISTRATION_PATH.relative_to(PROJECT_ROOT)),
            "sha256": file_sha256(V8_PREREGISTRATION_PATH),
        },
        "output": {
            "path": str(output_path.relative_to(PROJECT_ROOT)),
            "sha256": file_sha256(output_path),
        },
        "smoke_output": {
            "path": str(smoke_path.relative_to(PROJECT_ROOT)),
            "sha256": file_sha256(smoke_path),
        },
        "schema": {
            "path": str(schema_path.relative_to(PROJECT_ROOT)),
            "sha256": file_sha256(schema_path),
        },
        "publication": (
            "Rows contain transformed QT30 text and remain local/ignored; the tracked "
            "manifest and deterministic reconstruction code contain no corpus text."
        ),
    }
    manifest_path = output_dir / V8_MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest

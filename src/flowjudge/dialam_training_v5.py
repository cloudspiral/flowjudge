from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Literal

from pydantic import Field, model_validator

from .dialam import DEFAULT_DIALOGUES_PATH, DEFAULT_SOURCE_DIR, load_canonical_maps
from .dialam_training import DEFAULT_TRAINING_DIR, TrainingMessage, file_sha256
from .dialam_training_v2 import _content_tokens
from .dialam_training_v3 import (
    V3_DEV_EXAMPLES_FILENAME,
    _dialogue_split,
    _seeded_hash,
)
from .patch_ceiling import (
    FROZEN_JUDGE_RUBRIC_SHA256,
    render_patch_block,
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


PairwiseLabel = Literal["NONE", "SUPPORT", "ATTACK", "REPHRASE"]
POSITIVE_LABELS: tuple[PairwiseLabel, ...] = ("ATTACK", "REPHRASE", "SUPPORT")
V5_TRAINING_SIZE = 8192
V5_CONTRAST_COUNT = V5_TRAINING_SIZE // 2
V5_TRAINING_SEED = "dialam-qt30-qlora-v5-pairwise-classification"
V5_POSITIVE_TARGET_COUNTS: dict[str, int] = {
    "ATTACK": 1366,
    "REPHRASE": 1365,
    "SUPPORT": 1365,
}
V5_OUTPUT_FILENAME = "dialam_v5_n8192.jsonl"
V5_MANIFEST_FILENAME = "training_v5_manifest.json"
V5_SCHEMA_FILENAME = "training_example_v5.schema.json"
V5_DEV_INPUTS_FILENAME = "v5_dev_pairwise_inputs.jsonl"
V5_FROZEN_INPUTS_FILENAME = "v5_frozen_pairwise_inputs.jsonl"
V5_PROMPT_PATH = PROJECT_ROOT / "prompts" / "dialam" / "pairwise_v5.txt"


class CandidateHardness(StrictModel):
    shared_content_tokens: int = Field(ge=0)
    overlap_coefficient: float = Field(ge=0.0, le=1.0)
    jaccard_similarity: float = Field(ge=0.0, le=1.0)
    turn_gap: int = Field(gt=0)


class DialAMTrainingRowV5(StrictModel):
    schema_version: Literal["dialam_qlora_pairwise_v5"]
    training_row_id: str
    pair_group_id: str
    pair_role: Literal["POSITIVE", "NONE"]
    source_example_id: str
    update_id: str
    dialogue_id: str
    block_index: int = Field(ge=0)
    candidate_target_id: str
    paired_candidate_target_id: str
    label: PairwiseLabel
    positive_label: Literal["SUPPORT", "ATTACK", "REPHRASE"]
    repetition_index: int = Field(ge=0)
    is_repeated_positive: bool
    negative_hardness: CandidateHardness
    assistant_loss_weight: Literal[1.0]
    messages: list[TrainingMessage] = Field(min_length=2, max_length=2)

    @model_validator(mode="after")
    def validate_pairwise_row(self) -> "DialAMTrainingRowV5":
        expected_role = "NONE" if self.label == "NONE" else "POSITIVE"
        if self.pair_role != expected_role:
            raise ValueError("pair role and label disagree")
        if self.messages[1].role != "assistant" or self.messages[1].content != self.label:
            raise ValueError("assistant target must be the exact pairwise label")
        if self.candidate_target_id == self.paired_candidate_target_id:
            raise ValueError("contrast pair must use two different candidate IDs")
        return self


class PairwiseEvalCandidate(StrictModel):
    target_id: str
    prompt: str = Field(min_length=1)


class PairwiseEvalInput(StrictModel):
    schema_version: Literal["dialam_pairwise_eval_v5"]
    example_id: str
    update_id: str
    source_id: str
    candidates: list[PairwiseEvalCandidate] = Field(min_length=1)

    @model_validator(mode="after")
    def reject_duplicate_candidates(self) -> "PairwiseEvalInput":
        target_ids = [item.target_id for item in self.candidates]
        if len(target_ids) != len(set(target_ids)):
            raise ValueError("pairwise evaluation input contains duplicate targets")
        return self


@dataclass(frozen=True)
class PairwiseDecision:
    example: PatchExample
    candidate_target_id: str
    label: PairwiseLabel

    @property
    def key(self) -> str:
        return f"{self.example.example_id}|{self.candidate_target_id}|{self.label}"


@dataclass(frozen=True)
class _PositiveCandidate:
    positive: PairwiseDecision
    negatives: tuple[PairwiseDecision, ...]


@dataclass(frozen=True)
class SelectedContrast:
    pair_group_id: str
    positive: PairwiseDecision
    negative: PairwiseDecision
    repetition_index: int
    negative_hardness: CandidateHardness


def build_pairwise_prompt(example: PatchExample, candidate_target_id: str) -> str:
    earlier_ids = {item.id for item in example.earlier_propositions}
    if candidate_target_id not in earlier_ids:
        raise ValueError("candidate target ID is absent from the complete block")
    template = V5_PROMPT_PATH.read_text(encoding="utf-8")
    replacements = {
        "{{CANDIDATE_TARGET_ID}}": json.dumps(candidate_target_id),
        "{{BLOCK}}": render_patch_block(example),
    }
    for marker, value in replacements.items():
        if template.count(marker) != 1:
            raise ValueError(f"pairwise_v5.txt must contain exactly one {marker}")
        template = template.replace(marker, value)
    return template


def pairwise_decisions(example: PatchExample) -> tuple[list[PairwiseDecision], int]:
    relations_by_target: dict[str, list[str]] = defaultdict(list)
    for relation in example.gold_patch.relations:
        relations_by_target[relation.target].append(relation.type.value)

    decisions: list[PairwiseDecision] = []
    ambiguous = 0
    for earlier in example.earlier_propositions:
        labels = relations_by_target.get(earlier.id, [])
        if len(labels) > 1:
            ambiguous += 1
            continue
        label: PairwiseLabel = labels[0] if labels else "NONE"  # type: ignore[assignment]
        decisions.append(
            PairwiseDecision(
                example=example,
                candidate_target_id=earlier.id,
                label=label,
            )
        )
    return decisions, ambiguous


def candidate_label_distribution(
    examples: list[PatchExample],
) -> tuple[Counter[str], int]:
    counts: Counter[str] = Counter()
    ambiguous = 0
    for example in examples:
        decisions, excluded = pairwise_decisions(example)
        counts.update(item.label for item in decisions)
        ambiguous += excluded
    return counts, ambiguous


def _candidate_hardness(decision: PairwiseDecision) -> CandidateHardness:
    example = decision.example
    candidate = next(
        item
        for item in example.earlier_propositions
        if item.id == decision.candidate_target_id
    )
    new_tokens = _content_tokens(example.new_proposition.text)
    candidate_tokens = _content_tokens(candidate.text)
    shared = len(new_tokens & candidate_tokens)
    return CandidateHardness(
        shared_content_tokens=shared,
        overlap_coefficient=shared
        / max(1, min(len(new_tokens), len(candidate_tokens))),
        jaccard_similarity=shared / max(1, len(new_tokens | candidate_tokens)),
        turn_gap=example.new_proposition.chronological_turn
        - candidate.chronological_turn,
    )


def _negative_rank(decision: PairwiseDecision) -> tuple[object, ...]:
    hardness = _candidate_hardness(decision)
    return (
        -int(hardness.shared_content_tokens >= 2),
        -hardness.overlap_coefficient,
        -hardness.shared_content_tokens,
        -hardness.jaccard_similarity,
        hardness.turn_gap,
        _seeded_hash(V5_TRAINING_SEED, decision.key),
    )


def _positive_rank(item: _PositiveCandidate) -> tuple[str, str]:
    return (_seeded_hash(V5_TRAINING_SEED, item.positive.key), item.positive.key)


def _contrast_pool(examples: list[PatchExample]) -> tuple[dict[str, list[_PositiveCandidate]], int]:
    pools: dict[str, list[_PositiveCandidate]] = defaultdict(list)
    ambiguous = 0
    for example in examples:
        decisions, excluded = pairwise_decisions(example)
        ambiguous += excluded
        negatives = tuple(
            sorted(
                (item for item in decisions if item.label == "NONE"),
                key=_negative_rank,
            )
        )
        if not negatives:
            continue
        for decision in decisions:
            if decision.label != "NONE":
                pools[decision.label].append(
                    _PositiveCandidate(positive=decision, negatives=negatives)
                )
    for label in pools:
        pools[label] = sorted(pools[label], key=_positive_rank)
    return pools, ambiguous


def select_v5_contrasts(
    examples: list[PatchExample],
    *,
    target_counts: dict[str, int] = V5_POSITIVE_TARGET_COUNTS,
) -> list[SelectedContrast]:
    if sum(target_counts.values()) != V5_CONTRAST_COUNT and target_counts is V5_POSITIVE_TARGET_COUNTS:
        raise ValueError("v5 positive targets must sum to the registered contrast count")
    pools, _ = _contrast_pool(examples)
    selected: list[SelectedContrast] = []
    for label in POSITIVE_LABELS:
        target = target_counts.get(label, 0)
        pool = pools.get(label, [])
        if target and not pool:
            raise ValueError(f"v5 has no eligible {label} decisions")
        occurrences: Counter[str] = Counter()
        for index in range(target):
            item = pool[index % len(pool)]
            repetition_index = occurrences[item.positive.key]
            occurrences[item.positive.key] += 1
            negative = item.negatives[repetition_index % len(item.negatives)]
            pair_group_id = "v5pair-" + _seeded_hash(
                V5_TRAINING_SEED,
                f"{item.positive.key}|{negative.key}|{repetition_index}",
            )[:24]
            selected.append(
                SelectedContrast(
                    pair_group_id=pair_group_id,
                    positive=item.positive,
                    negative=negative,
                    repetition_index=repetition_index,
                    negative_hardness=_candidate_hardness(negative),
                )
            )
    actual = Counter(item.positive.label for item in selected)
    if actual != Counter(target_counts):
        raise ValueError(f"v5 positive target mix is invalid: {dict(actual)}")
    if len({item.pair_group_id for item in selected}) != len(selected):
        raise ValueError("v5 pair-group IDs are not unique")
    return sorted(
        selected,
        key=lambda item: _seeded_hash(V5_TRAINING_SEED, item.pair_group_id),
    )


def materialize_v5_rows(
    contrasts: list[SelectedContrast],
) -> list[DialAMTrainingRowV5]:
    rows: list[DialAMTrainingRowV5] = []
    for contrast in contrasts:
        for role, decision, paired in (
            ("POSITIVE", contrast.positive, contrast.negative),
            ("NONE", contrast.negative, contrast.positive),
        ):
            training_row_id = "v5row-" + _seeded_hash(
                V5_TRAINING_SEED,
                f"{contrast.pair_group_id}|{role}",
            )[:24]
            rows.append(
                DialAMTrainingRowV5(
                    schema_version="dialam_qlora_pairwise_v5",
                    training_row_id=training_row_id,
                    pair_group_id=contrast.pair_group_id,
                    pair_role=role,
                    source_example_id=decision.example.example_id,
                    update_id=decision.example.update_id,
                    dialogue_id=decision.example.dialogue_id,
                    block_index=decision.example.block_index,
                    candidate_target_id=decision.candidate_target_id,
                    paired_candidate_target_id=paired.candidate_target_id,
                    label=decision.label,
                    positive_label=contrast.positive.label,  # type: ignore[arg-type]
                    repetition_index=contrast.repetition_index,
                    is_repeated_positive=contrast.repetition_index > 0,
                    negative_hardness=contrast.negative_hardness,
                    assistant_loss_weight=1.0,
                    messages=[
                        TrainingMessage(
                            role="user",
                            content=build_pairwise_prompt(
                                decision.example,
                                decision.candidate_target_id,
                            ),
                        ),
                        TrainingMessage(role="assistant", content=decision.label),
                    ],
                )
            )
    return rows


def build_pairwise_eval_input(example: PatchExample) -> PairwiseEvalInput:
    return PairwiseEvalInput(
        schema_version="dialam_pairwise_eval_v5",
        example_id=example.example_id,
        update_id=example.update_id,
        source_id=example.new_proposition.id,
        candidates=[
            PairwiseEvalCandidate(
                target_id=item.id,
                prompt=build_pairwise_prompt(example, item.id),
            )
            for item in example.earlier_propositions
        ],
    )


def assemble_pairwise_patch(
    source_id: str,
    candidate_ids: list[str],
    labels: list[PairwiseLabel],
) -> str:
    if len(candidate_ids) != len(labels):
        raise ValueError("candidate and label coverage differs")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate IDs are not unique")
    relations = [
        {"source": source_id, "target": target_id, "type": label}
        for target_id, label in zip(candidate_ids, labels, strict=True)
        if label != "NONE"
    ]
    return json.dumps({"relations": relations}, separators=(",", ":"))


def _write_jsonl(path: Path, rows: list[StrictModel]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row.model_dump_json() + "\n")


def _distribution(values: list[int | float]) -> dict[str, int | float]:
    return {"minimum": min(values), "mean": mean(values), "maximum": max(values)}


def build_dialam_v5_training_corpus(
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
        raise ValueError("v5 parent-episode splits overlap")

    training_examples = [
        example
        for candidate in nonfrozen_candidates
        if candidate.canonical_map.dialogue_id in training_dialogues
        for example in _materialize_candidate(candidate, PatchSplit.TRAIN)
    ]
    available_decision_counts, available_ambiguous_count = candidate_label_distribution(
        training_examples
    )
    pools, ambiguous_pair_count = _contrast_pool(training_examples)
    if available_ambiguous_count != ambiguous_pair_count:
        raise ValueError("v5 ambiguous-decision accounting differs across passes")
    contrasts = select_v5_contrasts(training_examples)
    rows = materialize_v5_rows(contrasts)

    if len(rows) != V5_TRAINING_SIZE:
        raise ValueError(f"v5 must contain {V5_TRAINING_SIZE} rows")
    if len({row.training_row_id for row in rows}) != len(rows):
        raise ValueError("v5 training-row IDs are not unique")
    if {row.dialogue_id for row in rows} - training_dialogues:
        raise ValueError("v5 contains a development or frozen parent episode")
    pair_groups: dict[str, list[DialAMTrainingRowV5]] = defaultdict(list)
    for row in rows:
        pair_groups[row.pair_group_id].append(row)
    for pair_group_id, members in pair_groups.items():
        if len(members) != 2 or {item.pair_role for item in members} != {"POSITIVE", "NONE"}:
            raise ValueError(f"{pair_group_id} is not one positive/NONE contrast")
        if len({item.source_example_id for item in members}) != 1:
            raise ValueError(f"{pair_group_id} does not preserve the exact block")

    dev_examples_path = output_dir / V3_DEV_EXAMPLES_FILENAME
    dev_examples = load_jsonl(dev_examples_path, PatchExample)
    if {item.dialogue_id for item in dev_examples} != development_dialogues:
        raise ValueError("v5 development episodes differ from the frozen v3 split")
    frozen_examples = load_jsonl(DEFAULT_DIAGNOSTIC_EVAL_EXAMPLES_PATH, PatchExample)
    if {item.dialogue_id for item in frozen_examples} != frozen_dialogues:
        raise ValueError("v5 frozen episodes differ from the immutable evaluation")
    dev_inputs = [build_pairwise_eval_input(item) for item in dev_examples]
    frozen_inputs = [build_pairwise_eval_input(item) for item in frozen_examples]

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / V5_OUTPUT_FILENAME
    _write_jsonl(output_path, rows)
    schema_path = output_dir / V5_SCHEMA_FILENAME
    schema_path.write_text(
        json.dumps(DialAMTrainingRowV5.model_json_schema(), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    dev_inputs_path = output_dir / V5_DEV_INPUTS_FILENAME
    frozen_inputs_path = output_dir / V5_FROZEN_INPUTS_FILENAME
    _write_jsonl(dev_inputs_path, dev_inputs)
    _write_jsonl(frozen_inputs_path, frozen_inputs)

    negative_hardness = [item.negative_hardness for item in contrasts]
    positive_occurrences = Counter(item.positive.key for item in contrasts)
    prompt_lengths = [len(row.messages[0].content) for row in rows]
    manifest: dict[str, object] = {
        "schema_version": "dialam_training_manifest_v5",
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_version": "v5-pairwise-classification",
        "base_model": "Qwen/Qwen3-0.6B",
        "size": V5_TRAINING_SIZE,
        "contrast_pair_count": V5_CONTRAST_COUNT,
        "selection_seed": V5_TRAINING_SEED,
        "split_unit": "original_parent_episode",
        "training_parent_episode_count": len(training_dialogues),
        "development_parent_episode_count": len(development_dialogues),
        "frozen_parent_episode_count": len(frozen_dialogues),
        "dialogue_leakage": False,
        "row_label_counts": dict(sorted(Counter(row.label for row in rows).items())),
        "available_candidate_label_counts": dict(
            sorted(available_decision_counts.items())
        ),
        "available_candidate_positive_rate": (
            sum(
                count
                for label, count in available_decision_counts.items()
                if label != "NONE"
            )
            / sum(available_decision_counts.values())
        ),
        "positive_target_counts": dict(
            sorted(Counter(item.positive.label for item in contrasts).items())
        ),
        "available_unique_positive_decisions": {
            label: len(pools.get(label, [])) for label in POSITIVE_LABELS
        },
        "ambiguous_same_pair_decisions_excluded": ambiguous_pair_count,
        "pair_policy": {
            "same_complete_block_for_every_contrast": True,
            "one_positive_and_one_none_per_device_batch": True,
            "unique_positives_before_repetition": True,
            "unique_positive_decisions": len(positive_occurrences),
            "repeated_positive_rows": sum(value - 1 for value in positive_occurrences.values()),
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
            "labels": ["NONE", "SUPPORT", "ATTACK", "REPHRASE"],
            "prompt_sha256": file_sha256(V5_PROMPT_PATH),
            "assistant_target": "one exact class label",
            "inference": "mean label-token log probability, deterministic NONE-first tie break",
            "patch_assembly": "deterministic supplied-ID union of non-NONE decisions",
            "prompt_character_count": _distribution(prompt_lengths),
        },
        "loss_policy": {
            "name": "paired_sequential_per_example_assistant_token_mean",
            "assistant_loss_weight_per_row": 1.0,
            "positive_to_none_row_ratio": 1.0,
            "shuffle": False,
            "reason": "builder shuffles pair groups deterministically and keeps each two-row contrast in one device batch",
        },
        "development_evaluation": {
            "examples": len(dev_inputs),
            "candidate_decisions": sum(len(item.candidates) for item in dev_inputs),
            "examples_sha256": file_sha256(dev_examples_path),
            "inputs_path": str(dev_inputs_path.relative_to(PROJECT_ROOT)),
            "inputs_sha256": file_sha256(dev_inputs_path),
        },
        "frozen_evaluation": {
            "examples": len(frozen_inputs),
            "candidate_decisions": sum(len(item.candidates) for item in frozen_inputs),
            "eval_sha256": file_sha256(DEFAULT_DIAGNOSTIC_EVAL_EXAMPLES_PATH),
            "inputs_path": str(frozen_inputs_path.relative_to(PROJECT_ROOT)),
            "inputs_sha256": file_sha256(frozen_inputs_path),
            "judge_rubric_sha256": FROZEN_JUDGE_RUBRIC_SHA256,
            "policy": "run only if every preregistered development gate condition passes",
        },
        "development_gate": {
            "exact_patch_accuracy_min": 0.30,
            "edge_f1_min": 0.39146341463414636,
            "relation_macro_f1_min": 0.3151515151515151,
            "false_edges_per_update_max": 0.3333333333333333,
            "none_scenarios_with_false_edges_max": 2,
            "json_validity_rate_min": 1.0,
            "schema_validity_rate_min": 1.0,
        },
        "source": {
            "archive_sha256": file_sha256(
                PROJECT_ROOT / "data" / "source" / "dialam_qt30" / "dataset.zip"
            ),
            "dialogues_sha256": file_sha256(dialogues_path),
            "eval_summary_sha256": file_sha256(eval_summary_path),
            "v4_training_manifest_sha256": file_sha256(
                output_dir / "training_v4_manifest.json"
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
        "publication": (
            "The generated rows contain transformed QT30 text. They remain ignored during "
            "the experiment and may be uploaded only under the project owner's reported "
            "redistribution permission with that permission basis documented."
        ),
    }
    manifest_path = output_dir / V5_MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest

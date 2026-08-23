from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable, Literal, TypeVar

from pydantic import BaseModel, Field, model_validator

from .dialam import (
    DEFAULT_DIALOGUES_PATH,
    DEFAULT_SOURCE_DIR,
    CanonicalMap,
    CanonicalProposition,
    CanonicalRelation,
    NormalizedRelationLabel,
    load_canonical_maps,
)
from .schemas import StrictModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "dialam"
DEFAULT_TRAIN_PATH = DEFAULT_OUTPUT_DIR / "smoke_train.jsonl"
DEFAULT_EVAL_SCENARIOS_PATH = DEFAULT_OUTPUT_DIR / "smoke_eval_scenarios.jsonl"
DEFAULT_EVAL_EXAMPLES_PATH = DEFAULT_OUTPUT_DIR / "smoke_eval_examples.jsonl"
DEFAULT_DIAGNOSTIC_EVAL_SCENARIOS_PATH = (
    DEFAULT_OUTPUT_DIR / "balanced_diagnostic_eval_scenarios.jsonl"
)
DEFAULT_DIAGNOSTIC_EVAL_EXAMPLES_PATH = (
    DEFAULT_OUTPUT_DIR / "balanced_diagnostic_eval_examples.jsonl"
)
DEFAULT_EVAL_SUMMARY_PATH = DEFAULT_OUTPUT_DIR / "eval_summary.json"
DEFAULT_MANIFEST_PATH = DEFAULT_OUTPUT_DIR / "smoke_manifest.json"
SCHEMA_VERSION = "dialam_incremental_patch_v1"
SELECTION_SEED = "dialam-qt30-smoke-v1"
BLOCK_SIZE = 8
TRAIN_TARGET_EXAMPLES = 160
EVAL_SCENARIO_COUNT = 30
HELDOUT_DIALOGUE_COUNT = 6
DIAGNOSTIC_SCENARIO_COUNT = 30


class PatchSplit(StrEnum):
    TRAIN = "train"
    EVAL = "eval"


class PatchProposition(StrictModel):
    id: str
    text: str = Field(min_length=1)
    chronological_turn: int = Field(gt=0)
    speaker_id: str | None = None
    speaker: str | None = None
    raw_locution_id: str
    raw_locution_text: str = Field(min_length=1)


class PatchRelation(StrictModel):
    source: str
    target: str
    type: NormalizedRelationLabel


class PatchGraph(StrictModel):
    relations: list[PatchRelation]

    @model_validator(mode="after")
    def reject_duplicates(self) -> "PatchGraph":
        keys = [(edge.source, edge.target, edge.type) for edge in self.relations]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate patch relations are not allowed")
        return self


class GoldRelationProvenance(StrictModel):
    relation_node_id: str
    original_relation_label: str
    normalized_relation_label: NormalizedRelationLabel
    original_source_proposition_ids: list[str]
    original_target_proposition_ids: list[str]
    original_edge_direction: str
    canonical_source_proposition_id: str
    canonical_target_proposition_id: str


class PatchExample(StrictModel):
    schema_version: Literal["dialam_incremental_patch_v1"]
    example_id: str
    update_id: str
    split: PatchSplit
    dialogue_id: str
    dialogue_title: str
    map_id: str
    block_index: int = Field(ge=0)
    block_count: int = Field(gt=0)
    block_size: int = Field(gt=0)
    new_proposition: PatchProposition
    earlier_propositions: list[PatchProposition] = Field(min_length=1)
    gold_patch: PatchGraph
    gold_relation_provenance: list[GoldRelationProvenance]

    @model_validator(mode="after")
    def validate_example_shape(self) -> "PatchExample":
        if self.block_index >= self.block_count:
            raise ValueError("block_index must be smaller than block_count")
        if len(self.earlier_propositions) > self.block_size:
            raise ValueError("comparison block exceeds fixed block size")
        earlier_ids = [item.id for item in self.earlier_propositions]
        if len(earlier_ids) != len(set(earlier_ids)):
            raise ValueError("comparison block contains duplicate proposition IDs")
        if any(item.chronological_turn >= self.new_proposition.chronological_turn for item in self.earlier_propositions):
            raise ValueError("comparison block contains a non-earlier proposition")
        for relation in self.gold_patch.relations:
            if relation.source != self.new_proposition.id:
                raise ValueError("gold relation source is not the new proposition")
            if relation.target not in earlier_ids:
                raise ValueError("gold relation target is absent from the comparison block")
        gold_keys = {
            (edge.source, edge.target, edge.type) for edge in self.gold_patch.relations
        }
        provenance_keys = {
            (
                item.canonical_source_proposition_id,
                item.canonical_target_proposition_id,
                item.normalized_relation_label,
            )
            for item in self.gold_relation_provenance
        }
        if gold_keys != provenance_keys:
            raise ValueError("gold patch and source relation provenance do not align")
        return self


class PatchScenario(StrictModel):
    schema_version: Literal["dialam_incremental_patch_v1"]
    scenario_id: str
    update_id: str
    split: Literal[PatchSplit.EVAL]
    dialogue_id: str
    dialogue_title: str
    map_id: str
    new_proposition: PatchProposition
    earlier_proposition_count: int = Field(gt=0)
    blocks: list[PatchExample] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_scenario(self) -> "PatchScenario":
        if self.scenario_id != self.update_id:
            raise ValueError("eval scenario ID must equal its update ID")
        if any(block.update_id != self.update_id for block in self.blocks):
            raise ValueError("eval scenario contains a block from another update")
        if any(block.split != PatchSplit.EVAL for block in self.blocks):
            raise ValueError("eval scenario contains a non-eval block")
        if self.earlier_proposition_count != sum(len(block.earlier_propositions) for block in self.blocks):
            raise ValueError("eval scenario earlier proposition count does not match its blocks")
        return self


@dataclass(frozen=True)
class UpdateCandidate:
    canonical_map: CanonicalMap
    new_proposition: CanonicalProposition
    earlier_propositions: tuple[CanonicalProposition, ...]
    gold_relations: tuple[CanonicalRelation, ...]

    @property
    def update_id(self) -> str:
        return (
            f"qt30:{self.canonical_map.dialogue_id}:{self.canonical_map.map_id}:"
            f"{self.new_proposition.proposition_id}"
        )

    @property
    def block_count(self) -> int:
        return math.ceil(len(self.earlier_propositions) / BLOCK_SIZE)

    @property
    def is_positive(self) -> bool:
        return bool(self.gold_relations)

    @property
    def labels(self) -> set[NormalizedRelationLabel]:
        return {item.normalized_relation_label for item in self.gold_relations}


def _selection_key(candidate: UpdateCandidate) -> str:
    return hashlib.sha256(f"{SELECTION_SEED}|{candidate.update_id}".encode()).hexdigest()


def _has_unresolved_chronology(canonical_map: CanonicalMap) -> bool:
    return any(issue.startswith("TA cycle or unresolved precedence") for issue in canonical_map.chronology_issues)


def _relation_touches_update(
    relation: CanonicalRelation,
    new_id: str,
    earlier_ids: set[str],
) -> bool:
    endpoints = {
        *relation.original_source_proposition_ids,
        *relation.original_target_proposition_ids,
    }
    return new_id in endpoints and bool(endpoints & earlier_ids)


def _build_behavior_compatible_update_candidates(
    maps: Iterable[CanonicalMap],
) -> list[UpdateCandidate]:
    candidates: list[UpdateCandidate] = []
    for canonical_map in maps:
        if _has_unresolved_chronology(canonical_map):
            continue
        if len(canonical_map.propositions) < 2:
            continue
        if any(not item.has_unambiguous_grounding for item in canonical_map.propositions):
            continue
        ordered = sorted(
            canonical_map.propositions,
            key=lambda item: (item.chronological_turn or 10**9, item.proposition_order),
        )
        for new_proposition in ordered:
            earlier = tuple(
                item
                for item in ordered
                if item.chronological_turn is not None
                and new_proposition.chronological_turn is not None
                and item.chronological_turn < new_proposition.chronological_turn
            )
            if not earlier:
                continue
            earlier_ids = {item.proposition_id for item in earlier}
            touching = [
                relation
                for relation in canonical_map.relations
                if _relation_touches_update(
                    relation,
                    new_proposition.proposition_id,
                    earlier_ids,
                )
            ]
            if any(not relation.is_unambiguous_binary for relation in touching):
                continue
            gold = tuple(
                sorted(
                    (
                        relation
                        for relation in touching
                        if relation.canonical_source_proposition_id == new_proposition.proposition_id
                        and relation.canonical_target_proposition_id in earlier_ids
                    ),
                    key=lambda item: (
                        item.canonical_target_proposition_id or "",
                        item.normalized_relation_label,
                        item.relation_node_id,
                    ),
                )
            )
            candidates.append(
                UpdateCandidate(
                    canonical_map=canonical_map,
                    new_proposition=new_proposition,
                    earlier_propositions=earlier,
                    gold_relations=gold,
                )
            )

    return candidates


def _deduplicate_update_candidates(
    candidates: Iterable[UpdateCandidate],
) -> list[UpdateCandidate]:

    # A proposition can be repeated in overlapping argument maps. Keep the map
    # with the most complete direct patch, then the widest earlier context.
    best_by_proposition: dict[tuple[str, str], UpdateCandidate] = {}
    for candidate in candidates:
        key = (
            candidate.canonical_map.dialogue_id,
            candidate.new_proposition.proposition_id,
        )
        current = best_by_proposition.get(key)
        candidate_rank = (
            len(candidate.gold_relations),
            len(candidate.earlier_propositions),
            -candidate.canonical_map.map_order,
            candidate.canonical_map.map_id,
        )
        current_rank = (
            len(current.gold_relations),
            len(current.earlier_propositions),
            -current.canonical_map.map_order,
            current.canonical_map.map_id,
        ) if current is not None else None
        if current is None or candidate_rank > current_rank:
            best_by_proposition[key] = candidate
    return sorted(best_by_proposition.values(), key=_selection_key)


def build_update_candidates(maps: Iterable[CanonicalMap]) -> list[UpdateCandidate]:
    return _deduplicate_update_candidates(
        _build_behavior_compatible_update_candidates(maps)
    )


def _patch_proposition(proposition: CanonicalProposition) -> PatchProposition:
    if not proposition.has_unambiguous_grounding:
        raise ValueError(f"proposition {proposition.proposition_id} has ambiguous grounding")
    grounding = proposition.groundings[0]
    return PatchProposition(
        id=proposition.proposition_id,
        text=proposition.proposition_text,
        chronological_turn=grounding.chronological_turn,
        speaker_id=grounding.speaker_id,
        speaker=grounding.speaker,
        raw_locution_id=grounding.raw_locution_id,
        raw_locution_text=grounding.raw_locution_text,
    )


def _materialize_candidate(candidate: UpdateCandidate, split: PatchSplit) -> list[PatchExample]:
    blocks = [
        candidate.earlier_propositions[index : index + BLOCK_SIZE]
        for index in range(0, len(candidate.earlier_propositions), BLOCK_SIZE)
    ]
    examples: list[PatchExample] = []
    for block_index, block in enumerate(blocks):
        block_ids = {item.proposition_id for item in block}
        block_relations = [
            relation
            for relation in candidate.gold_relations
            if relation.canonical_target_proposition_id in block_ids
        ]
        gold_keys = sorted(
            {
                (
                    relation.canonical_source_proposition_id,
                    relation.canonical_target_proposition_id,
                    relation.normalized_relation_label,
                )
                for relation in block_relations
            },
            key=lambda item: (item[1] or "", item[2]),
        )
        provenance_by_key: dict[tuple[str, str, NormalizedRelationLabel], CanonicalRelation] = {}
        for relation in block_relations:
            key = (
                relation.canonical_source_proposition_id or "",
                relation.canonical_target_proposition_id or "",
                relation.normalized_relation_label,
            )
            provenance_by_key.setdefault(key, relation)
        examples.append(
            PatchExample(
                schema_version=SCHEMA_VERSION,
                example_id=f"{candidate.update_id}:b{block_index + 1:02d}",
                update_id=candidate.update_id,
                split=split,
                dialogue_id=candidate.canonical_map.dialogue_id,
                dialogue_title=candidate.canonical_map.dialogue_title,
                map_id=candidate.canonical_map.map_id,
                block_index=block_index,
                block_count=len(blocks),
                block_size=BLOCK_SIZE,
                new_proposition=_patch_proposition(candidate.new_proposition),
                earlier_propositions=[_patch_proposition(item) for item in block],
                gold_patch=PatchGraph(
                    relations=[
                        PatchRelation(source=source or "", target=target or "", type=label)
                        for source, target, label in gold_keys
                    ]
                ),
                gold_relation_provenance=[
                    GoldRelationProvenance(
                        relation_node_id=relation.relation_node_id,
                        original_relation_label=relation.original_relation_label,
                        normalized_relation_label=relation.normalized_relation_label,
                        original_source_proposition_ids=relation.original_source_proposition_ids,
                        original_target_proposition_ids=relation.original_target_proposition_ids,
                        original_edge_direction=relation.original_edge_direction,
                        canonical_source_proposition_id=relation.canonical_source_proposition_id or "",
                        canonical_target_proposition_id=relation.canonical_target_proposition_id or "",
                    )
                    for _, relation in sorted(
                        provenance_by_key.items(),
                        key=lambda item: (item[0][1], item[0][2], item[1].relation_node_id),
                    )
                ],
            )
        )
    return examples


def _select_train_candidates(
    candidates: list[UpdateCandidate],
    *,
    target_examples: int = TRAIN_TARGET_EXAMPLES,
) -> list[UpdateCandidate]:
    positive = [item for item in candidates if item.is_positive]
    negative = [item for item in candidates if not item.is_positive]
    positive.sort(key=_selection_key)
    negative.sort(key=_selection_key)
    queues = [positive, negative]
    indexes = [0, 0]
    selected: list[UpdateCandidate] = []
    example_count = 0
    turn = 0
    while example_count < target_examples and any(indexes[i] < len(queues[i]) for i in range(2)):
        queue_index = turn % 2
        turn += 1
        if indexes[queue_index] >= len(queues[queue_index]):
            queue_index = 1 - queue_index
        candidate = queues[queue_index][indexes[queue_index]]
        indexes[queue_index] += 1
        if example_count + candidate.block_count > target_examples:
            continue
        selected.append(candidate)
        example_count += candidate.block_count
    if not 100 <= example_count <= 200:
        raise ValueError(f"train smoke dataset has invalid size: {example_count}")
    return selected


def _select_eval_candidates(
    candidates: list[UpdateCandidate],
    heldout_dialogue_ids: list[str],
) -> list[UpdateCandidate]:
    by_dialogue: dict[str, list[UpdateCandidate]] = defaultdict(list)
    for candidate in candidates:
        if candidate.canonical_map.dialogue_id in heldout_dialogue_ids:
            by_dialogue[candidate.canonical_map.dialogue_id].append(candidate)

    per_dialogue = EVAL_SCENARIO_COUNT // len(heldout_dialogue_ids)
    selected: list[UpdateCandidate] = []
    for dialogue_id in heldout_dialogue_ids:
        dialogue_candidates = sorted(by_dialogue[dialogue_id], key=_selection_key)
        positive = [item for item in dialogue_candidates if item.is_positive]
        negative = [item for item in dialogue_candidates if not item.is_positive]
        chosen: list[UpdateCandidate] = []
        chosen.extend(positive[:3])
        chosen.extend(negative[:2])
        if len(chosen) < per_dialogue:
            remaining = [item for item in dialogue_candidates if item not in chosen]
            chosen.extend(remaining[: per_dialogue - len(chosen)])
        if len(chosen) != per_dialogue:
            raise ValueError(f"not enough eval candidates for dialogue {dialogue_id}")
        selected.extend(chosen)
    if len(selected) != EVAL_SCENARIO_COUNT:
        raise ValueError(f"expected {EVAL_SCENARIO_COUNT} eval scenarios, got {len(selected)}")
    return selected


def _select_balanced_diagnostic_candidates(
    candidates: list[UpdateCandidate],
    heldout_dialogue_ids: list[str],
) -> list[UpdateCandidate]:
    """Select 30 one-block scenarios: 8 per relation label and 6 negatives.

    Five scenarios come from each frozen held-out parent episode. Rotating the
    doubled label across episodes produces aggregate quotas of 8 SUPPORT,
    8 ATTACK, 8 REPHRASE, and 6 all-negative scenarios.
    """

    labels = [
        NormalizedRelationLabel.SUPPORT,
        NormalizedRelationLabel.ATTACK,
        NormalizedRelationLabel.REPHRASE,
    ]
    selected: list[UpdateCandidate] = []
    for dialogue_index, dialogue_id in enumerate(heldout_dialogue_ids):
        dialogue_candidates = [
            item
            for item in candidates
            if item.canonical_map.dialogue_id == dialogue_id and item.block_count == 1
        ]
        doubled_label = labels[dialogue_index % len(labels)]
        quotas = {label: 1 + int(label == doubled_label) for label in labels}
        for label in labels:
            eligible = sorted(
                (
                    item
                    for item in dialogue_candidates
                    if len(item.gold_relations) == 1 and item.labels == {label}
                ),
                key=_selection_key,
            )
            if len(eligible) < quotas[label]:
                raise ValueError(
                    f"not enough {label.value} diagnostic candidates for {dialogue_id}"
                )
            selected.extend(eligible[: quotas[label]])
        negatives = sorted(
            (item for item in dialogue_candidates if not item.is_positive),
            key=_selection_key,
        )
        if not negatives:
            raise ValueError(f"not enough negative diagnostic candidates for {dialogue_id}")
        selected.append(negatives[0])

    if len(selected) != DIAGNOSTIC_SCENARIO_COUNT:
        raise ValueError(
            f"expected {DIAGNOSTIC_SCENARIO_COUNT} diagnostic scenarios, got {len(selected)}"
        )
    return selected


def _read_jsonl(path: Path, model: type[BaseModel]) -> list[BaseModel]:
    rows: list[BaseModel] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(model.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"invalid {path.name} line {line_number}: {exc}") from exc
    return rows


ModelT = TypeVar("ModelT", bound=BaseModel)


def load_jsonl(path: Path, model: type[ModelT]) -> list[ModelT]:
    return [item for item in _read_jsonl(path, model)]  # type: ignore[misc]


def _write_jsonl(path: Path, rows: Iterable[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row.model_dump_json() + "\n")


def audit_corpus(maps: list[CanonicalMap], source_dir: Path = DEFAULT_SOURCE_DIR) -> dict[str, Any]:
    node_occurrences = Counter()
    unique_node_ids: dict[str, set[str]] = defaultdict(set)
    raw_locution_records = 0
    unique_locution_ids: set[str] = set()
    unique_valid_locution_ids: set[str] = set()
    invalid_locution_metadata_records = 0
    raw_edge_count = 0
    for path in sorted(source_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_edge_count += len(payload["edges"])
        raw_locution_records += len(payload["locutions"])
        unique_locution_ids.update(_as_string(item["nodeID"]) for item in payload["locutions"])
        node_type_by_id = {_as_string(node["nodeID"]): node["type"] for node in payload["nodes"]}
        for item in payload["locutions"]:
            locution_id = _as_string(item["nodeID"])
            if node_type_by_id.get(locution_id) == "L":
                unique_valid_locution_ids.add(locution_id)
            else:
                invalid_locution_metadata_records += 1
        for node in payload["nodes"]:
            node_type = node["type"]
            node_occurrences[node_type] += 1
            unique_node_ids[node_type].add(_as_string(node["nodeID"]))

    relations = [relation for item in maps for relation in item.relations]
    propositions = [proposition for item in maps for proposition in item.propositions]
    relation_arity = Counter(
        (
            len(relation.original_source_proposition_ids),
            len(relation.original_target_proposition_ids),
        )
        for relation in relations
    )
    issue_counts = Counter(
        issue.split(":", 1)[0]
        for item in maps
        for issue in item.chronology_issues
    )
    return {
        "inspected_map_count": len(maps),
        "dialogue_count": len({item.dialogue_id for item in maps}),
        "maps_by_dialogue": dict(sorted(Counter(item.dialogue_id for item in maps).items())),
        "raw_node_occurrences_by_type": dict(sorted(node_occurrences.items())),
        "unique_node_ids_by_type": {
            key: len(value) for key, value in sorted(unique_node_ids.items())
        },
        "raw_edge_count": raw_edge_count,
        "raw_locution_record_count": raw_locution_records,
        "unique_locution_metadata_ids": len(unique_locution_ids),
        "unique_valid_locution_ids": len(unique_valid_locution_ids),
        "invalid_locution_metadata_record_count": invalid_locution_metadata_records,
        "proposition_occurrence_count": len(propositions),
        "unique_proposition_ids": len({item.proposition_id for item in propositions}),
        "groundings_per_proposition_occurrence": dict(
            sorted(Counter(len(item.groundings) for item in propositions).items())
        ),
        "relation_counts_by_original_type": dict(
            sorted(Counter(item.relation_node_type for item in relations).items())
        ),
        "relation_counts_by_normalized_label": dict(
            sorted(Counter(item.normalized_relation_label for item in relations).items())
        ),
        "proposition_endpoint_arity": {
            f"{source_count}x{target_count}": count
            for (source_count, target_count), count in sorted(relation_arity.items())
        },
        "binary_direct_relation_count": sum(item.is_binary_direct for item in relations),
        "nary_relation_count": sum(
            len(item.original_source_proposition_ids) > 1
            or len(item.original_target_proposition_ids) > 1
            for item in relations
        ),
        "incomplete_relation_count": sum(
            not item.original_source_proposition_ids
            or not item.original_target_proposition_ids
            for item in relations
        ),
        "unambiguous_binary_relation_count": sum(item.is_unambiguous_binary for item in relations),
        "binary_original_edge_direction": dict(
            sorted(
                Counter(
                    item.original_edge_direction
                    for item in relations
                    if item.is_binary_direct
                ).items()
            )
        ),
        "chronology_issue_counts": dict(sorted(issue_counts.items())),
    }


def _as_string(value: Any) -> str:
    return str(value)


def _label_counts(examples: Iterable[PatchExample]) -> dict[str, int]:
    return dict(
        sorted(
            Counter(
                edge.type
                for example in examples
                for edge in example.gold_patch.relations
            ).items()
        )
    )


def _dataset_stats(
    train_examples: list[PatchExample],
    eval_scenarios: list[PatchScenario],
    eval_examples: list[PatchExample],
) -> dict[str, Any]:
    return {
        "train_example_count": len(train_examples),
        "train_update_count": len({item.update_id for item in train_examples}),
        "train_dialogue_count": len({item.dialogue_id for item in train_examples}),
        "train_positive_block_count": sum(bool(item.gold_patch.relations) for item in train_examples),
        "train_all_negative_block_count": sum(not item.gold_patch.relations for item in train_examples),
        "train_relation_counts": _label_counts(train_examples),
        "eval_scenario_count": len(eval_scenarios),
        "eval_example_count": len(eval_examples),
        "eval_dialogue_count": len({item.dialogue_id for item in eval_scenarios}),
        "eval_positive_block_count": sum(bool(item.gold_patch.relations) for item in eval_examples),
        "eval_all_negative_block_count": sum(not item.gold_patch.relations for item in eval_examples),
        "eval_relation_counts": _label_counts(eval_examples),
    }


def _candidate_stats(candidates: list[UpdateCandidate]) -> dict[str, Any]:
    relation_counts = Counter(
        relation.normalized_relation_label.value
        for candidate in candidates
        for relation in candidate.gold_relations
    )
    label_scenario_counts = Counter(
        label.value
        for candidate in candidates
        for label in candidate.labels
    )
    return {
        "scenario_count": len(candidates),
        "block_example_count": sum(item.block_count for item in candidates),
        "positive_scenario_count": sum(item.is_positive for item in candidates),
        "all_negative_scenario_count": sum(not item.is_positive for item in candidates),
        "relation_counts": dict(sorted(relation_counts.items())),
        "scenarios_containing_label": dict(sorted(label_scenario_counts.items())),
        "dialogue_ids": sorted({item.canonical_map.dialogue_id for item in candidates}),
    }


def _relation_identity(relation: CanonicalRelation) -> tuple[Any, ...]:
    return (
        relation.dialogue_id,
        tuple(relation.original_source_proposition_ids),
        relation.relation_node_id,
        tuple(relation.original_target_proposition_ids),
    )


def _funnel_stage(relations: Iterable[CanonicalRelation]) -> dict[str, Any]:
    rows = list(relations)
    return {
        "total": len(rows),
        "by_label": dict(
            sorted(Counter(item.normalized_relation_label.value for item in rows).items())
        ),
    }


def _relation_retention_funnel(
    maps: list[CanonicalMap],
    compatible_candidates: list[UpdateCandidate],
    retained_candidates: list[UpdateCandidate],
) -> dict[str, Any]:
    raw = [relation for canonical_map in maps for relation in canonical_map.relations]
    unique_by_identity: dict[tuple[Any, ...], CanonicalRelation] = {}
    for relation in raw:
        unique_by_identity.setdefault(_relation_identity(relation), relation)
    unique = list(unique_by_identity.values())
    binary = [item for item in unique if item.is_binary_direct]
    proposition_by_map = {
        (canonical_map.dialogue_id, canonical_map.map_id): {
            item.proposition_id: item for item in canonical_map.propositions
        }
        for canonical_map in maps
    }
    map_by_key = {
        (canonical_map.dialogue_id, canonical_map.map_id): canonical_map
        for canonical_map in maps
    }

    def endpoints(relation: CanonicalRelation) -> tuple[CanonicalProposition, CanonicalProposition]:
        propositions = proposition_by_map[(relation.dialogue_id, relation.map_id)]
        return (
            propositions[relation.original_source_proposition_ids[0]],
            propositions[relation.original_target_proposition_ids[0]],
        )

    grounded = [
        item
        for item in binary
        if all(len(proposition.groundings) == 1 for proposition in endpoints(item))
    ]
    chronology_resolvable = [
        item
        for item in grounded
        if not _has_unresolved_chronology(map_by_key[(item.dialogue_id, item.map_id)])
        and endpoints(item)[0].chronological_turn != endpoints(item)[1].chronological_turn
    ]
    behavior_ids = {
        _relation_identity(relation)
        for candidate in compatible_candidates
        for relation in candidate.gold_relations
    }
    retained_ids = {
        _relation_identity(relation)
        for candidate in retained_candidates
        for relation in candidate.gold_relations
    }
    behavior_compatible = [item for item in chronology_resolvable if _relation_identity(item) in behavior_ids]
    retained = [item for item in behavior_compatible if _relation_identity(item) in retained_ids]
    stages = {
        "raw": _funnel_stage(raw),
        "unique": _funnel_stage(unique),
        "binary": _funnel_stage(binary),
        "grounded": _funnel_stage(grounded),
        "chronology_resolvable": _funnel_stage(chronology_resolvable),
        "behavior_compatible": _funnel_stage(behavior_compatible),
        "retained": _funnel_stage(retained),
    }
    previous_total: int | None = None
    for stage in stages.values():
        stage["removed_from_previous"] = (
            0 if previous_total is None else previous_total - stage["total"]
        )
        previous_total = stage["total"]
    return {
        "identity_key": [
            "parent_dialogue_id",
            "ordered_original_source_proposition_ids",
            "original_relation_node_id",
            "ordered_original_target_proposition_ids",
        ],
        "published_relation_count": 10818,
        "observed_unique_minus_published": stages["unique"]["total"] - 10818,
        "stages": stages,
    }


def _grounding_and_chronology_defects(maps: list[CanonicalMap]) -> dict[str, Any]:
    binary_relations = [
        relation
        for canonical_map in maps
        for relation in canonical_map.relations
        if relation.is_binary_direct
    ]
    proposition_by_map = {
        (canonical_map.dialogue_id, canonical_map.map_id): {
            item.proposition_id: item for item in canonical_map.propositions
        }
        for canonical_map in maps
    }
    map_by_key = {
        (canonical_map.dialogue_id, canonical_map.map_id): canonical_map
        for canonical_map in maps
    }
    grounding = Counter()
    chronology = Counter()
    for relation in binary_relations:
        propositions = proposition_by_map[(relation.dialogue_id, relation.map_id)]
        endpoints = (
            propositions[relation.original_source_proposition_ids[0]],
            propositions[relation.original_target_proposition_ids[0]],
        )
        grounding_counts = [len(item.groundings) for item in endpoints]
        if all(count == 1 for count in grounding_counts):
            grounding["both_endpoints_exactly_one"] += 1
            canonical_map = map_by_key[(relation.dialogue_id, relation.map_id)]
            if _has_unresolved_chronology(canonical_map):
                chronology["excluded_unresolved_map"] += 1
            elif endpoints[0].chronological_turn == endpoints[1].chronological_turn:
                chronology["excluded_same_turn"] += 1
            else:
                chronology["resolvable"] += 1
        elif any(count == 0 for count in grounding_counts):
            grounding["at_least_one_endpoint_missing"] += 1
        else:
            grounding["ambiguous_only_no_missing_endpoint"] += 1

    issue_prefixes = {
        "maps_with_unresolved_ta_precedence": "TA cycle or unresolved precedence",
        "maps_with_duplicate_locution_metadata": "duplicate locution metadata IDs deduplicated",
        "maps_with_non_l_locution_metadata": "non-L nodes ignored in locution metadata",
        "maps_with_unknown_locution_metadata": "unknown nodes ignored in locution metadata",
    }
    return {
        "binary_relation_grounding": dict(sorted(grounding.items())),
        "singly_grounded_relation_chronology": dict(sorted(chronology.items())),
        **{
            name: sum(
                any(issue.startswith(prefix) for issue in canonical_map.chronology_issues)
                for canonical_map in maps
            )
            for name, prefix in issue_prefixes.items()
        },
        "exclusion_policy": {
            "map": (
                "exclude a map if TA precedence is cyclic/unresolved, it has fewer than two "
                "propositions, or any proposition lacks exactly one direct L->YA->I grounding"
            ),
            "update": (
                "exclude an update if any relation touching the new proposition and an earlier "
                "proposition is not an unambiguous binary direct relation"
            ),
            "relation": (
                "retain only direct binary I->RA/CA/MA->I relations whose singly grounded "
                "endpoints have distinct resolvable turns; normalize the later endpoint as source"
            ),
        },
    }


def _materialize_scenario(candidate: UpdateCandidate) -> PatchScenario:
    return PatchScenario(
        schema_version=SCHEMA_VERSION,
        scenario_id=candidate.update_id,
        update_id=candidate.update_id,
        split=PatchSplit.EVAL,
        dialogue_id=candidate.canonical_map.dialogue_id,
        dialogue_title=candidate.canonical_map.dialogue_title,
        map_id=candidate.canonical_map.map_id,
        new_proposition=_patch_proposition(candidate.new_proposition),
        earlier_proposition_count=len(candidate.earlier_propositions),
        blocks=_materialize_candidate(candidate, PatchSplit.EVAL),
    )


def validate_patch_datasets(
    train_examples: list[PatchExample],
    eval_scenarios: list[PatchScenario],
    maps: list[CanonicalMap],
) -> dict[str, Any]:
    eval_examples = [block for scenario in eval_scenarios for block in scenario.blocks]
    all_examples = [*train_examples, *eval_examples]
    map_by_id = {item.map_id: item for item in maps}
    if len(map_by_id) != len(maps):
        raise ValueError("canonical map IDs are not unique")
    train_dialogues = {item.dialogue_id for item in train_examples}
    eval_dialogues = {item.dialogue_id for item in eval_scenarios}
    if train_dialogues & eval_dialogues:
        raise ValueError(f"dialogue-level leakage: {sorted(train_dialogues & eval_dialogues)}")

    by_update: dict[str, list[PatchExample]] = defaultdict(list)
    for example in all_examples:
        by_update[example.update_id].append(example)
        canonical_map = map_by_id.get(example.map_id)
        if canonical_map is None or canonical_map.dialogue_id != example.dialogue_id:
            raise ValueError(f"unknown or mismatched source map for {example.example_id}")
        proposition_by_id = {item.proposition_id: item for item in canonical_map.propositions}
        relation_by_id = {item.relation_node_id: item for item in canonical_map.relations}
        expected_new = proposition_by_id.get(example.new_proposition.id)
        if expected_new is None:
            raise ValueError(f"unknown new proposition in {example.example_id}")
        if _patch_proposition(expected_new) != example.new_proposition:
            raise ValueError(f"new proposition grounding mismatch in {example.example_id}")
        direct_relations = {
            (
                relation.canonical_source_proposition_id,
                relation.canonical_target_proposition_id,
                relation.normalized_relation_label,
            )
            for relation in canonical_map.relations
            if relation.is_unambiguous_binary
        }
        gold_keys = {
            (edge.source, edge.target, edge.type) for edge in example.gold_patch.relations
        }
        for provenance in example.gold_relation_provenance:
            source_relation = relation_by_id.get(provenance.relation_node_id)
            if source_relation is None:
                raise ValueError(f"unknown source relation in {example.example_id}")
            expected_provenance = GoldRelationProvenance(
                relation_node_id=source_relation.relation_node_id,
                original_relation_label=source_relation.original_relation_label,
                normalized_relation_label=source_relation.normalized_relation_label,
                original_source_proposition_ids=source_relation.original_source_proposition_ids,
                original_target_proposition_ids=source_relation.original_target_proposition_ids,
                original_edge_direction=source_relation.original_edge_direction,
                canonical_source_proposition_id=source_relation.canonical_source_proposition_id or "",
                canonical_target_proposition_id=source_relation.canonical_target_proposition_id or "",
            )
            if provenance != expected_provenance:
                raise ValueError(f"source edge direction or label mismatch in {example.example_id}")
            if provenance.canonical_source_proposition_id != example.new_proposition.id:
                raise ValueError(f"canonical edge source mismatch in {example.example_id}")
        for earlier in example.earlier_propositions:
            expected = proposition_by_id.get(earlier.id)
            if expected is None or _patch_proposition(expected) != earlier:
                raise ValueError(f"earlier proposition grounding mismatch in {example.example_id}")
            pair_relations = {
                relation
                for relation in direct_relations
                if relation[0] == example.new_proposition.id and relation[1] == earlier.id
            }
            gold_pair_relations = {
                relation for relation in gold_keys if relation[1] == earlier.id
            }
            if pair_relations != gold_pair_relations:
                raise ValueError(f"direct-edge negative or gold mismatch in {example.example_id}")

    for update_id, examples in by_update.items():
        examples.sort(key=lambda item: item.block_index)
        expected_indexes = list(range(examples[0].block_count))
        if [item.block_index for item in examples] != expected_indexes:
            raise ValueError(f"incomplete or duplicate blocks for {update_id}")
        if any(item.block_count != len(examples) for item in examples):
            raise ValueError(f"inconsistent block count for {update_id}")
        earlier_ids = [
            proposition.id
            for example in examples
            for proposition in example.earlier_propositions
        ]
        if len(earlier_ids) != len(set(earlier_ids)):
            raise ValueError(f"earlier proposition appears in multiple blocks for {update_id}")
        canonical_map = map_by_id[examples[0].map_id]
        new_turn = examples[0].new_proposition.chronological_turn
        expected_ids = {
            item.proposition_id
            for item in canonical_map.propositions
            if item.chronological_turn is not None and item.chronological_turn < new_turn
        }
        if set(earlier_ids) != expected_ids:
            raise ValueError(f"block coverage is incomplete for {update_id}")
        for example in examples[:-1]:
            if len(example.earlier_propositions) != BLOCK_SIZE:
                raise ValueError(f"non-final block is not fixed-size for {update_id}")

    validation = {
        "split_unit": "original_parent_episode",
        "parent_episode_level_split": True,
        "chronology": True,
        "ids": True,
        "edge_direction": True,
        "grounding": True,
        "block_coverage": True,
        "gold_target_presence": True,
        "direct_edge_negatives": True,
        "dialogue_level_leakage": False,
        "train_dialogue_ids": sorted(train_dialogues),
        "eval_dialogue_ids": sorted(eval_dialogues),
    }
    return validation


def build_smoke_dataset(
    *,
    source_dir: Path = DEFAULT_SOURCE_DIR,
    dialogues_path: Path = DEFAULT_DIALOGUES_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    maps = load_canonical_maps(source_dir, dialogues_path)
    compatible_candidates = _build_behavior_compatible_update_candidates(maps)
    candidates = _deduplicate_update_candidates(compatible_candidates)
    dialogue_order = {
        item.dialogue_id: item.dialogue_order for item in maps
    }
    ordered_dialogues = [
        item[0] for item in sorted(dialogue_order.items(), key=lambda item: item[1])
    ]
    heldout_dialogue_ids = ordered_dialogues[-HELDOUT_DIALOGUE_COUNT:]
    train_candidates = [
        item for item in candidates if item.canonical_map.dialogue_id not in heldout_dialogue_ids
    ]
    selected_train = _select_train_candidates(train_candidates)
    selected_eval = _select_eval_candidates(candidates, heldout_dialogue_ids)
    selected_diagnostic = _select_balanced_diagnostic_candidates(
        candidates,
        heldout_dialogue_ids,
    )

    train_examples = [
        example
        for candidate in selected_train
        for example in _materialize_candidate(candidate, PatchSplit.TRAIN)
    ]
    eval_scenarios = [_materialize_scenario(candidate) for candidate in selected_eval]
    eval_examples = [block for scenario in eval_scenarios for block in scenario.blocks]
    diagnostic_scenarios = [
        _materialize_scenario(candidate) for candidate in selected_diagnostic
    ]
    diagnostic_examples = [
        block for scenario in diagnostic_scenarios for block in scenario.blocks
    ]
    validation = validate_patch_datasets(train_examples, eval_scenarios, maps)
    diagnostic_validation = validate_patch_datasets(
        train_examples,
        diagnostic_scenarios,
        maps,
    )
    stats = _dataset_stats(train_examples, eval_scenarios, eval_examples)
    heldout_candidates = [
        item
        for item in candidates
        if item.canonical_map.dialogue_id in heldout_dialogue_ids
    ]
    eval_summary = {
        "frozen_split_unit": "original QT30 parent episode/subcorpus",
        "heldout_dialogue_ids": heldout_dialogue_ids,
        "natural_distribution": _candidate_stats(heldout_candidates),
        "balanced_diagnostic": _candidate_stats(selected_diagnostic),
        "balanced_diagnostic_selection": {
            "scenario_count": DIAGNOSTIC_SCENARIO_COUNT,
            "per_dialogue_scenario_count": 5,
            "requirements": (
                "one block per scenario; exactly one gold relation for each positive; "
                "8 SUPPORT, 8 ATTACK, 8 REPHRASE, and 6 all-negative scenarios"
            ),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / DEFAULT_TRAIN_PATH.name, train_examples)
    _write_jsonl(output_dir / DEFAULT_EVAL_SCENARIOS_PATH.name, eval_scenarios)
    _write_jsonl(output_dir / DEFAULT_EVAL_EXAMPLES_PATH.name, eval_examples)
    _write_jsonl(
        output_dir / DEFAULT_DIAGNOSTIC_EVAL_SCENARIOS_PATH.name,
        diagnostic_scenarios,
    )
    _write_jsonl(
        output_dir / DEFAULT_DIAGNOSTIC_EVAL_EXAMPLES_PATH.name,
        diagnostic_examples,
    )
    (output_dir / DEFAULT_EVAL_SUMMARY_PATH.name).write_text(
        json.dumps(eval_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    corpus_audit = audit_corpus(maps, source_dir)
    corpus_audit["relation_retention_funnel"] = _relation_retention_funnel(
        maps,
        compatible_candidates,
        candidates,
    )
    corpus_audit["grounding_and_chronology_defects"] = (
        _grounding_and_chronology_defects(maps)
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "selection_seed": SELECTION_SEED,
        "block_size": BLOCK_SIZE,
        "source_archive_sha256": "9d3d70c5ad86c815f821928dc515ee80a8e2a24d3f5d224557045f50f7a73791",
        "permission_basis": (
            "Project owner reported direct approval for this educational toy-project use on 2026-08-22; "
            "this is not represented as an official public corpus license."
        ),
        "filter": {
            "comparison_scope": "all strictly earlier propositions in one complete source argument map",
            "split_scope": "original parent episode-level QT30 subcorpora, never individual map IDs",
            "grounding": "exactly one direct L->YA->I grounding for every proposition in a selected map",
            "relations": "only unambiguous binary direct I->RA/CA/MA->I proposition endpoints",
            "candidate_retrieval": "none; deterministic chronological fixed-size blocks",
        },
        "statistics": stats,
        "validation": validation,
        "diagnostic_validation": diagnostic_validation,
        "eval_summary": eval_summary,
        "corpus_audit": corpus_audit,
        "paths": {
            "train_examples": DEFAULT_TRAIN_PATH.name,
            "eval_scenarios": DEFAULT_EVAL_SCENARIOS_PATH.name,
            "eval_examples": DEFAULT_EVAL_EXAMPLES_PATH.name,
            "balanced_diagnostic_eval_scenarios": DEFAULT_DIAGNOSTIC_EVAL_SCENARIOS_PATH.name,
            "balanced_diagnostic_eval_examples": DEFAULT_DIAGNOSTIC_EVAL_EXAMPLES_PATH.name,
            "eval_summary": DEFAULT_EVAL_SUMMARY_PATH.name,
        },
    }
    (output_dir / DEFAULT_MANIFEST_PATH.name).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_generated_dataset(
    train_path: Path = DEFAULT_TRAIN_PATH,
    eval_scenarios_path: Path = DEFAULT_EVAL_SCENARIOS_PATH,
) -> tuple[list[PatchExample], list[PatchScenario]]:
    return (
        load_jsonl(train_path, PatchExample),
        load_jsonl(eval_scenarios_path, PatchScenario),
    )

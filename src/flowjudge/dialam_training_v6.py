from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Literal

from pydantic import Field, model_validator

from .dialam_training import DEFAULT_TRAINING_DIR, TrainingMessage, file_sha256
from .dialam_training_v2 import _content_tokens
from .dialam_training_v3 import _seeded_hash
from .dialam_training_v5 import (
    CandidateHardness,
    DialAMTrainingRowV5,
    PairwiseLabel,
    V5_MANIFEST_FILENAME,
    V5_OUTPUT_FILENAME,
    V5_PROMPT_PATH,
)
from .patch_data import PROJECT_ROOT
from .schemas import StrictModel


V6_TRAINING_SEED = "dialam-v6-vives-warmup-then-qt30-pairwise"
V6_VIVES_WARMUP_SIZE = 4096
V6_VIVES_PAIR_COUNT = V6_VIVES_WARMUP_SIZE // 2
V6_QT30_TARGET_SIZE = 8192
V6_TRAINING_SIZE = V6_VIVES_WARMUP_SIZE + V6_QT30_TARGET_SIZE
V6_VIVES_POSITIVE_TARGET_COUNTS: dict[str, int] = {
    "ATTACK": 683,
    "REPHRASE": 683,
    "SUPPORT": 682,
}
V6_OUTPUT_FILENAME = "dialam_v6_n12288.jsonl"
V6_MANIFEST_FILENAME = "training_v6_manifest.json"
V6_SCHEMA_FILENAME = "training_example_v6.schema.json"
DEFAULT_VIVES_SOURCE_DIR = PROJECT_ROOT / "data" / "source" / "vivesdebate"
VIVES_MANIFEST_FILENAME = "manifest.json"
RELATION_LABELS: dict[str, PairwiseLabel] = {
    "RA": "SUPPORT",
    "CA": "ATTACK",
    "MA": "REPHRASE",
}
POSITIVE_LABELS: tuple[PairwiseLabel, ...] = ("ATTACK", "REPHRASE", "SUPPORT")
VIVES_BLOCK_SIZE = 8


class DialAMTrainingRowV6(StrictModel):
    schema_version: Literal["dialam_qlora_pairwise_v6"]
    training_row_id: str
    pair_group_id: str
    pair_role: Literal["POSITIVE", "NONE"]
    source_corpus: Literal["VivesDebate", "DialAM-QT30"]
    curriculum_stage: Literal["vives_warmup", "qt30_target"]
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
    def validate_pairwise_row(self) -> "DialAMTrainingRowV6":
        expected_role = "NONE" if self.label == "NONE" else "POSITIVE"
        if self.pair_role != expected_role:
            raise ValueError("pair role and label disagree")
        if self.messages[1].role != "assistant" or self.messages[1].content != self.label:
            raise ValueError("assistant target must be the exact pairwise label")
        if self.candidate_target_id == self.paired_candidate_target_id:
            raise ValueError("contrast pair must use two different candidate IDs")
        return self


@dataclass(frozen=True)
class VivesUnit:
    debate_id: str
    source_id: int
    speaker: str
    text: str

    @property
    def id(self) -> str:
        return str(self.source_id)


@dataclass(frozen=True)
class VivesRelation:
    debate_id: str
    source_id: int
    target_id: int
    label: PairwiseLabel
    source_label: str
    source_column: str

    @property
    def key(self) -> str:
        return (
            f"{self.debate_id}|{self.source_id}|{self.target_id}|{self.label}|"
            f"{self.source_column}"
        )


@dataclass(frozen=True)
class VivesCorpus:
    units_by_debate: dict[str, dict[int, VivesUnit]]
    relations: tuple[VivesRelation, ...]
    audit: dict[str, object]
    source_hashes: dict[str, str]


@dataclass(frozen=True)
class VivesContrast:
    pair_group_id: str
    relation: VivesRelation
    negative_target_id: int
    block_ids: tuple[int, ...]
    repetition_index: int
    negative_hardness: CandidateHardness


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relation_columns(fieldnames: list[str]) -> list[tuple[str, str]]:
    columns: list[tuple[str, str]] = []
    for field in fieldnames:
        match = re.fullmatch(r"RELATED ID(?:\.([0-9]+))?", field)
        if not match:
            continue
        suffix = f".{match.group(1)}" if match.group(1) else ""
        columns.append((field, f"ARGUMENTAL RELATION  TYPE{suffix}"))
    return columns


def load_vives_binary_corpus(
    source_dir: Path = DEFAULT_VIVES_SOURCE_DIR,
) -> VivesCorpus:
    manifest_path = source_dir / VIVES_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("version") != "ver.3" or manifest.get("license") != "CC BY-NC-SA 4.0":
        raise ValueError("unexpected VivesDebate source version or license")

    units_by_debate: dict[str, dict[int, VivesUnit]] = {}
    relations: list[VivesRelation] = []
    source_hashes = {VIVES_MANIFEST_FILENAME: file_sha256(manifest_path)}
    audit = Counter()

    for item in manifest["selected_debates"]:
        debate_id = str(item["debate_id"])
        path = source_dir / str(item["file"])
        if _md5(path) != item["md5"]:
            raise ValueError(f"VivesDebate checksum mismatch: {path.name}")
        source_hashes[path.name] = file_sha256(path)
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fieldnames = list(reader.fieldnames or [])
        relation_columns = _relation_columns(fieldnames)
        units: dict[int, VivesUnit] = {}
        rows_by_id: dict[int, dict[str, str]] = {}
        for row in rows:
            source_id = int(float(row["ID (Chronological)"]))
            if source_id in units:
                raise ValueError(f"duplicate VivesDebate ID {debate_id}:{source_id}")
            # Spanish is the original fluent transcript. Qwen3 is multilingual,
            # and the final QT30 stage returns the curriculum to English.
            text = _clean_text(row.get("ADU_ES", "") or row.get("ADU_CAT", ""))
            speaker = _clean_text(row.get("TEAM STANCE", "")).upper()
            if not text or speaker not in {"FAVOUR", "AGAINST"}:
                raise ValueError(f"invalid VivesDebate unit {debate_id}:{source_id}")
            units[source_id] = VivesUnit(debate_id, source_id, speaker, text)
            rows_by_id[source_id] = row
        units_by_debate[debate_id] = units
        audit["source_units"] += len(units)

        for source_id, row in rows_by_id.items():
            for related_column, type_column in relation_columns:
                raw_target = _clean_text(row.get(related_column, ""))
                raw_label = _clean_text(row.get(type_column, "")).upper()
                if not raw_target and not raw_label:
                    continue
                audit["relation_slots_seen"] += 1
                if not raw_target or not raw_label:
                    audit["incomplete_relation_slots_excluded"] += 1
                    continue
                tokens = [token.strip() for token in raw_target.split(";")]
                if len(tokens) != 1:
                    audit["multi_target_slots_excluded"] += 1
                    audit["multi_target_edges_excluded"] += len(tokens)
                    continue
                if raw_label not in RELATION_LABELS:
                    audit["unknown_relation_labels_excluded"] += 1
                    continue
                try:
                    target_id = int(float(tokens[0]))
                except ValueError:
                    audit["malformed_target_ids_excluded"] += 1
                    continue
                if target_id not in units or target_id == source_id:
                    audit["missing_or_self_targets_excluded"] += 1
                    continue
                label = RELATION_LABELS[raw_label]
                if raw_label == "CA":
                    later_id, earlier_id = max(source_id, target_id), min(source_id, target_id)
                elif source_id > target_id:
                    later_id, earlier_id = source_id, target_id
                else:
                    audit["forward_direction_relations_excluded"] += 1
                    continue
                relations.append(
                    VivesRelation(
                        debate_id=debate_id,
                        source_id=later_id,
                        target_id=earlier_id,
                        label=label,
                        source_label=raw_label,
                        source_column=related_column,
                    )
                )
                audit[f"eligible_{label.lower()}"] += 1

    relation_keys = Counter(
        (item.debate_id, item.source_id, item.target_id, item.label)
        for item in relations
    )
    audit["duplicate_relation_annotations_removed"] = sum(
        count - 1 for count in relation_keys.values()
    )
    unique_relations: list[VivesRelation] = []
    seen: set[tuple[str, int, int, PairwiseLabel]] = set()
    for relation in relations:
        key = (
            relation.debate_id,
            relation.source_id,
            relation.target_id,
            relation.label,
        )
        if key not in seen:
            seen.add(key)
            unique_relations.append(relation)

    labels_by_pair: dict[tuple[str, int, int], set[PairwiseLabel]] = defaultdict(set)
    for relation in unique_relations:
        labels_by_pair[(relation.debate_id, relation.source_id, relation.target_id)].add(
            relation.label
        )
    ambiguous_pairs = {key for key, labels in labels_by_pair.items() if len(labels) > 1}
    audit["ambiguous_same_pair_relations_excluded"] = sum(
        1
        for item in unique_relations
        if (item.debate_id, item.source_id, item.target_id) in ambiguous_pairs
    )
    unique_relations = [
        item
        for item in unique_relations
        if (item.debate_id, item.source_id, item.target_id) not in ambiguous_pairs
    ]
    audit["eligible_unique_binary_relations"] = len(unique_relations)
    audit["eligible_unique_support"] = sum(item.label == "SUPPORT" for item in unique_relations)
    audit["eligible_unique_attack"] = sum(item.label == "ATTACK" for item in unique_relations)
    audit["eligible_unique_rephrase"] = sum(item.label == "REPHRASE" for item in unique_relations)
    return VivesCorpus(
        units_by_debate=units_by_debate,
        relations=tuple(unique_relations),
        audit=dict(sorted(audit.items())),
        source_hashes=dict(sorted(source_hashes.items())),
    )


def _hardness(source: VivesUnit, candidate: VivesUnit) -> CandidateHardness:
    source_tokens = _content_tokens(source.text)
    candidate_tokens = _content_tokens(candidate.text)
    shared = len(source_tokens & candidate_tokens)
    return CandidateHardness(
        shared_content_tokens=shared,
        overlap_coefficient=shared
        / max(1, min(len(source_tokens), len(candidate_tokens))),
        jaccard_similarity=shared / max(1, len(source_tokens | candidate_tokens)),
        turn_gap=source.source_id - candidate.source_id,
    )


def _negative_rank(
    source: VivesUnit,
    candidate: VivesUnit,
    *,
    seed_key: str,
) -> tuple[object, ...]:
    hardness = _hardness(source, candidate)
    return (
        -int(hardness.shared_content_tokens >= 2),
        -hardness.overlap_coefficient,
        -hardness.shared_content_tokens,
        -hardness.jaccard_similarity,
        hardness.turn_gap,
        _seeded_hash(V6_TRAINING_SEED, seed_key),
    )


def _eligible_vives_relations(
    corpus: VivesCorpus,
) -> dict[str, list[VivesRelation]]:
    related_targets: dict[tuple[str, int], set[int]] = defaultdict(set)
    for item in corpus.relations:
        related_targets[(item.debate_id, item.source_id)].add(item.target_id)
    pools: dict[str, list[VivesRelation]] = defaultdict(list)
    for relation in corpus.relations:
        earlier_ids = {
            source_id
            for source_id in corpus.units_by_debate[relation.debate_id]
            if source_id < relation.source_id
        }
        negatives = earlier_ids - related_targets[(relation.debate_id, relation.source_id)]
        if negatives:
            pools[relation.label].append(relation)
    for label in pools:
        pools[label].sort(
            key=lambda item: (_seeded_hash(V6_TRAINING_SEED, item.key), item.key)
        )
    return pools


def _build_vives_contrast(
    corpus: VivesCorpus,
    relation: VivesRelation,
    repetition_index: int,
) -> VivesContrast:
    units = corpus.units_by_debate[relation.debate_id]
    source = units[relation.source_id]
    related_targets = {
        item.target_id
        for item in corpus.relations
        if item.debate_id == relation.debate_id and item.source_id == relation.source_id
    }
    negatives = sorted(
        (
            item
            for item in units.values()
            if item.source_id < relation.source_id
            and item.source_id not in related_targets
        ),
        key=lambda item: _negative_rank(
            source,
            item,
            seed_key=f"{relation.key}|negative|{item.source_id}",
        ),
    )
    if not negatives:
        raise ValueError(f"no direct-edge negative exists for {relation.key}")
    negative = negatives[repetition_index % len(negatives)]

    candidates = [
        item
        for item in units.values()
        if item.source_id < relation.source_id
        and item.source_id not in {relation.target_id, negative.source_id}
    ]
    candidates.sort(
        key=lambda item: _negative_rank(
            source,
            item,
            seed_key=f"{relation.key}|context|{repetition_index}|{item.source_id}",
        )
    )
    if candidates:
        offset = repetition_index % len(candidates)
        candidates = candidates[offset:] + candidates[:offset]
    block_ids = [relation.target_id, negative.source_id]
    block_ids.extend(item.source_id for item in candidates[: VIVES_BLOCK_SIZE - 2])
    block_ids = sorted(set(block_ids))
    if relation.target_id not in block_ids or negative.source_id not in block_ids:
        raise ValueError("Vives contrast block lost one of its paired targets")
    hardness = _hardness(source, negative)
    pair_group_id = "v6vivespair-" + _seeded_hash(
        V6_TRAINING_SEED,
        f"{relation.key}|{negative.source_id}|{repetition_index}|{block_ids}",
    )[:24]
    return VivesContrast(
        pair_group_id=pair_group_id,
        relation=relation,
        negative_target_id=negative.source_id,
        block_ids=tuple(block_ids),
        repetition_index=repetition_index,
        negative_hardness=hardness,
    )


def select_vives_contrasts(
    corpus: VivesCorpus,
    *,
    target_counts: dict[str, int] = V6_VIVES_POSITIVE_TARGET_COUNTS,
) -> list[VivesContrast]:
    if sum(target_counts.values()) != V6_VIVES_PAIR_COUNT and target_counts is V6_VIVES_POSITIVE_TARGET_COUNTS:
        raise ValueError("v6 Vives targets must sum to the registered pair count")
    pools = _eligible_vives_relations(corpus)
    selected: list[VivesContrast] = []
    occurrences: Counter[str] = Counter()
    for label in POSITIVE_LABELS:
        target = target_counts.get(label, 0)
        pool = pools.get(label, [])
        if target and not pool:
            raise ValueError(f"VivesDebate has no eligible {label} relations")
        for index in range(target):
            relation = pool[index % len(pool)]
            repetition_index = occurrences[relation.key]
            occurrences[relation.key] += 1
            selected.append(_build_vives_contrast(corpus, relation, repetition_index))
    if Counter(item.relation.label for item in selected) != Counter(target_counts):
        raise ValueError("v6 Vives target mix differs from the registered counts")
    if len({item.pair_group_id for item in selected}) != len(selected):
        raise ValueError("v6 Vives pair-group IDs are not unique")
    return sorted(
        selected,
        key=lambda item: _seeded_hash(V6_TRAINING_SEED, item.pair_group_id),
    )


def _prompt_proposition(unit: VivesUnit) -> dict[str, object]:
    return {
        "id": unit.id,
        "chronological_turn": unit.source_id,
        "speaker": unit.speaker,
        "text": unit.text,
    }


def build_vives_pairwise_prompt(
    corpus: VivesCorpus,
    contrast: VivesContrast,
    candidate_target_id: int,
) -> str:
    if candidate_target_id not in contrast.block_ids:
        raise ValueError("candidate target is absent from the Vives comparison block")
    units = corpus.units_by_debate[contrast.relation.debate_id]
    block = json.dumps(
        {
            "new_proposition": _prompt_proposition(units[contrast.relation.source_id]),
            "complete_earlier_comparison_block": [
                _prompt_proposition(units[source_id]) for source_id in contrast.block_ids
            ],
        },
        ensure_ascii=False,
        indent=2,
    )
    template = V5_PROMPT_PATH.read_text(encoding="utf-8")
    replacements = {
        "{{CANDIDATE_TARGET_ID}}": json.dumps(str(candidate_target_id)),
        "{{BLOCK}}": block,
    }
    for marker, value in replacements.items():
        if template.count(marker) != 1:
            raise ValueError(f"pairwise prompt must contain exactly one {marker}")
        template = template.replace(marker, value)
    return template


def materialize_vives_rows(
    corpus: VivesCorpus,
    contrasts: list[VivesContrast],
) -> list[DialAMTrainingRowV6]:
    rows: list[DialAMTrainingRowV6] = []
    for contrast in contrasts:
        relation = contrast.relation
        for role, candidate_id, paired_id, label in (
            ("POSITIVE", relation.target_id, contrast.negative_target_id, relation.label),
            ("NONE", contrast.negative_target_id, relation.target_id, "NONE"),
        ):
            training_row_id = "v6vivesrow-" + _seeded_hash(
                V6_TRAINING_SEED,
                f"{contrast.pair_group_id}|{role}",
            )[:24]
            rows.append(
                DialAMTrainingRowV6(
                    schema_version="dialam_qlora_pairwise_v6",
                    training_row_id=training_row_id,
                    pair_group_id=contrast.pair_group_id,
                    pair_role=role,  # type: ignore[arg-type]
                    source_corpus="VivesDebate",
                    curriculum_stage="vives_warmup",
                    source_example_id=(
                        f"vives:{relation.debate_id}:{relation.source_id}:"
                        f"{relation.target_id}:{contrast.repetition_index}"
                    ),
                    update_id=f"vives:{relation.debate_id}:{relation.source_id}",
                    dialogue_id=f"vives:{relation.debate_id}",
                    block_index=0,
                    candidate_target_id=str(candidate_id),
                    paired_candidate_target_id=str(paired_id),
                    label=label,  # type: ignore[arg-type]
                    positive_label=relation.label,  # type: ignore[arg-type]
                    repetition_index=contrast.repetition_index,
                    is_repeated_positive=contrast.repetition_index > 0,
                    negative_hardness=contrast.negative_hardness,
                    assistant_loss_weight=1.0,
                    messages=[
                        TrainingMessage(
                            role="user",
                            content=build_vives_pairwise_prompt(
                                corpus,
                                contrast,
                                candidate_id,
                            ),
                        ),
                        TrainingMessage(role="assistant", content=label),
                    ],
                )
            )
    return rows


def convert_v5_target_rows(
    rows: list[DialAMTrainingRowV5],
) -> list[DialAMTrainingRowV6]:
    converted: list[DialAMTrainingRowV6] = []
    for row in rows:
        converted.append(
            DialAMTrainingRowV6(
                schema_version="dialam_qlora_pairwise_v6",
                training_row_id=f"v6qt30:{row.training_row_id}",
                pair_group_id=f"v6qt30:{row.pair_group_id}",
                pair_role=row.pair_role,
                source_corpus="DialAM-QT30",
                curriculum_stage="qt30_target",
                source_example_id=row.source_example_id,
                update_id=row.update_id,
                dialogue_id=row.dialogue_id,
                block_index=row.block_index,
                candidate_target_id=row.candidate_target_id,
                paired_candidate_target_id=row.paired_candidate_target_id,
                label=row.label,
                positive_label=row.positive_label,
                repetition_index=row.repetition_index,
                is_repeated_positive=row.is_repeated_positive,
                negative_hardness=row.negative_hardness,
                assistant_loss_weight=row.assistant_loss_weight,
                messages=row.messages,
            )
        )
    return converted


def _load_v5_rows(path: Path) -> list[DialAMTrainingRowV5]:
    return [
        DialAMTrainingRowV5.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _write_jsonl(path: Path, rows: list[DialAMTrainingRowV6]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row.model_dump_json() + "\n")


def _distribution(values: list[int | float]) -> dict[str, int | float]:
    return {"minimum": min(values), "mean": mean(values), "maximum": max(values)}


def _validate_pairs(rows: list[DialAMTrainingRowV6]) -> None:
    if len(rows) % 2:
        raise ValueError("v6 training rows must form two-row contrast batches")
    for index in range(0, len(rows), 2):
        positive, negative = rows[index : index + 2]
        if [positive.pair_role, negative.pair_role] != ["POSITIVE", "NONE"]:
            raise ValueError(f"v6 rows {index}:{index + 2} break pair order")
        if positive.pair_group_id != negative.pair_group_id:
            raise ValueError(f"v6 rows {index}:{index + 2} cross pair groups")
        if positive.messages[0].content != negative.messages[0].content.replace(
            json.dumps(negative.candidate_target_id),
            json.dumps(positive.candidate_target_id),
            1,
        ):
            # The only intentional prompt difference within a pair is the
            # candidate-ID selector; the complete block must remain identical.
            positive_block = positive.messages[0].content.split("INPUT\n", 1)[-1]
            negative_block = negative.messages[0].content.split("INPUT\n", 1)[-1]
            if positive_block != negative_block:
                raise ValueError(f"v6 pair {positive.pair_group_id} changed its block")


def build_dialam_v6_training_corpus(
    *,
    source_dir: Path = DEFAULT_VIVES_SOURCE_DIR,
    output_dir: Path = DEFAULT_TRAINING_DIR,
    v5_path: Path | None = None,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    v5_path = v5_path or output_dir / V5_OUTPUT_FILENAME
    v5_rows = _load_v5_rows(v5_path)
    if len(v5_rows) != V6_QT30_TARGET_SIZE:
        raise ValueError(
            f"v6 requires the exact {V6_QT30_TARGET_SIZE}-row v5 target corpus"
        )

    corpus = load_vives_binary_corpus(source_dir)
    pools = _eligible_vives_relations(corpus)
    contrasts = select_vives_contrasts(corpus)
    vives_rows = materialize_vives_rows(corpus, contrasts)
    qt30_rows = convert_v5_target_rows(v5_rows)
    rows = [*vives_rows, *qt30_rows]
    if len(vives_rows) != V6_VIVES_WARMUP_SIZE or len(rows) != V6_TRAINING_SIZE:
        raise ValueError("v6 curriculum size differs from the preregistration")
    if len({row.training_row_id for row in rows}) != len(rows):
        raise ValueError("v6 training-row IDs are not unique")
    _validate_pairs(vives_rows)
    _validate_pairs(qt30_rows)
    if [row.curriculum_stage for row in rows[:V6_VIVES_WARMUP_SIZE]] != [
        "vives_warmup"
    ] * V6_VIVES_WARMUP_SIZE:
        raise ValueError("v6 warm-up rows are not the first curriculum stage")
    if set(row.curriculum_stage for row in rows[V6_VIVES_WARMUP_SIZE:]) != {
        "qt30_target"
    }:
        raise ValueError("v6 target rows are not the final curriculum stage")

    output_path = output_dir / V6_OUTPUT_FILENAME
    _write_jsonl(output_path, rows)
    schema_path = output_dir / V6_SCHEMA_FILENAME
    schema_path.write_text(
        json.dumps(DialAMTrainingRowV6.model_json_schema(), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    positive_occurrences = Counter(item.relation.key for item in contrasts)
    hardness = [item.negative_hardness for item in contrasts]
    manifest: dict[str, object] = {
        "schema_version": "dialam_training_manifest_v6",
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_version": "v6-vives-warmup-then-qt30-pairwise",
        "base_model": "Qwen/Qwen3-0.6B",
        "size": V6_TRAINING_SIZE,
        "selection_seed": V6_TRAINING_SEED,
        "curriculum": [
            {
                "stage": "vives_warmup",
                "rows": len(vives_rows),
                "pair_count": len(contrasts),
                "language": "Spanish",
                "source": "VivesDebate version 3",
            },
            {
                "stage": "qt30_target",
                "rows": len(qt30_rows),
                "pair_count": len(qt30_rows) // 2,
                "source": "exact unchanged DialAM v5 corpus",
            },
        ],
        "row_label_counts": dict(sorted(Counter(row.label for row in rows).items())),
        "row_source_counts": dict(
            sorted(Counter(row.source_corpus for row in rows).items())
        ),
        "vives_positive_target_counts": dict(
            sorted(Counter(item.relation.label for item in contrasts).items())
        ),
        "vives_available_unique_positive_relations": {
            label: len(pools.get(label, [])) for label in POSITIVE_LABELS
        },
        "vives_repetition": {
            "unique_positive_relations_used": len(positive_occurrences),
            "repeated_positive_rows": sum(
                count - 1 for count in positive_occurrences.values()
            ),
            "maximum_positive_occurrences": max(positive_occurrences.values()),
        },
        "vives_source_audit": corpus.audit,
        "pair_policy": {
            "one_positive_and_one_none_per_device_batch": True,
            "same_complete_block_within_each_pair": True,
            "binary_direct_source_relations_only": True,
            "negative_semantic_retrieval": "deterministic lexical hardness within the same source debate",
            "negative_shared_content_tokens": _distribution(
                [item.shared_content_tokens for item in hardness]
            ),
            "negative_overlap_coefficient": _distribution(
                [item.overlap_coefficient for item in hardness]
            ),
        },
        "objective": {
            "labels": ["NONE", "SUPPORT", "ATTACK", "REPHRASE"],
            "prompt_sha256": file_sha256(V5_PROMPT_PATH),
            "external_behavior_unchanged": True,
            "assistant_target": "one exact class label",
            "patch_assembly": "unchanged deterministic v5 supplied-ID union",
        },
        "loss_policy": {
            "name": "paired_sequential_per_example_assistant_token_mean",
            "assistant_loss_weight_per_row": 1.0,
            "positive_to_none_row_ratio": 1.0,
            "shuffle": False,
            "fixed_qlora_config_unchanged_from_v5": True,
        },
        "leakage_policy": {
            "qt30_target_rows_are_byte_content_equivalent_to_v5_messages": True,
            "qt30_parent_episode_split_unchanged": True,
            "vives_and_qt30_are_separate_source_corpora": True,
            "frozen_qt30_examples_in_training": False,
        },
        "source": {
            "vives_manifest": str(
                (source_dir / VIVES_MANIFEST_FILENAME).relative_to(PROJECT_ROOT)
            ),
            "vives_source_hashes": corpus.source_hashes,
            "vives_license": "CC BY-NC-SA 4.0",
            "v5_training_path": str(v5_path.relative_to(PROJECT_ROOT)),
            "v5_training_sha256": file_sha256(v5_path),
            "v5_manifest_sha256": file_sha256(output_dir / V5_MANIFEST_FILENAME),
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
        "output": {
            "path": str(output_path.relative_to(PROJECT_ROOT)),
            "sha256": file_sha256(output_path),
        },
        "schema": {
            "path": str(schema_path.relative_to(PROJECT_ROOT)),
            "sha256": file_sha256(schema_path),
        },
        "publication": (
            "VivesDebate-derived rows are governed by CC BY-NC-SA 4.0. "
            "QT30-derived rows remain governed by the owner's project-specific "
            "redistribution permission. The combined JSONL stays ignored until selected."
        ),
    }
    manifest_path = output_dir / V6_MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest

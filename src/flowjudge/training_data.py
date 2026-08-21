from __future__ import annotations

import csv
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .data import PROJECT_ROOT, load_benchmark, render_transcript
from .schemas import BenchmarkCase, Relation, RelationGraph, Side

TRAIN_DEBATES = tuple(f"Debate{number}" for number in range(1, 8))
OWN_EVAL_DEBATES = ("Debate8", "Debate9", "Debate10")
DATASET_SIZES = (12, 24, 48, 96)
DATA_SEED = 20260821
RESOLUTION = "Spain ought to legalize surrogacy under a regulated model."

TRAINING_DIR = PROJECT_ROOT / "data" / "training"
EVAL_DIR = PROJECT_ROOT / "data" / "eval"
DEFAULT_CANDIDATES_PATH = TRAINING_DIR / "candidates.jsonl"
DEFAULT_OWN_EVAL_PATH = EVAL_DIR / "own_eval.jsonl"
DEFAULT_SPLIT_MANIFEST_PATH = TRAINING_DIR / "split_manifest.json"


class DataModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TrainingUnit(DataModel):
    id: str = Field(pattern=r"^U[1-9][0-9]*$")
    side: Side
    text: str = Field(min_length=1)


class TrainingCandidate(DataModel):
    example_id: str
    source_debate: str
    source_anchor_pair: tuple[int, int]
    source_unit_ids: list[int] = Field(min_length=6, max_length=12)
    resolution: str
    title: str
    units: list[TrainingUnit] = Field(min_length=6, max_length=12)
    gold_graph: RelationGraph

    @model_validator(mode="after")
    def validate_candidate(self) -> "TrainingCandidate":
        unit_ids = [int(unit.id[1:]) for unit in self.units]
        if unit_ids != self.source_unit_ids or unit_ids != sorted(set(unit_ids)):
            raise ValueError("training unit IDs must match unique chronological source IDs")
        unit_by_id = {unit.id: unit for unit in self.units}
        for edge in self.gold_graph.relations:
            if edge.source not in unit_by_id or edge.target not in unit_by_id:
                raise ValueError("gold edge references an unknown unit")
            if int(edge.source[1:]) <= int(edge.target[1:]):
                raise ValueError("gold edge must point from later to earlier")
            if unit_by_id[edge.source].side == unit_by_id[edge.target].side:
                raise ValueError("gold edge must cross sides")
        return self

    def transcript(self) -> str:
        lines = [f"Resolution: {self.resolution}", f"Excerpt topic: {self.title}"]
        lines.extend(f"{unit.id} [{unit.side.value}]: {unit.text}" for unit in self.units)
        return "\n".join(lines)


class RewrittenUnit(DataModel):
    id: str = Field(pattern=r"^U[1-9][0-9]*$")
    side: Side
    text: str = Field(min_length=1, max_length=600)


class TeacherRewrite(DataModel):
    units: list[RewrittenUnit] = Field(min_length=6, max_length=12)


class QualityAssessment(DataModel):
    accept: bool
    meaning_preserved: bool
    edge_support_preserved: bool
    no_new_response_cues: bool
    fluent_and_atomic: bool
    brief_reason: str = Field(min_length=1)


class ChatMessage(DataModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class TrainingExample(DataModel):
    example_id: str
    source_debate: str
    source_unit_ids: list[int]
    teacher_model: str
    filter_model: str
    quality: QualityAssessment
    messages: list[ChatMessage] = Field(min_length=2, max_length=2)


@dataclass(frozen=True)
class SourceUnit:
    source_id: int
    side: Side
    text: str


def build_offline_training_assets(limit: int = DATASET_SIZES[-1]) -> dict[str, Any]:
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    candidates = build_training_candidates(limit=limit)
    _write_jsonl(DEFAULT_CANDIDATES_PATH, (candidate.model_dump() for candidate in candidates))

    own_eval = [
        case
        for case in load_benchmark()
        if case.scenario.source is not None
        and case.scenario.source.debate_id in OWN_EVAL_DEBATES
    ]
    _write_jsonl(DEFAULT_OWN_EVAL_PATH, (case.model_dump() for case in own_eval))

    manifest = {
        "seed": DATA_SEED,
        "train_debates": list(TRAIN_DEBATES),
        "own_eval_debates": list(OWN_EVAL_DEBATES),
        "staff_heldout_policy": "accepted at evaluation time as BenchmarkCase JSONL; never used for generation or training",
        "candidate_count": len(candidates),
        "own_eval_count": len(own_eval),
        "planned_dataset_sizes": [size for size in DATASET_SIZES if size <= len(candidates)],
        "spacing_justification": "approximately log2-spaced to expose the low-data learning curve while capping the first full sweep at 96 strictly filtered examples",
        "candidate_path": str(DEFAULT_CANDIDATES_PATH.relative_to(PROJECT_ROOT)),
        "own_eval_path": str(DEFAULT_OWN_EVAL_PATH.relative_to(PROJECT_ROOT)),
        "source_license": "VivesDebate CC BY-NC-SA 4.0",
        "network_calls_made": 0,
    }
    DEFAULT_SPLIT_MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_training_candidates(limit: int = DATASET_SIZES[-1]) -> list[TrainingCandidate]:
    candidates: list[TrainingCandidate] = []
    seen_windows: set[tuple[str, tuple[int, ...]]] = set()
    for debate_id in TRAIN_DEBATES:
        rows = _load_source_debate(debate_id)
        source_units = {unit.source_id: unit for unit in rows}
        ordered_ids = [unit.source_id for unit in rows]
        conflicts = _cross_stance_conflicts(debate_id, source_units)
        all_conflict_pairs = {
            _later_earlier(source_id, target_id)
            for source_id, target_id in conflicts
        }
        for anchor_source, anchor_target in conflicts:
            later, earlier = _later_earlier(anchor_source, anchor_target)
            selected_ids = _window_ids(ordered_ids, earlier, later)
            window_key = (debate_id, tuple(selected_ids))
            if window_key in seen_windows:
                continue
            seen_windows.add(window_key)
            selected_set = set(selected_ids)
            edges = [
                Relation(source=f"U{edge_source}", target=f"U{edge_target}", type="responds_to")
                for edge_source, edge_target in sorted(all_conflict_pairs)
                if edge_source in selected_set and edge_target in selected_set
            ]
            if not edges:
                continue
            units = [
                TrainingUnit(
                    id=f"U{source_id}",
                    side=source_units[source_id].side,
                    text=source_units[source_id].text,
                )
                for source_id in selected_ids
            ]
            example_id = f"train_{debate_id.lower()}_{later}_{earlier}"
            candidates.append(
                TrainingCandidate(
                    example_id=example_id,
                    source_debate=debate_id,
                    source_anchor_pair=(later, earlier),
                    source_unit_ids=selected_ids,
                    resolution=RESOLUTION,
                    title=f"Source-derived response window anchored at U{later} → U{earlier}",
                    units=units,
                    gold_graph=RelationGraph(relations=edges),
                )
            )

    random.Random(DATA_SEED).shuffle(candidates)
    return candidates[:limit]


def load_training_candidates(path: Path = DEFAULT_CANDIDATES_PATH) -> list[TrainingCandidate]:
    return [TrainingCandidate.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_eval_cases(path: Path) -> list[BenchmarkCase]:
    return [BenchmarkCase.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def build_training_prompt(candidate: TrainingCandidate, rewritten_units: list[RewrittenUnit]) -> str:
    transcript_lines = [f"Resolution: {candidate.resolution}"]
    transcript_lines.extend(
        f"{unit.id} [{unit.side.value}]: {unit.text}" for unit in rewritten_units
    )
    transcript = "\n".join(transcript_lines)
    return (
        "Identify every direct response edge in this debate excerpt. Return only one bare JSON object "
        "with a relations array. An edge must point from a later opposing-side unit to the earlier unit "
        "it directly answers, attacks, mitigates, or turns. Exclude topical similarity, same-side "
        "extensions, repetition/rephrase, and independent counterarguments.\n\n"
        f"{transcript}"
    )


def make_training_example(
    candidate: TrainingCandidate,
    rewrite: TeacherRewrite,
    quality: QualityAssessment,
    teacher_model: str,
    filter_model: str,
) -> TrainingExample:
    expected = [(unit.id, unit.side) for unit in candidate.units]
    actual = [(unit.id, unit.side) for unit in rewrite.units]
    if actual != expected:
        raise ValueError(f"teacher changed unit IDs, order, or sides for {candidate.example_id}")
    if not (
        quality.accept
        and quality.meaning_preserved
        and quality.edge_support_preserved
        and quality.no_new_response_cues
        and quality.fluent_and_atomic
    ):
        raise ValueError(f"quality filter rejected {candidate.example_id}: {quality.brief_reason}")
    return TrainingExample(
        example_id=candidate.example_id,
        source_debate=candidate.source_debate,
        source_unit_ids=candidate.source_unit_ids,
        teacher_model=teacher_model,
        filter_model=filter_model,
        quality=quality,
        messages=[
            ChatMessage(role="user", content=build_training_prompt(candidate, rewrite.units)),
            ChatMessage(role="assistant", content=candidate.gold_graph.model_dump_json()),
        ],
    )


def write_dataset_slices(examples: list[TrainingExample]) -> list[Path]:
    if len(examples) < DATASET_SIZES[-1]:
        raise ValueError(f"need at least {DATASET_SIZES[-1]} accepted examples, got {len(examples)}")
    paths: list[Path] = []
    for size in DATASET_SIZES:
        path = TRAINING_DIR / f"v1_n{size}.jsonl"
        _write_jsonl(path, (example.model_dump() for example in examples[:size]))
        paths.append(path)
    return paths


def render_candidate_for_teacher(candidate: TrainingCandidate) -> str:
    payload = {
        "example_id": candidate.example_id,
        "resolution": candidate.resolution,
        "units": [unit.model_dump() for unit in candidate.units],
        "fixed_gold_graph": candidate.gold_graph.model_dump(),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _load_source_debate(debate_id: str) -> list[SourceUnit]:
    path = PROJECT_ROOT / "data" / "source" / "vivesdebate" / f"{debate_id}.csv"
    units: list[SourceUnit] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            source_id = int(row["ID (Chronological)"])
            stance = row["TEAM STANCE"].strip().upper()
            if stance not in {"FAVOUR", "AGAINST"}:
                continue
            raw_text = row.get("ADU_EN", "") or row.get("ADU_ES", "") or row.get("ADU_CAT", "")
            text = re.sub(r"\s+", " ", raw_text).strip()
            if not text:
                continue
            side = Side.AFF if stance == "FAVOUR" else Side.NEG
            units.append(SourceUnit(source_id=source_id, side=side, text=text))
    return units


def _cross_stance_conflicts(
    debate_id: str,
    source_units: dict[int, SourceUnit],
) -> list[tuple[int, int]]:
    path = PROJECT_ROOT / "data" / "source" / "vivesdebate" / f"{debate_id}.csv"
    pairs: set[tuple[int, int]] = set()
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            source_id = int(row["ID (Chronological)"])
            if source_id not in source_units:
                continue
            for field, raw_value in row.items():
                match = re.fullmatch(r"RELATED ID(?:\.([0-9]+))?", field)
                if not match or not raw_value:
                    continue
                suffix = f".{match.group(1)}" if match.group(1) else ""
                relation_type = (row.get(f"ARGUMENTAL RELATION  TYPE{suffix}", "") or "").upper()
                if relation_type != "CA":
                    continue
                for token in re.findall(r"[0-9]+", raw_value):
                    target_id = int(token)
                    if target_id not in source_units:
                        continue
                    if source_units[source_id].side != source_units[target_id].side:
                        pairs.add((source_id, target_id))
    return sorted(pairs)


def _window_ids(ordered_ids: list[int], earlier: int, later: int) -> list[int]:
    index_by_id = {source_id: index for index, source_id in enumerate(ordered_ids)}
    selected: set[int] = {earlier, later}
    for endpoint in (earlier, later):
        index = index_by_id[endpoint]
        for offset in (-2, -1, 1, 2):
            neighbor_index = index + offset
            if 0 <= neighbor_index < len(ordered_ids):
                selected.add(ordered_ids[neighbor_index])
    midpoint = (index_by_id[earlier] + index_by_id[later]) / 2
    for source_id in sorted(ordered_ids, key=lambda item: abs(index_by_id[item] - midpoint)):
        if len(selected) >= 8:
            break
        selected.add(source_id)
    return sorted(selected)[:10]


def _later_earlier(source_id: int, target_id: int) -> tuple[int, int]:
    return (source_id, target_id) if source_id > target_id else (target_id, source_id)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

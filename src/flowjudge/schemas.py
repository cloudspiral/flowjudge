from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Side(StrEnum):
    AFF = "AFF"
    NEG = "NEG"


class SourceStance(StrEnum):
    FAVOUR = "FAVOUR"
    AGAINST = "AGAINST"


class Category(StrEnum):
    VIVESDEBATE = "vivesdebate_real"
    SYNTHETIC_DROPPED = "synthetic_dropped_argument"
    SYNTHETIC_CROSS_APPLICATION = "synthetic_cross_application"


class SourceRelationType(StrEnum):
    INFERENCE = "inference"
    CONFLICT = "conflict"
    REPHRASE = "rephrase"


class Unit(StrictModel):
    id: str = Field(pattern=r"^U[1-9][0-9]*$")
    side: Side
    text: str = Field(min_length=1)
    source_id: int | None = Field(default=None, gt=0)
    phase: str | None = None
    argument_number: str | None = None
    source_stance: SourceStance | None = None
    text_ca: str | None = None
    text_es: str | None = None
    text_en: str | None = None


class SourceRelation(StrictModel):
    source: str = Field(pattern=r"^U[1-9][0-9]*$")
    target: str = Field(pattern=r"^U[1-9][0-9]*$")
    type: SourceRelationType
    source_label: str = Field(min_length=2, max_length=2)
    source_column: str = Field(min_length=10)


class SourceAnnotationIssue(StrictModel):
    source: str = Field(pattern=r"^U[1-9][0-9]*$")
    source_column: str = Field(min_length=10)
    type_column: str = Field(min_length=10)
    raw_related_id: str
    raw_relation_type: str
    reason: Literal["incomplete_pair", "unknown_relation_type", "unknown_target"]


class SourceProvenance(StrictModel):
    corpus: Literal["VivesDebate"]
    debate_id: str = Field(pattern=r"^Debate([1-9]|[12][0-9])$")
    source_file: str
    source_md5: str = Field(pattern=r"^[0-9a-f]{32}$")
    source_doi: Literal["10.5281/zenodo.6531487"]
    license: Literal["CC BY-NC-SA 4.0"]
    selected_language: Literal["ADU_EN"]


class JuryScore(StrictModel):
    stance: SourceStance
    score: float
    thesis_solidity: float
    argumentation_quality: float
    adaptability: float


class JuryOutcome(StrictModel):
    winner: Literal["FAVOUR", "AGAINST", "TIE"]
    margin: float = Field(ge=0)
    scores: list[JuryScore] = Field(min_length=2, max_length=2)

    @model_validator(mode="after")
    def validate_outcome(self) -> "JuryOutcome":
        if {score.stance for score in self.scores} != {SourceStance.FAVOUR, SourceStance.AGAINST}:
            raise ValueError("jury outcome must contain one score for each stance")
        by_stance = {score.stance.value: score.score for score in self.scores}
        expected = "TIE"
        if by_stance["FAVOUR"] > by_stance["AGAINST"]:
            expected = "FAVOUR"
        elif by_stance["AGAINST"] > by_stance["FAVOUR"]:
            expected = "AGAINST"
        if self.winner != expected:
            raise ValueError(f"jury winner should be {expected}")
        expected_margin = abs(by_stance["FAVOUR"] - by_stance["AGAINST"])
        if abs(self.margin - expected_margin) > 1e-9:
            raise ValueError(f"jury margin should be {expected_margin}")
        return self


class Scenario(StrictModel):
    scenario_id: str
    category: Category
    title: str
    resolution: str
    units: list[Unit] = Field(min_length=6)
    source: SourceProvenance | None = None
    source_relations: list[SourceRelation] = Field(default_factory=list)
    source_annotation_issues: list[SourceAnnotationIssue] = Field(default_factory=list)
    jury_outcome: JuryOutcome | None = None

    @model_validator(mode="after")
    def validate_scenario(self) -> "Scenario":
        unit_numbers = [int(unit.id[1:]) for unit in self.units]
        if unit_numbers != sorted(set(unit_numbers)):
            raise ValueError("unit ids must be unique and strictly chronological")
        unit_ids = {unit.id for unit in self.units}
        for relation in self.source_relations:
            if relation.source not in unit_ids or relation.target not in unit_ids:
                raise ValueError(f"source relation references unknown unit: {relation.source}->{relation.target}")

        if self.category == Category.VIVESDEBATE:
            if self.source is None or self.jury_outcome is None:
                raise ValueError("VivesDebate scenarios require source provenance and jury outcome")
            expected_side = {SourceStance.FAVOUR: Side.AFF, SourceStance.AGAINST: Side.NEG}
            for unit in self.units:
                if unit.source_id != int(unit.id[1:]):
                    raise ValueError(f"source id mismatch for {unit.id}")
                if unit.source_stance is None or unit.side != expected_side[unit.source_stance]:
                    raise ValueError(f"stance mapping mismatch for {unit.id}")
                if unit.text != unit.text_en or not all((unit.text_ca, unit.text_es, unit.text_en)):
                    raise ValueError(f"VivesDebate text metadata is incomplete for {unit.id}")
        elif (
            self.source is not None
            or self.jury_outcome is not None
            or self.source_relations
            or self.source_annotation_issues
        ):
            raise ValueError("synthetic scenarios cannot claim VivesDebate source annotations")
        return self


class Relation(StrictModel):
    source: str = Field(pattern=r"^U[1-9][0-9]*$")
    target: str = Field(pattern=r"^U[1-9][0-9]*$")
    type: Literal["responds_to"]


class RelationGraph(StrictModel):
    relations: list[Relation]

    @model_validator(mode="after")
    def reject_duplicate_edges(self) -> "RelationGraph":
        keys = [(edge.source, edge.target, edge.type) for edge in self.relations]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate relations are not allowed")
        return self


class JudgePair(StrictModel):
    source: str = Field(pattern=r"^U[1-9][0-9]*$")
    target: str = Field(pattern=r"^U[1-9][0-9]*$")


class JudgeAssessment(StrictModel):
    assessment: Literal["correct", "incorrect"]
    missed_edges: list[JudgePair]
    spurious_edges: list[JudgePair]
    brief_reason: str = Field(min_length=1)


class ExplainedRelation(Relation):
    explanation: str = Field(min_length=1)


class HardNegative(StrictModel):
    source: str = Field(pattern=r"^U[1-9][0-9]*$")
    target: str = Field(pattern=r"^U[1-9][0-9]*$")
    explanation: str = Field(min_length=1)


class GoldBlueprint(StrictModel):
    scenario_id: str
    category: Category
    design_intent: str = Field(min_length=1)
    gold_relations: list[ExplainedRelation]
    hard_negatives: list[HardNegative] = Field(default_factory=list)


class BenchmarkCase(StrictModel):
    scenario: Scenario
    gold: GoldBlueprint

    def graph(self) -> RelationGraph:
        return RelationGraph(
            relations=[Relation(**edge.model_dump(exclude={"explanation"})) for edge in self.gold.gold_relations]
        )

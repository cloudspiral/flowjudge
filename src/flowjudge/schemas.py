from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Side(StrEnum):
    AFF = "AFF"
    NEG = "NEG"


class Category(StrEnum):
    CLEAR_DIRECT = "clear_direct_responses"
    NONRESPONSIVE = "topically_related_nonresponsive"
    SAME_SIDE = "same_side_extensions"
    BRANCHING = "branching_or_partial"
    LONG_DISTANCE = "long_distance_paraphrased"
    DISTRACTOR = "distractors_or_embedded_instructions"


class Unit(StrictModel):
    id: str = Field(pattern=r"^U[1-9][0-9]*$")
    side: Side
    text: str = Field(min_length=1)


class Scenario(StrictModel):
    scenario_id: str
    category: Category
    title: str
    resolution: str
    units: list[Unit] = Field(min_length=6, max_length=12)

    @model_validator(mode="after")
    def validate_unit_sequence(self) -> "Scenario":
        expected = [f"U{index}" for index in range(1, len(self.units) + 1)]
        actual = [unit.id for unit in self.units]
        if actual != expected:
            raise ValueError(f"unit ids must be sequential: expected {expected}, got {actual}")
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
    hard_negatives: list[HardNegative]


class PilotCase(StrictModel):
    scenario: Scenario
    gold: GoldBlueprint

    def graph(self) -> RelationGraph:
        return RelationGraph(
            relations=[Relation(**edge.model_dump(exclude={"explanation"})) for edge in self.gold.gold_relations]
        )

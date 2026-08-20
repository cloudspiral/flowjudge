from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from pydantic import ValidationError

from .data import eligible_response_pairs
from .schemas import Category, JudgeAssessment, PilotCase, RelationGraph

EdgeKey = tuple[str, str, str]


@dataclass(frozen=True)
class ParsedPrediction:
    graph: RelationGraph | None
    error: str | None

    @property
    def valid_json(self) -> bool:
        return self.graph is not None


def parse_prediction(raw_response: str) -> ParsedPrediction:
    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        return ParsedPrediction(graph=None, error=f"invalid JSON: {exc.msg}")
    try:
        return ParsedPrediction(graph=RelationGraph.model_validate(payload), error=None)
    except ValidationError as exc:
        return ParsedPrediction(graph=None, error=f"schema validation failed: {exc}")


def parse_judge_assessment(raw_response: str) -> JudgeAssessment | None:
    try:
        payload = json.loads(raw_response)
        return JudgeAssessment.model_validate(payload)
    except (json.JSONDecodeError, ValidationError):
        return None


def graph_edges(graph: RelationGraph | None) -> set[EdgeKey]:
    if graph is None:
        return set()
    return {(edge.source, edge.target, edge.type) for edge in graph.relations}


def score_records(cases: Iterable[PilotCase], records: list[dict[str, Any]]) -> dict[str, Any]:
    case_by_id = {case.scenario.scenario_id: case for case in cases}
    category_accumulators: dict[str, _Accumulator] = defaultdict(_Accumulator)
    combination_accumulators: dict[str, _Accumulator] = defaultdict(_Accumulator)
    overall = _Accumulator()
    failures: list[dict[str, Any]] = []

    for record in records:
        case = case_by_id[record["scenario_id"]]
        parsed = parse_prediction(record["raw_response"])
        predicted = graph_edges(parsed.graph)
        gold = graph_edges(case.graph())
        category = case.scenario.category.value
        eligible = {
            (source, target, "responds_to")
            for source, target in eligible_response_pairs(case.scenario)
        }
        is_topical_nonresponse = case.scenario.category == Category.NONRESPONSIVE
        judge_attempted = "raw_judge_response" in record
        judge = parse_judge_assessment(record["raw_judge_response"]) if judge_attempted else None

        overall.add(parsed.valid_json, predicted, gold, eligible, is_topical_nonresponse, judge_attempted, judge)
        category_accumulators[category].add(
            parsed.valid_json, predicted, gold, eligible, is_topical_nonresponse, judge_attempted, judge
        )
        if all(field in record for field in ("provider", "model", "prompt")):
            combination = f'{record["provider"]}:{record["model"]}:{record["prompt"]}'
            combination_accumulators[combination].add(
                parsed.valid_json, predicted, gold, eligible, is_topical_nonresponse, judge_attempted, judge
            )
        if not (parsed.valid_json and predicted == gold):
            failures.append(
                {
                    "assignment_id": record.get("assignment_id"),
                    "scenario_id": case.scenario.scenario_id,
                    "category": category,
                    "provider": record.get("provider"),
                    "model": record.get("model"),
                    "prompt": record.get("prompt"),
                    "parse_error": parsed.error,
                    "missed_edges": [_edge_dict(edge) for edge in sorted(gold - predicted)],
                    "spurious_edges": [_edge_dict(edge) for edge in sorted(predicted - gold)],
                }
            )

    return {
        **overall.to_metrics(),
        "by_category": {
            category: accumulator.to_metrics()
            for category, accumulator in sorted(category_accumulators.items())
        },
        "by_model_prompt": {
            combination: accumulator.to_metrics()
            for combination, accumulator in sorted(combination_accumulators.items())
        },
        "failure_cases": failures,
    }


class _Accumulator:
    def __init__(self) -> None:
        self.assignments = 0
        self.valid = 0
        self.exact = 0
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.topical_assignments = 0
        self.topical_assignments_with_fp = 0
        self.topical_pair_fp = 0
        self.topical_pair_candidates = 0
        self.judge_assignments = 0
        self.judge_valid = 0
        self.judge_correct = 0

    def add(
        self,
        valid: bool,
        predicted: set[EdgeKey],
        gold: set[EdgeKey],
        eligible: set[EdgeKey],
        is_topical_nonresponse: bool,
        judge_attempted: bool,
        judge: JudgeAssessment | None,
    ) -> None:
        self.assignments += 1
        self.valid += int(valid)
        self.exact += int(valid and predicted == gold)
        self.tp += len(predicted & gold)
        self.fp += len(predicted - gold)
        self.fn += len(gold - predicted)
        if is_topical_nonresponse:
            self.topical_assignments += 1
            self.topical_assignments_with_fp += int(bool(predicted - gold))
            self.topical_pair_fp += len((predicted - gold) & eligible)
            self.topical_pair_candidates += len(eligible)
        if judge_attempted:
            self.judge_assignments += 1
        if judge is not None:
            self.judge_valid += 1
            self.judge_correct += int(judge.assessment == "correct")

    def to_metrics(self) -> dict[str, int | float | None]:
        precision = _divide(self.tp, self.tp + self.fp)
        recall = _divide(self.tp, self.tp + self.fn)
        if precision is None or recall is None or precision + recall == 0:
            f1 = 0.0 if precision is not None and recall is not None else None
        else:
            f1 = 2 * precision * recall / (precision + recall)
        return {
            "assignments": self.assignments,
            "valid_json_rate": _divide(self.valid, self.assignments),
            "exact_graph_match_rate": _divide(self.exact, self.assignments),
            "edge_precision": precision,
            "edge_recall": recall,
            "edge_f1": f1,
            "true_positive_edges": self.tp,
            "false_positive_edges": self.fp,
            "false_negative_edges": self.fn,
            "topical_nonresponse_false_positive_rate": _divide(
                self.topical_assignments_with_fp, self.topical_assignments
            ),
            "topical_nonresponse_pair_false_positive_rate": _divide(
                self.topical_pair_fp, self.topical_pair_candidates
            ),
            "fixed_judge_valid_json_rate": _divide(self.judge_valid, self.judge_assignments),
            "fixed_judge_correct_rate": _divide(self.judge_correct, self.judge_assignments),
        }


def _divide(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _edge_dict(edge: EdgeKey) -> dict[str, str]:
    source, target, edge_type = edge
    return {"source": source, "target": target, "type": edge_type}


def score_run(run_directory: Path, cases: Iterable[PilotCase]) -> dict[str, Any]:
    records_path = run_directory / "records.jsonl"
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line]
    summary = score_records(cases, records)
    (run_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary

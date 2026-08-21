from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from pydantic import ValidationError

from .schemas import BenchmarkCase, JudgeAssessment, RelationGraph

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


def score_records(cases: Iterable[BenchmarkCase], records: list[dict[str, Any]]) -> dict[str, Any]:
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
        hard_negatives = {
            (pair.source, pair.target, "responds_to")
            for pair in case.gold.hard_negatives
        }
        judge_attempted = "raw_judge_response" in record
        judge = parse_judge_assessment(record["raw_judge_response"]) if judge_attempted else None

        overall.add(parsed.valid_json, predicted, gold, hard_negatives, judge_attempted, judge)
        category_accumulators[category].add(
            parsed.valid_json, predicted, gold, hard_negatives, judge_attempted, judge
        )
        if all(field in record for field in ("provider", "model", "prompt")):
            combination = f'{record["provider"]}:{record["model"]}:{record["prompt"]}'
            combination_accumulators[combination].add(
                parsed.valid_json, predicted, gold, hard_negatives, judge_attempted, judge
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
        self.hard_negative_fp = 0
        self.hard_negative_pairs = 0
        self.judge_assignments = 0
        self.judge_valid = 0
        self.judge_assessed_correct = 0
        self.judge_decision_agreement = 0
        self.judge_edge_diagnosis_match = 0

    def add(
        self,
        valid: bool,
        predicted: set[EdgeKey],
        gold: set[EdgeKey],
        hard_negatives: set[EdgeKey],
        judge_attempted: bool,
        judge: JudgeAssessment | None,
    ) -> None:
        self.assignments += 1
        self.valid += int(valid)
        self.exact += int(valid and predicted == gold)
        self.tp += len(predicted & gold)
        self.fp += len(predicted - gold)
        self.fn += len(gold - predicted)
        self.hard_negative_fp += len(predicted & hard_negatives)
        self.hard_negative_pairs += len(hard_negatives)
        if judge_attempted:
            self.judge_assignments += 1
        if judge is not None:
            self.judge_valid += 1
            deterministic_correct = valid and predicted == gold
            judge_says_correct = judge.assessment == "correct"
            self.judge_assessed_correct += int(judge_says_correct)
            self.judge_decision_agreement += int(judge_says_correct == deterministic_correct)
            expected_missed = {(source, target) for source, target, _ in gold - predicted}
            expected_spurious = {(source, target) for source, target, _ in predicted - gold}
            judge_missed = {(pair.source, pair.target) for pair in judge.missed_edges}
            judge_spurious = {(pair.source, pair.target) for pair in judge.spurious_edges}
            self.judge_edge_diagnosis_match += int(
                judge_missed == expected_missed and judge_spurious == expected_spurious
            )

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
            "annotated_hard_negative_false_positive_rate": _divide(
                self.hard_negative_fp, self.hard_negative_pairs
            ),
            "fixed_judge_valid_json_rate": _divide(self.judge_valid, self.judge_assignments),
            "fixed_judge_assessed_correct_rate": _divide(
                self.judge_assessed_correct, self.judge_assignments
            ),
            "fixed_judge_decision_agreement_rate": _divide(
                self.judge_decision_agreement, self.judge_assignments
            ),
            "fixed_judge_edge_diagnosis_match_rate": _divide(
                self.judge_edge_diagnosis_match, self.judge_assignments
            ),
        }


def _divide(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _edge_dict(edge: EdgeKey) -> dict[str, str]:
    source, target, edge_type = edge
    return {"source": source, "target": target, "type": edge_type}


def score_run(run_directory: Path, cases: Iterable[BenchmarkCase]) -> dict[str, Any]:
    records_path = run_directory / "records.jsonl"
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line]
    summary = score_records(cases, records)
    (run_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from pydantic import ValidationError

from .schemas import BenchmarkCase, HardNegativePhenomenon, JudgeAssessment, RelationGraph

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


def parse_prediction_normalized(raw_response: str) -> ParsedPrediction:
    """Parse strict JSON, or JSON wrapped in exactly one Markdown code fence.

    This deliberately does not repair prose, malformed JSON, field names, edge
    directions, or schema violations. It isolates a common serialization-only
    failure without changing the predicted graph.
    """
    strict = parse_prediction(raw_response)
    if strict.valid_json:
        return strict
    match = re.fullmatch(
        r"\s*```(?:json)?\s*\n?(.*?)\n?```\s*",
        raw_response,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if match is None:
        return ParsedPrediction(graph=None, error="normalization rejected non-fence output")
    normalized = parse_prediction(match.group(1).strip())
    if normalized.valid_json:
        return normalized
    return ParsedPrediction(graph=None, error=f"fence-normalized {normalized.error}")


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
    split_accumulators: dict[str, _Accumulator] = defaultdict(_Accumulator)
    phenomenon_accumulators: dict[str, _Accumulator] = defaultdict(_Accumulator)
    combination_accumulators: dict[str, _Accumulator] = defaultdict(_Accumulator)
    overall = _Accumulator()
    failures: list[dict[str, Any]] = []
    formatting_only_failures: list[dict[str, Any]] = []

    for record in records:
        case = case_by_id[record["scenario_id"]]
        parsed = parse_prediction(record["raw_response"])
        normalized = parse_prediction_normalized(record["raw_response"])
        predicted = graph_edges(parsed.graph)
        normalized_predicted = graph_edges(normalized.graph)
        gold = graph_edges(case.graph())
        category = case.scenario.category.value
        split = case.scenario.split.value
        phenomena = [phenomenon.value for phenomenon in case.scenario.phenomena]
        hard_negatives = {
            phenomenon.value: {
                (pair.source, pair.target, "responds_to")
                for pair in case.gold.hard_negatives
                if pair.phenomenon == phenomenon
            }
            for phenomenon in HardNegativePhenomenon
        }
        judge_attempted = "raw_judge_response" in record
        judge = parse_judge_assessment(record["raw_judge_response"]) if judge_attempted else None

        add_args = (
            parsed.valid_json,
            predicted,
            normalized.valid_json,
            normalized_predicted,
            gold,
            hard_negatives,
            judge_attempted,
            judge,
        )
        overall.add(*add_args)
        category_accumulators[category].add(
            *add_args
        )
        split_accumulators[split].add(*add_args)
        for phenomenon in phenomena:
            phenomenon_accumulators[phenomenon].add(*add_args)
        if all(field in record for field in ("provider", "model", "prompt")):
            combination = f'{record["provider"]}:{record["model"]}:{record["prompt"]}'
            combination_accumulators[combination].add(*add_args)
        failure = {
            "assignment_id": record.get("assignment_id"),
            "scenario_id": case.scenario.scenario_id,
            "category": category,
            "split": split,
            "phenomena": phenomena,
            "provider": record.get("provider"),
            "model": record.get("model"),
            "prompt": record.get("prompt"),
            "strict_parse_error": parsed.error,
            "normalized_parse_error": normalized.error,
            "missed_edges": [_edge_dict(edge) for edge in sorted(gold - normalized_predicted)],
            "spurious_edges": [_edge_dict(edge) for edge in sorted(normalized_predicted - gold)],
        }
        if not (normalized.valid_json and normalized_predicted == gold):
            failures.append(failure)
        elif not (parsed.valid_json and predicted == gold):
            formatting_only_failures.append(
                {
                    **failure,
                    "missed_edges": [],
                    "spurious_edges": [],
                }
            )

    return {
        **overall.to_metrics(),
        "by_category": {
            category: accumulator.to_metrics()
            for category, accumulator in sorted(category_accumulators.items())
        },
        "by_split": {
            split: accumulator.to_metrics()
            for split, accumulator in sorted(split_accumulators.items())
        },
        "by_phenomenon": {
            phenomenon: accumulator.to_metrics()
            for phenomenon, accumulator in sorted(phenomenon_accumulators.items())
        },
        "by_model_prompt": {
            combination: accumulator.to_metrics()
            for combination, accumulator in sorted(combination_accumulators.items())
        },
        "failure_cases": failures,
        "formatting_only_failures": formatting_only_failures,
    }


class _Accumulator:
    def __init__(self) -> None:
        self.assignments = 0
        self.valid = 0
        self.exact = 0
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.normalized_valid = 0
        self.normalized_exact = 0
        self.normalized_tp = 0
        self.normalized_fp = 0
        self.normalized_fn = 0
        self.hard_negative_fp = 0
        self.hard_negative_pairs = 0
        self.normalized_hard_negative_fp = 0
        self.hard_negative_fp_by_phenomenon: dict[str, int] = defaultdict(int)
        self.normalized_hard_negative_fp_by_phenomenon: dict[str, int] = defaultdict(int)
        self.hard_negative_pairs_by_phenomenon: dict[str, int] = defaultdict(int)
        self.judge_assignments = 0
        self.judge_valid = 0
        self.judge_assessed_correct = 0
        self.judge_decision_agreement = 0
        self.judge_edge_diagnosis_match = 0
        self.judge_normalized_decision_agreement = 0
        self.judge_normalized_edge_diagnosis_match = 0
        self.judge_rubric_scored = 0
        self.judge_spec_adherence_total = 0
        self.judge_robustness_total = 0

    def add(
        self,
        valid: bool,
        predicted: set[EdgeKey],
        normalized_valid: bool,
        normalized_predicted: set[EdgeKey],
        gold: set[EdgeKey],
        hard_negatives: dict[str, set[EdgeKey]],
        judge_attempted: bool,
        judge: JudgeAssessment | None,
    ) -> None:
        self.assignments += 1
        self.valid += int(valid)
        self.exact += int(valid and predicted == gold)
        self.tp += len(predicted & gold)
        self.fp += len(predicted - gold)
        self.fn += len(gold - predicted)
        self.normalized_valid += int(normalized_valid)
        self.normalized_exact += int(normalized_valid and normalized_predicted == gold)
        self.normalized_tp += len(normalized_predicted & gold)
        self.normalized_fp += len(normalized_predicted - gold)
        self.normalized_fn += len(gold - normalized_predicted)
        for phenomenon, pairs in hard_negatives.items():
            strict_count = len(predicted & pairs)
            normalized_count = len(normalized_predicted & pairs)
            self.hard_negative_fp += strict_count
            self.normalized_hard_negative_fp += normalized_count
            self.hard_negative_pairs += len(pairs)
            self.hard_negative_fp_by_phenomenon[phenomenon] += strict_count
            self.normalized_hard_negative_fp_by_phenomenon[phenomenon] += normalized_count
            self.hard_negative_pairs_by_phenomenon[phenomenon] += len(pairs)
        if judge_attempted:
            self.judge_assignments += 1
        if judge is not None:
            self.judge_valid += 1
            if judge.spec_adherence is not None and judge.robustness is not None:
                self.judge_rubric_scored += 1
                self.judge_spec_adherence_total += judge.spec_adherence
                self.judge_robustness_total += judge.robustness
            deterministic_correct = valid and predicted == gold
            judge_says_correct = judge.assessment == "correct"
            self.judge_assessed_correct += int(judge_says_correct)
            self.judge_decision_agreement += int(judge_says_correct == deterministic_correct)
            normalized_deterministic_correct = normalized_valid and normalized_predicted == gold
            self.judge_normalized_decision_agreement += int(
                judge_says_correct == normalized_deterministic_correct
            )
            expected_missed = {(source, target) for source, target, _ in gold - predicted}
            expected_spurious = {(source, target) for source, target, _ in predicted - gold}
            judge_missed = {(pair.source, pair.target) for pair in judge.missed_edges}
            judge_spurious = {(pair.source, pair.target) for pair in judge.spurious_edges}
            self.judge_edge_diagnosis_match += int(
                judge_missed == expected_missed and judge_spurious == expected_spurious
            )
            expected_normalized_missed = {
                (source, target) for source, target, _ in gold - normalized_predicted
            }
            expected_normalized_spurious = {
                (source, target) for source, target, _ in normalized_predicted - gold
            }
            self.judge_normalized_edge_diagnosis_match += int(
                judge_missed == expected_normalized_missed
                and judge_spurious == expected_normalized_spurious
            )

    def to_metrics(self) -> dict[str, Any]:
        precision = _divide(self.tp, self.tp + self.fp)
        recall = _divide(self.tp, self.tp + self.fn)
        if precision is None or recall is None or precision + recall == 0:
            f1 = 0.0 if precision is not None and recall is not None else None
        else:
            f1 = 2 * precision * recall / (precision + recall)
        normalized_precision = _divide(
            self.normalized_tp, self.normalized_tp + self.normalized_fp
        )
        normalized_recall = _divide(
            self.normalized_tp, self.normalized_tp + self.normalized_fn
        )
        if (
            normalized_precision is None
            or normalized_recall is None
            or normalized_precision + normalized_recall == 0
        ):
            normalized_f1 = (
                0.0
                if normalized_precision is not None and normalized_recall is not None
                else None
            )
        else:
            normalized_f1 = (
                2
                * normalized_precision
                * normalized_recall
                / (normalized_precision + normalized_recall)
            )
        hard_negative_rates = {
            phenomenon: _divide(
                self.hard_negative_fp_by_phenomenon[phenomenon],
                self.hard_negative_pairs_by_phenomenon[phenomenon],
            )
            for phenomenon in sorted(self.hard_negative_pairs_by_phenomenon)
            if self.hard_negative_pairs_by_phenomenon[phenomenon]
        }
        normalized_hard_negative_rates = {
            phenomenon: _divide(
                self.normalized_hard_negative_fp_by_phenomenon[phenomenon],
                self.hard_negative_pairs_by_phenomenon[phenomenon],
            )
            for phenomenon in sorted(self.hard_negative_pairs_by_phenomenon)
            if self.hard_negative_pairs_by_phenomenon[phenomenon]
        }
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
            "normalized_valid_json_rate": _divide(self.normalized_valid, self.assignments),
            "normalized_exact_graph_match_rate": _divide(
                self.normalized_exact, self.assignments
            ),
            "normalized_edge_precision": normalized_precision,
            "normalized_edge_recall": normalized_recall,
            "normalized_edge_f1": normalized_f1,
            "normalized_true_positive_edges": self.normalized_tp,
            "normalized_false_positive_edges": self.normalized_fp,
            "normalized_false_negative_edges": self.normalized_fn,
            "annotated_hard_negative_false_positive_rate": _divide(
                self.hard_negative_fp, self.hard_negative_pairs
            ),
            "normalized_annotated_hard_negative_false_positive_rate": _divide(
                self.normalized_hard_negative_fp, self.hard_negative_pairs
            ),
            "hard_negative_false_positive_rate_by_phenomenon": hard_negative_rates,
            "normalized_hard_negative_false_positive_rate_by_phenomenon": (
                normalized_hard_negative_rates
            ),
            "strict_false_positive_rate_on_topically_related_nonresponses": hard_negative_rates.get(
                HardNegativePhenomenon.TOPICAL_NONRESPONSE.value
            ),
            "false_positive_rate_on_topically_related_nonresponses": normalized_hard_negative_rates.get(
                HardNegativePhenomenon.TOPICAL_NONRESPONSE.value
            ),
            "fixed_judge_valid_json_rate": _divide(self.judge_valid, self.judge_assignments),
            "fixed_judge_rubric_score_rate": _divide(
                self.judge_rubric_scored, self.judge_assignments
            ),
            "mean_spec_adherence": _divide(
                self.judge_spec_adherence_total, self.judge_rubric_scored
            ),
            "mean_robustness": _divide(
                self.judge_robustness_total, self.judge_rubric_scored
            ),
            "fixed_judge_assessed_correct_rate": _divide(
                self.judge_assessed_correct, self.judge_assignments
            ),
            "fixed_judge_decision_agreement_rate": _divide(
                self.judge_decision_agreement, self.judge_assignments
            ),
            "fixed_judge_edge_diagnosis_match_rate": _divide(
                self.judge_edge_diagnosis_match, self.judge_assignments
            ),
            "fixed_judge_normalized_decision_agreement_rate": _divide(
                self.judge_normalized_decision_agreement, self.judge_assignments
            ),
            "fixed_judge_normalized_edge_diagnosis_match_rate": _divide(
                self.judge_normalized_edge_diagnosis_match, self.judge_assignments
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

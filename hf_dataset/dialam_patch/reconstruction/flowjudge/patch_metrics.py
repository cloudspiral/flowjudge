from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError, model_validator

from .dialam import NormalizedRelationLabel
from .patch_data import PatchExample, PatchRelation, load_jsonl
from .schemas import StrictModel


class PredictedPatch(StrictModel):
    relations: list[PatchRelation]

    @model_validator(mode="after")
    def reject_duplicates(self) -> "PredictedPatch":
        keys = [(edge.source, edge.target, edge.type) for edge in self.relations]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate predicted relations are not allowed")
        return self


@dataclass(frozen=True)
class ParsedPatchPrediction:
    json_valid: bool
    schema_valid: bool
    relations: tuple[PatchRelation, ...]
    error: str | None


def parse_patch_prediction(raw_response: str) -> ParsedPatchPrediction:
    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        return ParsedPatchPrediction(
            json_valid=False,
            schema_valid=False,
            relations=(),
            error=f"invalid JSON: {exc.msg}",
        )
    try:
        parsed = PredictedPatch.model_validate(payload)
        return ParsedPatchPrediction(
            json_valid=True,
            schema_valid=True,
            relations=tuple(parsed.relations),
            error=None,
        )
    except ValidationError as exc:
        salvaged: list[PatchRelation] = []
        if isinstance(payload, dict) and isinstance(payload.get("relations"), list):
            for item in payload["relations"]:
                try:
                    salvaged.append(PatchRelation.model_validate(item))
                except ValidationError:
                    continue
        return ParsedPatchPrediction(
            json_valid=True,
            schema_valid=False,
            relations=tuple(salvaged),
            error=f"schema validation failed: {exc}",
        )


def _safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _edge_key(relation: PatchRelation) -> tuple[str, str, NormalizedRelationLabel]:
    return relation.source, relation.target, relation.type


def score_patch_predictions(
    examples: list[PatchExample],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    record_by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        example_id = record.get("example_id")
        if not isinstance(example_id, str):
            raise ValueError("each prediction record requires a string example_id")
        if example_id in record_by_id:
            raise ValueError(f"duplicate prediction for {example_id}")
        record_by_id[example_id] = record
    known_ids = {item.example_id for item in examples}
    unknown_ids = sorted(record_by_id.keys() - known_ids)
    if unknown_ids:
        raise ValueError(f"predictions contain unknown example IDs: {unknown_ids}")

    json_valid_count = 0
    schema_valid_count = 0
    exact_count = 0
    invalid_id_count = 0
    invalid_relation_count = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    by_label: dict[NormalizedRelationLabel, Counter[str]] = {
        label: Counter() for label in NormalizedRelationLabel
    }
    direction_correct = 0
    direction_evaluable = 0
    false_edges_by_update: dict[str, set[tuple[str, str, NormalizedRelationLabel]]] = defaultdict(set)
    failures: list[dict[str, Any]] = []

    for example in examples:
        record = record_by_id.get(example.example_id)
        raw_response = record.get("raw_response") if record is not None else None
        if not isinstance(raw_response, str):
            parsed = ParsedPatchPrediction(
                json_valid=False,
                schema_valid=False,
                relations=(),
                error="missing string raw_response",
            )
        else:
            parsed = parse_patch_prediction(raw_response)
        json_valid_count += parsed.json_valid
        schema_valid_count += parsed.schema_valid

        predicted = {_edge_key(item) for item in parsed.relations}
        gold = {_edge_key(item) for item in example.gold_patch.relations}
        allowed_targets = {item.id for item in example.earlier_propositions}
        invalid_relations: set[tuple[str, str, NormalizedRelationLabel]] = set()
        for relation in parsed.relations:
            invalid_endpoints = int(relation.source != example.new_proposition.id) + int(
                relation.target not in allowed_targets
            )
            invalid_id_count += invalid_endpoints
            if invalid_endpoints:
                invalid_relations.add(_edge_key(relation))
        invalid_relation_count += len(invalid_relations)

        tp = predicted & gold
        fp = predicted - gold
        fn = gold - predicted
        true_positive += len(tp)
        false_positive += len(fp)
        false_negative += len(fn)
        false_edges_by_update[example.update_id].update(fp)
        if parsed.schema_valid and not invalid_relations and predicted == gold:
            exact_count += 1
        else:
            failures.append(
                {
                    "example_id": example.example_id,
                    "error": parsed.error,
                    "invalid_relations": sorted(
                        (source, target, label.value) for source, target, label in invalid_relations
                    ),
                    "false_positive": sorted(
                        (source, target, label.value) for source, target, label in fp
                    ),
                    "false_negative": sorted(
                        (source, target, label.value) for source, target, label in fn
                    ),
                }
            )

        for label in NormalizedRelationLabel:
            label_predicted = {item for item in predicted if item[2] == label}
            label_gold = {item for item in gold if item[2] == label}
            by_label[label]["tp"] += len(label_predicted & label_gold)
            by_label[label]["fp"] += len(label_predicted - label_gold)
            by_label[label]["fn"] += len(label_gold - label_predicted)

        gold_by_unordered = {
            (frozenset((source, target)), label): (source, target)
            for source, target, label in gold
        }
        for source, target, label in predicted:
            gold_direction = gold_by_unordered.get((frozenset((source, target)), label))
            if gold_direction is None:
                continue
            direction_evaluable += 1
            if (source, target) == gold_direction:
                direction_correct += 1

    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    per_label: dict[str, dict[str, float | int]] = {}
    label_f1_values: list[float] = []
    for label in NormalizedRelationLabel:
        counts = by_label[label]
        label_precision = _safe_divide(counts["tp"], counts["tp"] + counts["fp"])
        label_recall = _safe_divide(counts["tp"], counts["tp"] + counts["fn"])
        label_f1 = _f1(label_precision, label_recall)
        label_f1_values.append(label_f1)
        per_label[label.value] = {
            "precision": label_precision,
            "recall": label_recall,
            "f1": label_f1,
            "true_positive": counts["tp"],
            "false_positive": counts["fp"],
            "false_negative": counts["fn"],
        }

    update_count = len({item.update_id for item in examples})
    return {
        "example_count": len(examples),
        "update_count": update_count,
        "json_validity_rate": _safe_divide(json_valid_count, len(examples)),
        "schema_validity_rate": _safe_divide(schema_valid_count, len(examples)),
        "invalid_id_count": invalid_id_count,
        "invalid_relation_count": invalid_relation_count,
        "exact_patch_accuracy": _safe_divide(exact_count, len(examples)),
        "edge_precision": precision,
        "edge_recall": recall,
        "edge_f1": _f1(precision, recall),
        "relation_macro_f1": sum(label_f1_values) / len(label_f1_values),
        "relation_metrics": per_label,
        "direction_accuracy": (
            _safe_divide(direction_correct, direction_evaluable)
            if direction_evaluable
            else None
        ),
        "direction_evaluable_edge_count": direction_evaluable,
        "false_edges_per_update": _safe_divide(
            sum(len(items) for items in false_edges_by_update.values()),
            update_count,
        ),
        "true_positive_edges": true_positive,
        "false_positive_edges": false_positive,
        "false_negative_edges": false_negative,
        "failure_cases": failures,
    }


def score_patch_scenario_predictions(
    examples: list[PatchExample],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Union block-level patches, then score exactness once per update.

    Every block remains a separate behavior-spec input and model response. A
    scenario is exact only when every block response is schema-valid, every
    predicted endpoint is valid for that specific block, and the union of all
    block predictions exactly equals the union of the block gold patches.
    """

    record_by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        example_id = record.get("example_id")
        if not isinstance(example_id, str):
            raise ValueError("each prediction record requires a string example_id")
        if example_id in record_by_id:
            raise ValueError(f"duplicate prediction for {example_id}")
        record_by_id[example_id] = record
    known_ids = {item.example_id for item in examples}
    unknown_ids = sorted(record_by_id.keys() - known_ids)
    if unknown_ids:
        raise ValueError(f"predictions contain unknown example IDs: {unknown_ids}")

    examples_by_update: dict[str, list[PatchExample]] = defaultdict(list)
    for example in examples:
        examples_by_update[example.update_id].append(example)

    exact_count = 0
    failures: list[dict[str, Any]] = []
    for update_id, blocks in sorted(examples_by_update.items()):
        blocks.sort(key=lambda item: item.block_index)
        predicted_union: set[tuple[str, str, NormalizedRelationLabel]] = set()
        gold_union: set[tuple[str, str, NormalizedRelationLabel]] = set()
        block_failures: list[dict[str, Any]] = []
        for block in blocks:
            record = record_by_id.get(block.example_id)
            raw_response = record.get("raw_response") if record is not None else None
            if not isinstance(raw_response, str):
                parsed = ParsedPatchPrediction(
                    json_valid=False,
                    schema_valid=False,
                    relations=(),
                    error="missing string raw_response",
                )
            else:
                parsed = parse_patch_prediction(raw_response)
            predicted = {_edge_key(item) for item in parsed.relations}
            gold = {_edge_key(item) for item in block.gold_patch.relations}
            predicted_union.update(predicted)
            gold_union.update(gold)
            allowed_targets = {item.id for item in block.earlier_propositions}
            invalid = sorted(
                (source, target, label.value)
                for source, target, label in predicted
                if source != block.new_proposition.id or target not in allowed_targets
            )
            if not parsed.schema_valid or invalid:
                block_failures.append(
                    {
                        "example_id": block.example_id,
                        "error": parsed.error,
                        "invalid_relations": invalid,
                    }
                )

        missed = gold_union - predicted_union
        spurious = predicted_union - gold_union
        if not block_failures and not missed and not spurious:
            exact_count += 1
        else:
            failures.append(
                {
                    "update_id": update_id,
                    "block_failures": block_failures,
                    "false_positive": sorted(
                        (source, target, label.value)
                        for source, target, label in spurious
                    ),
                    "false_negative": sorted(
                        (source, target, label.value)
                        for source, target, label in missed
                    ),
                }
            )

    scenario_count = len(examples_by_update)
    return {
        "scenario_count": scenario_count,
        "block_example_count": len(examples),
        "exact_scenario_patch_count": exact_count,
        "exact_scenario_patch_accuracy": _safe_divide(exact_count, scenario_count),
        "scenario_failure_cases": failures,
    }


def score_patch_prediction_file(
    predictions_path: Path,
    examples_path: Path,
) -> dict[str, Any]:
    examples = load_jsonl(examples_path, PatchExample)
    records: list[dict[str, Any]] = []
    with predictions_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid predictions line {line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"prediction line {line_number} is not an object")
            records.append(record)
    return {
        **score_patch_predictions(examples, records),
        **score_patch_scenario_predictions(examples, records),
    }

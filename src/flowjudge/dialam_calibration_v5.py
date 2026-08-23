from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from .dialam_training_v5 import PairwiseLabel, assemble_pairwise_patch
from .patch_data import PatchExample
from .patch_metrics import parse_patch_prediction
from .patch_model_eval import deterministic_model_metrics


POSITIVE_TIE_ORDER: tuple[PairwiseLabel, ...] = (
    "SUPPORT",
    "ATTACK",
    "REPHRASE",
)
CALIBRATION_MARGINS: tuple[float, ...] = tuple(
    round(index * 0.05, 2) for index in range(61)
)


def calibrated_pairwise_label(
    label_scores: dict[str, float],
    *,
    margin: float,
) -> PairwiseLabel:
    if margin < 0:
        raise ValueError("NONE margin must be nonnegative")
    if set(label_scores) != {"NONE", *POSITIVE_TIE_ORDER}:
        raise ValueError("pairwise label scores do not cover the four fixed labels")
    best_positive = max(
        POSITIVE_TIE_ORDER,
        key=lambda label: (
            label_scores[label],
            -POSITIVE_TIE_ORDER.index(label),
        ),
    )
    if label_scores[best_positive] - label_scores["NONE"] > margin:
        return best_positive
    return "NONE"


def calibrate_prediction_rows(
    examples: list[PatchExample],
    predictions: list[dict[str, Any]],
    *,
    margin: float,
    dataset_version: str = "v5.1",
) -> list[dict[str, Any]]:
    if not dataset_version.strip():
        raise ValueError("calibrated dataset version must be nonempty")
    examples_by_id = {item.example_id: item for item in examples}
    if len(examples_by_id) != len(examples):
        raise ValueError("calibration examples contain duplicate IDs")
    if {str(row.get("example_id", "")) for row in predictions} != set(examples_by_id):
        raise ValueError("calibration prediction coverage differs from the examples")

    calibrated: list[dict[str, Any]] = []
    for original in predictions:
        row = deepcopy(original)
        example = examples_by_id[row["example_id"]]
        decisions = row.get("pairwise_decisions")
        if not isinstance(decisions, list):
            raise ValueError("calibration requires persisted pairwise decisions")
        by_target = {item.get("target_id"): item for item in decisions}
        candidate_ids = [item.id for item in example.earlier_propositions]
        if len(by_target) != len(decisions) or set(by_target) != set(candidate_ids):
            raise ValueError("pairwise decisions do not cover every supplied ID exactly once")
        labels: list[PairwiseLabel] = []
        ordered_decisions: list[dict[str, Any]] = []
        for candidate_id in candidate_ids:
            decision = by_target[candidate_id]
            scores = decision.get("label_scores")
            if not isinstance(scores, dict):
                raise ValueError("pairwise decision is missing label scores")
            label = calibrated_pairwise_label(scores, margin=margin)
            decision["calibrated_label"] = label
            labels.append(label)
            ordered_decisions.append(decision)
        row["uncalibrated_raw_response"] = row["raw_response"]
        row["raw_response"] = assemble_pairwise_patch(
            example.new_proposition.id,
            candidate_ids,
            labels,
        )
        row["pairwise_decisions"] = ordered_decisions
        row["dataset_version"] = dataset_version
        row["calibration_margin"] = margin
        calibrated.append(row)
    return calibrated


def none_diagnostics(
    examples: list[PatchExample],
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    predictions_by_id = {row["example_id"]: row for row in predictions}
    none_examples = [item for item in examples if not item.gold_patch.relations]
    with_false_edges = 0
    false_edge_labels: dict[str, int] = {}
    for example in none_examples:
        parsed = parse_patch_prediction(
            predictions_by_id[example.example_id]["raw_response"]
        )
        if parsed.relations:
            with_false_edges += 1
        for relation in parsed.relations:
            label = relation.type.value
            false_edge_labels[label] = false_edge_labels.get(label, 0) + 1
    return {
        "none_scenarios": len(none_examples),
        "scenarios_with_false_edges": with_false_edges,
        "false_positive_rate": (
            with_false_edges / len(none_examples) if none_examples else 0.0
        ),
        "false_edge_labels": dict(sorted(false_edge_labels.items())),
    }


def development_gate_checks(
    metrics: dict[str, Any],
    none: dict[str, Any],
    thresholds: dict[str, Any],
) -> dict[str, bool]:
    return {
        "exact_patch_accuracy": (
            metrics["exact_patch_accuracy"] >= thresholds["exact_patch_accuracy_min"]
        ),
        "edge_f1": metrics["edge_f1"] >= thresholds["edge_f1_min"],
        "relation_macro_f1": (
            metrics["relation_macro_f1"]
            >= thresholds["relation_macro_f1_min"]
        ),
        "false_edges_per_update": (
            metrics["false_edges_per_update"]
            <= thresholds["false_edges_per_update_max"]
        ),
        "none_scenarios_with_false_edges": (
            none["scenarios_with_false_edges"]
            <= thresholds["none_scenarios_with_false_edges_max"]
        ),
        "json_validity": (
            metrics["json_validity_rate"] >= thresholds["json_validity_rate_min"]
        ),
        "schema_validity": (
            metrics["schema_validity_rate"]
            >= thresholds["schema_validity_rate_min"]
        ),
    }


def select_none_margin(
    examples: list[PatchExample],
    predictions: list[dict[str, Any]],
    *,
    thresholds: dict[str, Any],
    margins: tuple[float, ...] = CALIBRATION_MARGINS,
    dataset_version: str = "v5.1",
) -> dict[str, Any]:
    if not margins or tuple(sorted(set(margins))) != margins:
        raise ValueError("calibration margins must be unique and increasing")
    candidates: list[dict[str, Any]] = []
    rows_by_margin: dict[float, list[dict[str, Any]]] = {}
    for margin in margins:
        rows = calibrate_prediction_rows(
            examples,
            predictions,
            margin=margin,
            dataset_version=dataset_version,
        )
        metrics = deterministic_model_metrics(examples, rows)
        none = none_diagnostics(examples, rows)
        checks = development_gate_checks(metrics, none, thresholds)
        rows_by_margin[margin] = rows
        candidates.append(
            {
                "margin": margin,
                "development_gate_passed": all(checks.values()),
                "development_gate_checks": checks,
                "passed_check_count": sum(checks.values()),
                "metrics": {
                    key: value
                    for key, value in metrics.items()
                    if key not in {"failure_cases", "scenario_failure_cases"}
                },
                "none_diagnostics": none,
            }
        )

    def rank(item: dict[str, Any]) -> tuple[Any, ...]:
        metrics = item["metrics"]
        none = item["none_diagnostics"]
        return (
            item["development_gate_passed"],
            metrics["edge_f1"],
            metrics["exact_patch_accuracy"],
            metrics["relation_macro_f1"],
            -metrics["false_edges_per_update"],
            -none["scenarios_with_false_edges"],
            item["margin"],
        )

    selected = max(candidates, key=rank)
    return {
        "selected": selected,
        "selected_predictions": rows_by_margin[selected["margin"]],
        "candidates": candidates,
    }

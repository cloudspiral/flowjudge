from __future__ import annotations

import json

from flowjudge.dialam_calibration_v5 import (
    calibrate_prediction_rows,
    calibrated_pairwise_label,
    select_none_margin,
)
from test_dialam_training_v5 import _example


def _prediction(example, positive_label: str) -> dict:
    decisions = []
    for earlier in example.earlier_propositions:
        if earlier.id == "old-a":
            scores = {
                "NONE": -1.0,
                "SUPPORT": 0.0 if positive_label == "SUPPORT" else -2.0,
                "ATTACK": 0.0 if positive_label == "ATTACK" else -2.0,
                "REPHRASE": -2.0,
            }
        else:
            scores = {
                "NONE": -0.2,
                "SUPPORT": 0.0,
                "ATTACK": -1.0,
                "REPHRASE": -1.0,
            }
        label = max(scores, key=scores.get)
        decisions.append(
            {
                "target_id": earlier.id,
                "label": label,
                "label_scores": scores,
            }
        )
    return {
        "example_id": example.example_id,
        "update_id": example.update_id,
        "target": "tuned",
        "model": "Qwen/Qwen3-0.6B",
        "adapter_size": 8192,
        "dataset_version": "v5",
        "eval_split": "v3_dev",
        "raw_response": json.dumps({"relations": []}),
        "pairwise_decisions": decisions,
    }


def test_calibrated_pairwise_label_uses_none_on_boundary_and_fixed_ties() -> None:
    scores = {"NONE": 0.0, "SUPPORT": 0.0, "ATTACK": 0.0, "REPHRASE": -1.0}

    assert calibrated_pairwise_label(scores, margin=0.0) == "NONE"
    scores["SUPPORT"] = 0.1
    scores["ATTACK"] = 0.1
    assert calibrated_pairwise_label(scores, margin=0.0) == "SUPPORT"


def test_margin_calibration_preserves_ids_and_selects_passing_grid_point() -> None:
    examples = [
        _example(update="attack-a", label="ATTACK"),
        _example(update="support-a", label="SUPPORT"),
    ]
    predictions = [
        _prediction(examples[0], "ATTACK"),
        _prediction(examples[1], "SUPPORT"),
    ]
    thresholds = {
        "exact_patch_accuracy_min": 1.0,
        "edge_f1_min": 1.0,
        "relation_macro_f1_min": 2 / 3,
        "false_edges_per_update_max": 0.0,
        "none_scenarios_with_false_edges_max": 0,
        "json_validity_rate_min": 1.0,
        "schema_validity_rate_min": 1.0,
    }

    selection = select_none_margin(
        examples,
        predictions,
        thresholds=thresholds,
        margins=(0.0, 0.25),
    )

    assert selection["selected"]["margin"] == 0.25
    assert selection["selected"]["development_gate_passed"] is True
    calibrated = calibrate_prediction_rows(examples, predictions, margin=0.25)
    assert json.loads(calibrated[0]["raw_response"]) == {
        "relations": [
            {"source": "new-attack-a", "target": "old-a", "type": "ATTACK"}
        ]
    }
    assert calibrated[0]["dataset_version"] == "v5.1"

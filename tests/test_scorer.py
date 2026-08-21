import json

from flowjudge.data import load_benchmark
from flowjudge.scorer import parse_prediction, score_records


def _record(case, relations) -> dict[str, str]:
    return {
        "scenario_id": case.scenario.scenario_id,
        "raw_response": json.dumps({"relations": relations}),
    }


def test_parse_prediction_is_strict_about_json_and_schema() -> None:
    assert not parse_prediction("```json\n{\"relations\": []}\n```").valid_json
    assert not parse_prediction('{"relations": [], "commentary": "done"}').valid_json
    assert not parse_prediction('{"relations": [{"source":"U2","target":"U1","type":"supports"}]}').valid_json
    assert parse_prediction('{"relations": []}').valid_json


def test_exact_gold_predictions_score_perfectly() -> None:
    cases = load_benchmark()
    records = [
        _record(case, [edge.model_dump(exclude={"explanation"}) for edge in case.gold.gold_relations])
        for case in cases
    ]

    summary = score_records(cases, records)

    assert summary["valid_json_rate"] == 1.0
    assert summary["exact_graph_match_rate"] == 1.0
    assert summary["edge_precision"] == 1.0
    assert summary["edge_recall"] == 1.0
    assert summary["edge_f1"] == 1.0
    assert summary["annotated_hard_negative_false_positive_rate"] == 0.0


def test_annotated_hard_negative_false_positive_rate() -> None:
    case = next(case for case in load_benchmark() if len(case.gold.hard_negatives) == 2)
    hard_negative = case.gold.hard_negatives[0]
    records = [
        _record(
            case,
            [{"source": hard_negative.source, "target": hard_negative.target, "type": "responds_to"}],
        )
    ]

    summary = score_records([case], records)

    assert summary["annotated_hard_negative_false_positive_rate"] == 0.5
    assert summary["false_positive_edges"] == 1
    assert summary["exact_graph_match_rate"] == 0.0


def test_invalid_output_counts_as_nonexact_and_misses_gold_edges() -> None:
    case = next(case for case in load_benchmark() if case.gold.gold_relations)
    records = [{"scenario_id": case.scenario.scenario_id, "raw_response": "not json"}]

    summary = score_records([case], records)

    assert summary["valid_json_rate"] == 0.0
    assert summary["exact_graph_match_rate"] == 0.0
    assert summary["false_negative_edges"] == len(case.gold.gold_relations)


def test_model_prompt_and_fixed_judge_summaries_are_secondary() -> None:
    case = load_benchmark()[0]
    record = _record(
        case,
        [edge.model_dump(exclude={"explanation"}) for edge in case.gold.gold_relations],
    )
    record.update(
        {
            "provider": "openai",
            "model": "frontier-model",
            "prompt": "zero_shot",
            "raw_judge_response": json.dumps(
                {
                    "assessment": "correct",
                    "missed_edges": [],
                    "spurious_edges": [],
                    "brief_reason": "The candidate matches the gold graph.",
                }
            ),
        }
    )

    summary = score_records([case], [record])

    combination = summary["by_model_prompt"]["openai:frontier-model:zero_shot"]
    assert combination["exact_graph_match_rate"] == 1.0
    assert summary["fixed_judge_valid_json_rate"] == 1.0
    assert summary["fixed_judge_assessed_correct_rate"] == 1.0
    assert summary["fixed_judge_decision_agreement_rate"] == 1.0
    assert summary["fixed_judge_edge_diagnosis_match_rate"] == 1.0
    assert summary["failure_cases"] == []


def test_fixed_judge_agreement_is_not_the_same_as_saying_correct() -> None:
    case = load_benchmark()[0]
    record = _record(case, [])
    record["raw_judge_response"] = json.dumps(
        {
            "assessment": "incorrect",
            "missed_edges": [
                {"source": edge.source, "target": edge.target}
                for edge in case.gold.gold_relations
            ],
            "spurious_edges": [],
            "brief_reason": "The candidate omits the gold response.",
        }
    )

    summary = score_records([case], [record])

    assert summary["fixed_judge_assessed_correct_rate"] == 0.0
    assert summary["fixed_judge_decision_agreement_rate"] == 1.0
    assert summary["fixed_judge_edge_diagnosis_match_rate"] == 1.0

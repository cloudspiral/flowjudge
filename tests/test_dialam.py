from __future__ import annotations

import json
from pathlib import Path

import pytest

from flowjudge.dialam import (
    DEFAULT_SOURCE_DIR,
    DialogueMapEntry,
    NormalizedRelationLabel,
    OriginalEdgeDirection,
    parse_nodeset,
)
from flowjudge.patch_data import (
    DEFAULT_DIAGNOSTIC_EVAL_EXAMPLES_PATH,
    DEFAULT_DIAGNOSTIC_EVAL_SCENARIOS_PATH,
    DEFAULT_EVAL_EXAMPLES_PATH,
    DEFAULT_EVAL_SCENARIOS_PATH,
    DEFAULT_EVAL_SUMMARY_PATH,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_TRAIN_PATH,
    PatchExample,
    PatchScenario,
    build_smoke_dataset,
    load_generated_dataset,
    load_jsonl,
    validate_patch_datasets,
)
from flowjudge.patch_ceiling import (
    APPROVAL_PHRASE,
    build_judge_prompt,
    evaluate_reliability_gate,
    frozen_judge_rubric_sha256,
    parse_judge_response,
    patch_ceiling_dry_run,
)
from flowjudge.patch_metrics import (
    parse_patch_prediction,
    score_patch_predictions,
    score_patch_scenario_predictions,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_synthetic_nodeset(path: Path) -> None:
    nodes = [
        {"nodeID": "1", "text": "Alice : Schools need more funding", "type": "L"},
        {"nodeID": "2", "text": "Asserting", "type": "YA"},
        {"nodeID": "3", "text": "schools need more funding", "type": "I"},
        {"nodeID": "4", "text": "Default Transition", "type": "TA"},
        {"nodeID": "5", "text": "Bob : Funding improves teacher retention", "type": "L"},
        {"nodeID": "6", "text": "Asserting", "type": "YA"},
        {"nodeID": "7", "text": "funding improves teacher retention", "type": "I"},
        {"nodeID": "8", "text": "Default Transition", "type": "TA"},
        {"nodeID": "9", "text": "Cara : Retention is not the main problem", "type": "L"},
        {"nodeID": "10", "text": "Asserting", "type": "YA"},
        {"nodeID": "11", "text": "retention is not the main problem", "type": "I"},
        {"nodeID": "12", "text": "Default Inference", "type": "RA"},
        {"nodeID": "13", "text": "Default Conflict", "type": "CA"},
        {"nodeID": "14", "text": "Default Rephrase", "type": "MA"},
        {"nodeID": "15", "text": "Default Inference", "type": "RA"},
        {"nodeID": "16", "text": "Arguing", "type": "YA"},
    ]
    raw_edges = [
        ("1", "2"), ("2", "3"),
        ("1", "4"), ("4", "5"),
        ("5", "6"), ("6", "7"),
        ("5", "8"), ("8", "9"),
        ("9", "10"), ("10", "11"),
        ("3", "12"), ("12", "7"), ("16", "12"),
        ("11", "13"), ("13", "7"),
        ("11", "14"), ("14", "3"),
        ("3", "15"), ("7", "15"), ("15", "11"),
    ]
    payload = {
        "nodes": nodes,
        "edges": [
            {"edgeID": str(index), "fromID": source, "toID": target, "formEdgeID": None}
            for index, (source, target) in enumerate(raw_edges, start=100)
        ],
        "locutions": [
            {"nodeID": "1", "personID": "p1", "timestamp": None, "start": "2024-01-01 10:00:00", "end": None, "source": None},
            {"nodeID": "5", "personID": "p2", "timestamp": None, "start": "2024-01-01 10:00:01", "end": None, "source": None},
            {"nodeID": "9", "personID": "p3", "timestamp": None, "start": "2024-01-01 10:00:02", "end": None, "source": None},
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_parser_preserves_grounding_labels_arity_and_original_direction(tmp_path: Path) -> None:
    source_path = tmp_path / "nodeset_fixture.json"
    _write_synthetic_nodeset(source_path)
    parsed = parse_nodeset(
        source_path,
        dialogue=DialogueMapEntry(
            dialogue_id="fixture_dialogue",
            dialogue_title="Fixture dialogue",
            dialogue_order=0,
            map_order=0,
        ),
    )

    assert [item.proposition_id for item in parsed.propositions] == ["3", "7", "11"]
    assert [item.chronological_turn for item in parsed.propositions] == [1, 2, 3]
    assert parsed.propositions[0].groundings[0].raw_locution_id == "1"
    assert parsed.propositions[0].groundings[0].raw_locution_text == "Alice : Schools need more funding"
    assert parsed.propositions[0].groundings[0].speaker == "Alice"

    by_id = {item.relation_node_id: item for item in parsed.relations}
    support = by_id["12"]
    assert support.original_relation_label == "Default Inference"
    assert support.normalized_relation_label == NormalizedRelationLabel.SUPPORT
    assert support.original_source_proposition_ids == ["3"]
    assert support.original_target_proposition_ids == ["7"]
    assert support.original_edge_direction == OriginalEdgeDirection.EARLIER_TO_LATER
    assert support.canonical_source_proposition_id == "7"
    assert support.canonical_target_proposition_id == "3"
    assert support.is_binary_direct is True
    assert support.is_unambiguous_binary is True

    attack = by_id["13"]
    assert attack.normalized_relation_label == NormalizedRelationLabel.ATTACK
    assert attack.original_edge_direction == OriginalEdgeDirection.LATER_TO_EARLIER
    assert attack.canonical_source_proposition_id == "11"
    assert attack.canonical_target_proposition_id == "7"

    nary = by_id["15"]
    assert nary.original_source_proposition_ids == ["3", "7"]
    assert nary.original_target_proposition_ids == ["11"]
    assert nary.is_binary_direct is False
    assert nary.is_unambiguous_binary is False


def test_generated_smoke_dataset_has_locked_sizes_coverage_and_no_dialogue_leakage() -> None:
    train, eval_scenarios = load_generated_dataset()
    eval_examples = load_jsonl(DEFAULT_EVAL_EXAMPLES_PATH, PatchExample)
    manifest = json.loads(DEFAULT_MANIFEST_PATH.read_text(encoding="utf-8"))

    assert len(train) == 160
    assert len(eval_scenarios) == 30
    assert len(eval_examples) == 46
    assert {item.dialogue_id for item in train}.isdisjoint(
        {item.dialogue_id for item in eval_scenarios}
    )
    assert manifest["statistics"]["train_positive_block_count"] == 58
    assert manifest["statistics"]["train_all_negative_block_count"] == 102
    assert manifest["statistics"]["eval_positive_block_count"] == 18
    assert manifest["statistics"]["eval_all_negative_block_count"] == 28
    funnel = manifest["corpus_audit"]["relation_retention_funnel"]
    assert funnel["identity_key"] == [
        "parent_dialogue_id",
        "ordered_original_source_proposition_ids",
        "original_relation_node_id",
        "ordered_original_target_proposition_ids",
    ]
    assert funnel["stages"]["raw"]["total"] == 12_767
    assert funnel["stages"]["unique"]["total"] == 12_767
    assert funnel["stages"]["binary"]["total"] == 11_961
    assert funnel["stages"]["grounded"]["total"] == 9_976
    assert funnel["stages"]["chronology_resolvable"]["total"] == 9_791
    assert funnel["stages"]["behavior_compatible"]["total"] == 8_450
    assert funnel["stages"]["retained"]["total"] == 8_443
    assert funnel["observed_unique_minus_published"] == 1_949
    assert set(manifest["statistics"]["train_relation_counts"]) == {
        "SUPPORT", "ATTACK", "REPHRASE"
    }
    assert set(manifest["statistics"]["eval_relation_counts"]) == {
        "SUPPORT", "ATTACK", "REPHRASE"
    }

    for rows in (train, eval_examples):
        by_update: dict[str, list[PatchExample]] = {}
        for row in rows:
            by_update.setdefault(row.update_id, []).append(row)
            block_ids = {item.id for item in row.earlier_propositions}
            assert all(edge.target in block_ids for edge in row.gold_patch.relations)
            assert all(edge.source == row.new_proposition.id for edge in row.gold_patch.relations)
        for blocks in by_update.values():
            blocks.sort(key=lambda item: item.block_index)
            assert [item.block_index for item in blocks] == list(range(blocks[0].block_count))
            covered = [item.id for block in blocks for item in block.earlier_propositions]
            assert len(covered) == len(set(covered))


def test_balanced_diagnostic_eval_preserves_parent_episodes_and_class_balance() -> None:
    scenarios = load_jsonl(DEFAULT_DIAGNOSTIC_EVAL_SCENARIOS_PATH, PatchScenario)
    examples = load_jsonl(DEFAULT_DIAGNOSTIC_EVAL_EXAMPLES_PATH, PatchExample)
    summary = json.loads(DEFAULT_EVAL_SUMMARY_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(DEFAULT_MANIFEST_PATH.read_text(encoding="utf-8"))

    assert len(scenarios) == 30
    assert len(examples) == 30
    assert all(len(item.blocks) == 1 for item in scenarios)
    assert summary["frozen_split_unit"] == "original QT30 parent episode/subcorpus"
    assert summary["balanced_diagnostic"]["relation_counts"] == {
        "ATTACK": 8,
        "REPHRASE": 8,
        "SUPPORT": 8,
    }
    assert summary["balanced_diagnostic"]["all_negative_scenario_count"] == 6
    assert set(summary["heldout_dialogue_ids"]) == set(
        manifest["validation"]["eval_dialogue_ids"]
    )
    assert {item.dialogue_id for item in examples}.isdisjoint(
        set(manifest["validation"]["train_dialogue_ids"])
    )

    smoke_scenarios = load_jsonl(DEFAULT_EVAL_SCENARIOS_PATH, PatchScenario)
    assert len(smoke_scenarios) == 30
    assert sum(len(item.blocks) for item in smoke_scenarios) == 46


def test_patch_prediction_parser_is_strict_about_schema_but_tracks_json_validity() -> None:
    assert parse_patch_prediction('{"relations": []}').schema_valid
    fenced = parse_patch_prediction('```json\n{"relations": []}\n```')
    assert fenced.json_valid is False
    assert fenced.schema_valid is False
    extra = parse_patch_prediction('{"relations": [], "commentary": "done"}')
    assert extra.json_valid is True
    assert extra.schema_valid is False
    bad_label = parse_patch_prediction(
        '{"relations": [{"source":"2","target":"1","type":"responds_to"}]}'
    )
    assert bad_label.json_valid is True
    assert bad_label.schema_valid is False


def test_perfect_eval_predictions_score_all_requested_metrics_perfectly() -> None:
    examples = load_jsonl(DEFAULT_EVAL_EXAMPLES_PATH, PatchExample)
    records = [
        {
            "example_id": example.example_id,
            "raw_response": example.gold_patch.model_dump_json(),
        }
        for example in examples
    ]

    metrics = score_patch_predictions(examples, records)

    assert metrics["json_validity_rate"] == 1.0
    assert metrics["schema_validity_rate"] == 1.0
    assert metrics["invalid_id_count"] == 0
    assert metrics["exact_patch_accuracy"] == 1.0
    assert metrics["edge_precision"] == 1.0
    assert metrics["edge_recall"] == 1.0
    assert metrics["edge_f1"] == 1.0
    assert metrics["relation_macro_f1"] == 1.0
    assert metrics["direction_accuracy"] == 1.0
    assert metrics["false_edges_per_update"] == 0.0


def test_reversed_gold_edge_counts_invalid_ids_wrong_direction_and_false_edge() -> None:
    example = next(
        item
        for item in load_jsonl(DEFAULT_EVAL_EXAMPLES_PATH, PatchExample)
        if item.gold_patch.relations
    )
    gold = example.gold_patch.relations[0]
    response = {
        "relations": [
            {"source": gold.target, "target": gold.source, "type": gold.type.value}
        ]
    }

    metrics = score_patch_predictions(
        [example],
        [{"example_id": example.example_id, "raw_response": json.dumps(response)}],
    )

    assert metrics["json_validity_rate"] == 1.0
    assert metrics["schema_validity_rate"] == 1.0
    assert metrics["invalid_id_count"] == 2
    assert metrics["exact_patch_accuracy"] == 0.0
    assert metrics["edge_precision"] == 0.0
    assert metrics["edge_recall"] == 0.0
    assert metrics["direction_accuracy"] == 0.0
    assert metrics["false_edges_per_update"] == 1.0


def test_scenario_metrics_union_every_separate_block_before_exact_match() -> None:
    all_examples = load_jsonl(DEFAULT_EVAL_EXAMPLES_PATH, PatchExample)
    update_id = next(
        item.update_id
        for item in all_examples
        if sum(other.update_id == item.update_id for other in all_examples) > 1
    )
    examples = [item for item in all_examples if item.update_id == update_id]
    records = [
        {
            "example_id": example.example_id,
            "raw_response": example.gold_patch.model_dump_json(),
        }
        for example in examples
    ]

    perfect = score_patch_scenario_predictions(examples, records)
    assert perfect["scenario_count"] == 1
    assert perfect["block_example_count"] == len(examples)
    assert perfect["exact_scenario_patch_accuracy"] == 1.0

    missing_one_block = score_patch_scenario_predictions(examples, records[:-1])
    assert missing_one_block["exact_scenario_patch_accuracy"] == 0.0


def test_prompt_ceiling_dry_run_is_complete_and_makes_no_calls() -> None:
    dry_run = patch_ceiling_dry_run()

    assert dry_run["network_calls_made"] == 0
    assert dry_run["approval_required"] == APPROVAL_PHRASE
    assert dry_run["scenario_count"] == 30
    assert dry_run["block_example_count"] == 30
    assert dry_run["primary_call_count"] == 180
    assert dry_run["judge_call_count"] == 180
    assert dry_run["models"] == {
        "openai": "gpt-5.4-mini-2026-03-17",
        "anthropic": "claude-haiku-4-5-20251001",
        "judge": "gpt-5.4-mini-2026-03-17",
    }
    assert len(dry_run["few_shot_example_ids"]) == 4
    assert dry_run["scenario_scoring"].startswith("union all separately predicted blocks")
    assert dry_run["judge_identity_blinded"] is True


def test_judge_prompt_is_identity_blinded_and_rubric_is_fingerprinted() -> None:
    example = load_jsonl(DEFAULT_DIAGNOSTIC_EVAL_EXAMPLES_PATH, PatchExample)[0]
    prompt = build_judge_prompt(
        example,
        '{"relations":[]}',
        {
            "json_validity_rate": 1.0,
            "schema_validity_rate": 1.0,
            "invalid_id_count": 0,
            "false_positive_edges": 0,
            "false_negative_edges": 1,
        },
    )

    assert "gpt-5.4-mini-2026-03-17" not in prompt
    assert "claude-haiku-4-5-20251001" not in prompt
    assert "zero_shot" not in prompt
    assert "few_shot" not in prompt
    assert frozen_judge_rubric_sha256() == (
        "b27fe77a9adce4fe586327fa556a51352efb76d402b93676df2173225be35f66"
    )


def test_prompt_ceiling_judge_schema_and_two_family_gate() -> None:
    assert parse_judge_response(
        '{"spec_adherence":4,"robustness":4,"brief_reason":"fully compliant"}'
    ) is not None
    assert parse_judge_response(
        '{"assessment":"correct","spec_adherence":4,"robustness":4,"brief_reason":"extra"}'
    ) is None

    passing = {
        "update_count": 30,
        "json_validity_rate": 1.0,
        "schema_validity_rate": 1.0,
        "invalid_id_count": 0,
        "exact_patch_accuracy": 0.9,
        "exact_scenario_patch_accuracy": 0.9,
        "edge_f1": 0.9,
        "relation_macro_f1": 0.8,
        "direction_accuracy": 1.0,
        "false_edges_per_update": 0.1,
        "judge_validity_rate": 1.0,
        "judge_spec_adherence_mean": 3.8,
        "judge_robustness_mean": 3.8,
    }
    below_bar = {**passing, "exact_scenario_patch_accuracy": 0.5}
    complete_below_bar = {
        f"{provider}:model:{prompt_name}": below_bar
        for provider in ("openai", "anthropic")
        for prompt_name in ("zero_shot", "few_shot", "strong_structured")
    }
    gate = evaluate_reliability_gate(complete_below_bar)
    assert gate["status"] == "PASS"
    assert gate["measurement_complete"] is True
    assert gate["bar_clearing_combinations"] == []

    one_clears = {**complete_below_bar, "openai:model:strong_structured": passing}
    cleared_gate = evaluate_reliability_gate(one_clears)
    assert cleared_gate["status"] == "FAIL"
    assert cleared_gate["bar_clearing_combinations"] == [
        "openai:model:strong_structured"
    ]

    incomplete_gate = evaluate_reliability_gate(
        {"openai:model:strong_structured": below_bar}
    )
    assert incomplete_gate["status"] == "FAIL"
    assert incomplete_gate["measurement_complete"] is False


@pytest.mark.skipif(not DEFAULT_SOURCE_DIR.exists(), reason="official QT30 source is intentionally untracked")
def test_generated_artifacts_are_deterministic_and_pass_source_backed_validation(tmp_path: Path) -> None:
    manifest = build_smoke_dataset(output_dir=tmp_path)
    for name in (
        DEFAULT_TRAIN_PATH.name,
        DEFAULT_EVAL_SCENARIOS_PATH.name,
        DEFAULT_EVAL_EXAMPLES_PATH.name,
        DEFAULT_DIAGNOSTIC_EVAL_SCENARIOS_PATH.name,
        DEFAULT_DIAGNOSTIC_EVAL_EXAMPLES_PATH.name,
        DEFAULT_EVAL_SUMMARY_PATH.name,
        DEFAULT_MANIFEST_PATH.name,
    ):
        assert (tmp_path / name).read_bytes() == (PROJECT_ROOT / "data" / "dialam" / name).read_bytes()

    train = load_jsonl(tmp_path / DEFAULT_TRAIN_PATH.name, PatchExample)
    eval_scenarios = load_jsonl(tmp_path / DEFAULT_EVAL_SCENARIOS_PATH.name, PatchScenario)
    from flowjudge.dialam import load_canonical_maps

    validation = validate_patch_datasets(train, eval_scenarios, load_canonical_maps())
    assert all(
        value is True
        for key, value in validation.items()
        if key
        not in {
            "dialogue_level_leakage",
            "train_dialogue_ids",
            "eval_dialogue_ids",
            "split_unit",
        }
    )
    assert validation["dialogue_level_leakage"] is False
    assert manifest["statistics"]["train_example_count"] == 160

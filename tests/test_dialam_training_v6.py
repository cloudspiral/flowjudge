from __future__ import annotations

import json
from collections import Counter

from flowjudge.dialam_training import TrainingMessage
from flowjudge.dialam_training_v5 import CandidateHardness, DialAMTrainingRowV5
from flowjudge.dialam_training_v6 import (
    _eligible_vives_relations,
    convert_v5_target_rows,
    load_vives_binary_corpus,
    materialize_vives_rows,
    select_vives_contrasts,
)


def test_vives_binary_audit_is_complete_and_deterministic() -> None:
    corpus = load_vives_binary_corpus()

    assert corpus.audit["source_units"] == 2932
    assert corpus.audit["relation_slots_seen"] == 2191
    assert corpus.audit["multi_target_slots_excluded"] == 312
    assert corpus.audit["multi_target_edges_excluded"] == 964
    assert corpus.audit["forward_direction_relations_excluded"] == 7
    assert corpus.audit["incomplete_relation_slots_excluded"] == 1
    assert corpus.audit["eligible_unique_binary_relations"] == 1871
    assert {
        label: len(items)
        for label, items in _eligible_vives_relations(corpus).items()
    } == {"SUPPORT": 1452, "REPHRASE": 262, "ATTACK": 151}


def test_vives_rows_keep_balanced_positive_none_pairs_and_exact_blocks() -> None:
    corpus = load_vives_binary_corpus()
    contrasts = select_vives_contrasts(
        corpus,
        target_counts={"ATTACK": 2, "REPHRASE": 2, "SUPPORT": 2},
    )
    rows = materialize_vives_rows(corpus, contrasts)

    assert Counter(item.relation.label for item in contrasts) == {
        "ATTACK": 2,
        "REPHRASE": 2,
        "SUPPORT": 2,
    }
    assert len(rows) == 12
    assert len({row.training_row_id for row in rows}) == 12
    for index in range(0, len(rows), 2):
        positive, negative = rows[index : index + 2]
        assert [positive.pair_role, negative.pair_role] == ["POSITIVE", "NONE"]
        assert positive.pair_group_id == negative.pair_group_id
        assert positive.source_corpus == negative.source_corpus == "VivesDebate"
        assert positive.curriculum_stage == negative.curriculum_stage == "vives_warmup"
        assert positive.messages[1].content == positive.positive_label
        assert negative.messages[1].content == "NONE"
        positive_block = positive.messages[0].content.split("INPUT\n", 1)[1]
        negative_block = negative.messages[0].content.split("INPUT\n", 1)[1]
        assert json.loads(positive_block) == json.loads(negative_block)
        supplied_ids = {
            item["id"]
            for item in json.loads(positive_block)["complete_earlier_comparison_block"]
        }
        assert positive.candidate_target_id in supplied_ids
        assert negative.candidate_target_id in supplied_ids


def test_v5_target_conversion_preserves_messages_and_adds_curriculum_lineage() -> None:
    hardness = CandidateHardness(
        shared_content_tokens=2,
        overlap_coefficient=0.5,
        jaccard_similarity=0.25,
        turn_gap=3,
    )
    v5_rows = [
        DialAMTrainingRowV5(
            schema_version="dialam_qlora_pairwise_v5",
            training_row_id=f"v5-row-{role.lower()}",
            pair_group_id="v5-pair",
            pair_role=role,
            source_example_id="qt30-example",
            update_id="qt30-update",
            dialogue_id="qt30-parent",
            block_index=0,
            candidate_target_id=candidate,
            paired_candidate_target_id=paired,
            label=label,
            positive_label="ATTACK",
            repetition_index=0,
            is_repeated_positive=False,
            negative_hardness=hardness,
            assistant_loss_weight=1.0,
            messages=[
                TrainingMessage(role="user", content=f"fixed prompt for {candidate}"),
                TrainingMessage(role="assistant", content=label),
            ],
        )
        for role, candidate, paired, label in (
            ("POSITIVE", "old-a", "old-b", "ATTACK"),
            ("NONE", "old-b", "old-a", "NONE"),
        )
    ]

    converted = convert_v5_target_rows(v5_rows)

    assert [row.curriculum_stage for row in converted] == ["qt30_target"] * 2
    assert [row.source_corpus for row in converted] == ["DialAM-QT30"] * 2
    assert [row.messages for row in converted] == [row.messages for row in v5_rows]
    assert converted[0].pair_group_id == converted[1].pair_group_id
    assert converted[0].training_row_id != v5_rows[0].training_row_id

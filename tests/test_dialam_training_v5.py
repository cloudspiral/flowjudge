from __future__ import annotations

import json
from collections import Counter

from flowjudge.dialam_training_v5 import (
    assemble_pairwise_patch,
    build_pairwise_eval_input,
    candidate_label_distribution,
    materialize_v5_rows,
    pairwise_decisions,
    select_v5_contrasts,
)
from flowjudge.patch_data import PatchExample


def _example(
    *,
    update: str,
    label: str,
    positive_target: str = "old-a",
) -> PatchExample:
    source_id = f"new-{update}"
    earlier = [
        {
            "id": target_id,
            "text": text,
            "chronological_turn": turn,
            "speaker_id": f"speaker-{turn}",
            "speaker": f"Speaker {turn}",
            "raw_locution_id": f"loc-{update}-{turn}",
            "raw_locution_text": text,
        }
        for target_id, text, turn in (
            ("old-a", "The proposal will reduce costs", 1),
            ("old-b", "The proposal may change access", 2),
            ("old-c", "A separate historical observation", 3),
        )
    ]
    relation = {
        "source": source_id,
        "target": positive_target,
        "type": label,
    }
    provenance = {
        "relation_node_id": f"relation-{update}",
        "original_relation_label": {
            "SUPPORT": "RA",
            "ATTACK": "CA",
            "REPHRASE": "MA",
        }[label],
        "normalized_relation_label": label,
        "original_source_proposition_ids": [source_id],
        "original_target_proposition_ids": [positive_target],
        "original_edge_direction": "source_to_relation_to_target",
        "canonical_source_proposition_id": source_id,
        "canonical_target_proposition_id": positive_target,
    }
    return PatchExample.model_validate(
        {
            "schema_version": "dialam_incremental_patch_v1",
            "example_id": f"{update}:b0",
            "update_id": update,
            "split": "train",
            "dialogue_id": "episode-train",
            "dialogue_title": "Episode",
            "map_id": f"map-{update}",
            "block_index": 0,
            "block_count": 1,
            "block_size": 8,
            "new_proposition": {
                "id": source_id,
                "text": "This directly changes the proposal's expected costs",
                "chronological_turn": 4,
                "speaker_id": "speaker-new",
                "speaker": "New Speaker",
                "raw_locution_id": f"loc-new-{update}",
                "raw_locution_text": "This directly changes the proposal's expected costs",
            },
            "earlier_propositions": earlier,
            "gold_patch": {"relations": [relation]},
            "gold_relation_provenance": [provenance],
        }
    )


def test_pairwise_decisions_cover_every_candidate_once() -> None:
    example = _example(update="attack-a", label="ATTACK")

    decisions, ambiguous = pairwise_decisions(example)

    assert ambiguous == 0
    assert [(item.candidate_target_id, item.label) for item in decisions] == [
        ("old-a", "ATTACK"),
        ("old-b", "NONE"),
        ("old-c", "NONE"),
    ]


def test_candidate_label_distribution_exposes_natural_none_prevalence() -> None:
    counts, ambiguous = candidate_label_distribution(
        [
            _example(update="attack-a", label="ATTACK"),
            _example(update="support-a", label="SUPPORT"),
        ]
    )

    assert ambiguous == 0
    assert counts == {"ATTACK": 1, "SUPPORT": 1, "NONE": 4}


def test_v5_contrasts_pair_positive_and_hard_none_from_exact_block() -> None:
    examples = [
        _example(update="attack-a", label="ATTACK"),
        _example(update="attack-b", label="ATTACK", positive_target="old-b"),
        _example(update="support-a", label="SUPPORT"),
        _example(update="rephrase-a", label="REPHRASE"),
    ]

    contrasts = select_v5_contrasts(
        examples,
        target_counts={"ATTACK": 3, "REPHRASE": 1, "SUPPORT": 1},
    )

    assert Counter(item.positive.label for item in contrasts) == {
        "ATTACK": 3,
        "REPHRASE": 1,
        "SUPPORT": 1,
    }
    assert all(item.negative.label == "NONE" for item in contrasts)
    assert all(
        item.positive.example.example_id == item.negative.example.example_id
        for item in contrasts
    )
    attack_occurrences = Counter(
        item.positive.key for item in contrasts if item.positive.label == "ATTACK"
    )
    assert sorted(attack_occurrences.values()) == [1, 2]


def test_v5_rows_keep_each_contrast_in_one_two_row_batch() -> None:
    contrasts = select_v5_contrasts(
        [
            _example(update="attack-a", label="ATTACK"),
            _example(update="support-a", label="SUPPORT"),
            _example(update="rephrase-a", label="REPHRASE"),
        ],
        target_counts={"ATTACK": 1, "REPHRASE": 1, "SUPPORT": 1},
    )

    rows = materialize_v5_rows(contrasts)

    assert len(rows) == 6
    assert len({row.training_row_id for row in rows}) == 6
    for index in range(0, len(rows), 2):
        positive, negative = rows[index : index + 2]
        assert positive.pair_group_id == negative.pair_group_id
        assert [positive.pair_role, negative.pair_role] == ["POSITIVE", "NONE"]
        assert positive.source_example_id == negative.source_example_id
        assert positive.messages[1].content == positive.positive_label
        assert negative.messages[1].content == "NONE"
        assert "complete_earlier_comparison_block" in positive.messages[0].content


def test_pairwise_eval_input_and_assembly_preserve_supplied_ids() -> None:
    example = _example(update="support-a", label="SUPPORT")
    eval_input = build_pairwise_eval_input(example)

    assert [item.target_id for item in eval_input.candidates] == [
        "old-a",
        "old-b",
        "old-c",
    ]
    raw = assemble_pairwise_patch(
        eval_input.source_id,
        [item.target_id for item in eval_input.candidates],
        ["SUPPORT", "NONE", "REPHRASE"],
    )
    assert json.loads(raw) == {
        "relations": [
            {"source": "new-support-a", "target": "old-a", "type": "SUPPORT"},
            {"source": "new-support-a", "target": "old-c", "type": "REPHRASE"},
        ]
    }

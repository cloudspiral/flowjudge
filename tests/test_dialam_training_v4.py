from __future__ import annotations

from collections import Counter

from flowjudge.dialam_training_v3 import select_paired_examples
from flowjudge.dialam_training_v4 import (
    materialize_v4_rows,
    select_v4_supplement,
)
from flowjudge.patch_data import PatchExample


def _example(*, update: str, block: int, category: str) -> PatchExample:
    earlier_id = f"old-{update}-{block}"
    relations = []
    provenance = []
    if category != "NONE":
        relations = [{"source": f"new-{update}", "target": earlier_id, "type": category}]
        provenance = [
            {
                "relation_node_id": f"relation-{update}-{block}",
                "original_relation_label": "CA" if category == "ATTACK" else "RA",
                "normalized_relation_label": category,
                "original_source_proposition_ids": [f"new-{update}"],
                "original_target_proposition_ids": [earlier_id],
                "original_edge_direction": "source_to_relation_to_target",
                "canonical_source_proposition_id": f"new-{update}",
                "canonical_target_proposition_id": earlier_id,
            }
        ]
    return PatchExample.model_validate(
        {
            "schema_version": "dialam_incremental_patch_v1",
            "example_id": f"{update}:b{block}",
            "update_id": update,
            "split": "train",
            "dialogue_id": "episode-train",
            "dialogue_title": "Episode",
            "map_id": f"map-{update}",
            "block_index": block,
            "block_count": 2,
            "block_size": 8,
            "new_proposition": {
                "id": f"new-{update}",
                "text": "This proposal directly changes the expected result",
                "chronological_turn": 3,
                "speaker_id": "s1",
                "speaker": "Speaker 1",
                "raw_locution_id": f"loc-new-{update}",
                "raw_locution_text": "This proposal directly changes the expected result",
            },
            "earlier_propositions": [
                {
                    "id": earlier_id,
                    "text": "The proposal changes the expected result",
                    "chronological_turn": 2,
                    "speaker_id": "s2",
                    "speaker": "Speaker 2",
                    "raw_locution_id": f"loc-{update}-{block}",
                    "raw_locution_text": "The proposal changes the expected result",
                }
            ],
            "gold_patch": {"relations": relations},
            "gold_relation_provenance": provenance,
        }
    )


def test_v4_supplement_uses_unique_rows_before_attack_repetition() -> None:
    examples = [
        _example(update="attack-a", block=0, category="ATTACK"),
        _example(update="attack-a", block=1, category="NONE"),
        _example(update="attack-b", block=0, category="ATTACK"),
        _example(update="attack-b", block=1, category="NONE"),
        _example(update="attack-c", block=0, category="ATTACK"),
        _example(update="attack-c", block=1, category="NONE"),
        _example(update="negative-only-a", block=0, category="NONE"),
        _example(update="negative-only-b", block=0, category="NONE"),
    ]
    foundation = select_paired_examples(examples, target_counts={"ATTACK": 1})

    supplement = select_v4_supplement(
        examples,
        foundation,
        positive_counts={"ATTACK": 4},
        none_count=2,
    )
    positives = [item for item in supplement if item.sampling_role == "POSITIVE"]
    negatives = [item for item in supplement if item.sampling_role == "NONE"]

    assert len(positives) == 4
    assert len({item.example.example_id for item in positives}) == 3
    assert len({item.example.example_id for item in negatives}) == 2
    assert negatives[0].negative_metadata is not None
    assert negatives[0].negative_metadata.kind == "direct_edge_elsewhere"


def test_v4_rows_have_unique_row_ids_and_mark_only_repeated_copies() -> None:
    example = _example(update="attack", block=0, category="ATTACK")
    selections = [
        *select_v4_supplement(
            [
                example,
                _example(update="negative-a", block=0, category="NONE"),
                _example(update="negative-b", block=0, category="NONE"),
            ],
            [],
            positive_counts={"ATTACK": 3},
            none_count=2,
        )
    ]

    rows = materialize_v4_rows(selections)

    assert len(rows) == 5
    assert len({row.training_row_id for row in rows}) == 5
    assert Counter(row.category for row in rows) == {"ATTACK": 3, "NONE": 2}
    attack_rows = sorted(
        (row for row in rows if row.category == "ATTACK"),
        key=lambda row: row.repetition_index,
    )
    assert [row.repetition_index for row in attack_rows] == [0, 1, 2]
    assert [row.is_repeated_copy for row in attack_rows] == [False, True, True]

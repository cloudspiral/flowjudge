from __future__ import annotations

from flowjudge.dialam_training_v3 import (
    _dialogue_split,
    _negative_metadata,
    _training_row,
    select_paired_examples,
)
from flowjudge.patch_data import PatchExample


def _example(
    *,
    update: str,
    block: int,
    category: str,
    new_text: str = "Renewable energy lowers carbon emissions",
    earlier_text: str = "Carbon emissions fall with renewable energy",
) -> PatchExample:
    earlier_id = f"old-{block}"
    relations = []
    provenance = []
    if category != "NONE":
        relations = [{"source": f"new-{update}", "target": earlier_id, "type": category}]
        provenance = [
            {
                "relation_node_id": f"relation-{update}-{block}",
                "original_relation_label": "RA" if category == "SUPPORT" else "CA",
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
            "dialogue_id": f"episode-{update}",
            "dialogue_title": "Episode",
            "map_id": f"map-{update}",
            "block_index": block,
            "block_count": 2,
            "block_size": 8,
            "new_proposition": {
                "id": f"new-{update}",
                "text": new_text,
                "chronological_turn": 3,
                "speaker_id": "s1",
                "speaker": "Speaker 1",
                "raw_locution_id": f"loc-new-{update}",
                "raw_locution_text": new_text,
            },
            "earlier_propositions": [
                {
                    "id": earlier_id,
                    "text": earlier_text,
                    "chronological_turn": 2,
                    "speaker_id": "s2",
                    "speaker": "Speaker 2",
                    "raw_locution_id": f"loc-{update}-{block}",
                    "raw_locution_text": earlier_text,
                }
            ],
            "gold_patch": {"relations": relations},
            "gold_relation_provenance": provenance,
        }
    )


def test_v3_selection_co_selects_one_positive_and_none_from_each_update() -> None:
    examples = [
        _example(update="support", block=0, category="SUPPORT"),
        _example(update="support", block=1, category="NONE"),
        _example(update="attack", block=0, category="ATTACK"),
        _example(update="attack", block=1, category="NONE"),
    ]

    pairs = select_paired_examples(
        examples,
        target_counts={"SUPPORT": 1, "ATTACK": 1},
    )

    assert len(pairs) == 2
    assert len({item.positive.update_id for item in pairs}) == 2
    assert all(item.positive.update_id == item.negative.update_id for item in pairs)
    assert {item.positive.gold_patch.relations[0].type.value for item in pairs} == {
        "SUPPORT",
        "ATTACK",
    }
    assert all(not item.negative.gold_patch.relations for item in pairs)


def test_v3_pair_rows_are_reciprocal_and_equal_weight() -> None:
    pair = select_paired_examples(
        [
            _example(update="support", block=0, category="SUPPORT"),
            _example(update="support", block=1, category="NONE"),
        ],
        target_counts={"SUPPORT": 1},
    )[0]

    positive = _training_row(pair, "POSITIVE")
    negative = _training_row(pair, "NONE")

    assert positive.paired_example_id == negative.example_id
    assert negative.paired_example_id == positive.example_id
    assert positive.assistant_loss_weight == negative.assistant_loss_weight == 1.0
    assert positive.update_id == negative.update_id
    assert positive.messages[1].content != negative.messages[1].content


def test_v3_negative_metadata_keeps_zero_overlap_same_update_siblings() -> None:
    negative = _example(
        update="zero-overlap",
        block=1,
        category="NONE",
        earlier_text="Completely unrelated municipal zoning proposal",
    )

    metadata = _negative_metadata(negative)

    assert metadata.kind == "same_update_direct_edge_elsewhere"
    assert metadata.max_shared_content_tokens == 0
    assert metadata.max_overlap_coefficient == 0.0


def test_v3_development_episode_split_is_deterministic_and_disjoint() -> None:
    dialogues = {f"episode-{index}" for index in range(8)}

    train_a, dev_a = _dialogue_split(dialogues)
    train_b, dev_b = _dialogue_split(dialogues)

    assert (train_a, dev_a) == (train_b, dev_b)
    assert len(dev_a) == 4
    assert train_a.isdisjoint(dev_a)
    assert train_a | dev_a == dialogues

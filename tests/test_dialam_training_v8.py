from __future__ import annotations

from collections import Counter

import pytest

from flowjudge.dialam_training_v8 import materialize_v8_rows, select_v8_groups
from test_dialam_training_v5 import _example


def test_v8_groups_use_one_positive_and_two_distinct_same_block_negatives() -> None:
    examples = [
        _example(update="attack-a", label="ATTACK"),
        _example(update="rephrase-a", label="REPHRASE"),
        _example(update="support-a", label="SUPPORT"),
    ]
    groups = select_v8_groups(
        examples,
        target_counts={"ATTACK": 1, "REPHRASE": 1, "SUPPORT": 1},
    )
    rows = materialize_v8_rows(groups)

    assert len(groups) == 3
    assert len(rows) == 9
    assert Counter(row.label for row in rows) == {
        "NONE": 6,
        "ATTACK": 1,
        "REPHRASE": 1,
        "SUPPORT": 1,
    }
    for index in range(0, len(rows), 3):
        triple = rows[index : index + 3]
        assert [row.group_role for row in triple] == [
            "POSITIVE",
            "NONE_PRIMARY",
            "NONE_SECONDARY",
        ]
        assert len({row.group_id for row in triple}) == 1
        assert len({row.source_example_id for row in triple}) == 1
        assert len({row.candidate_target_id for row in triple}) == 3
        assert [row.messages[1].content for row in triple] == [
            triple[0].positive_label,
            "NONE",
            "NONE",
        ]


def test_v8_repetition_keeps_two_negatives_distinct() -> None:
    groups = select_v8_groups(
        [_example(update="attack-a", label="ATTACK")],
        target_counts={"ATTACK": 3, "REPHRASE": 0, "SUPPORT": 0},
    )

    assert sorted(group.repetition_index for group in groups) == [0, 1, 2]
    assert all(
        group.primary_negative.candidate_target_id
        != group.secondary_negative.candidate_target_id
        for group in groups
    )


def test_v8_rejects_positive_without_two_none_candidates() -> None:
    example = _example(update="attack-a", label="ATTACK")
    example = example.model_copy(
        update={"earlier_propositions": example.earlier_propositions[:2]}
    )

    with pytest.raises(ValueError, match="two negatives"):
        select_v8_groups(
            [example],
            target_counts={"ATTACK": 1, "REPHRASE": 0, "SUPPORT": 0},
        )

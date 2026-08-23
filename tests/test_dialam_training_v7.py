from __future__ import annotations

from flowjudge.dialam_training import TrainingMessage
from flowjudge.dialam_training_v5 import CandidateHardness, DialAMTrainingRowV5
from flowjudge.dialam_training_v7 import convert_v5_rows_to_preferences


def _v5_pair() -> list[DialAMTrainingRowV5]:
    hardness = CandidateHardness(
        shared_content_tokens=2,
        overlap_coefficient=0.5,
        jaccard_similarity=0.25,
        turn_gap=2,
    )
    return [
        DialAMTrainingRowV5(
            schema_version="dialam_qlora_pairwise_v5",
            training_row_id=f"v5-{role.lower()}",
            pair_group_id="v5-pair",
            pair_role=role,
            source_example_id="example-1",
            update_id="update-1",
            dialogue_id="parent-episode-1",
            block_index=0,
            candidate_target_id=target,
            paired_candidate_target_id=paired,
            label=label,
            positive_label="SUPPORT",
            repetition_index=0,
            is_repeated_positive=False,
            negative_hardness=hardness,
            assistant_loss_weight=1.0,
            messages=[
                TrainingMessage(role="user", content=f"candidate {target}"),
                TrainingMessage(role="assistant", content=label),
            ],
        )
        for role, target, paired, label in (
            ("POSITIVE", "old-positive", "old-hard-none", "SUPPORT"),
            ("NONE", "old-hard-none", "old-positive", "NONE"),
        )
    ]


def test_v7_converts_each_v5_contrast_to_reciprocal_preferences() -> None:
    rows = convert_v5_rows_to_preferences(_v5_pair())

    assert len(rows) == 2
    positive, negative = rows
    assert (positive.chosen_label, positive.rejected_label) == ("SUPPORT", "NONE")
    assert (negative.chosen_label, negative.rejected_label) == ("NONE", "SUPPORT")
    assert positive.pair_group_id == negative.pair_group_id == "v5-pair"
    assert positive.dialogue_id == negative.dialogue_id == "parent-episode-1"
    assert positive.messages[0] == _v5_pair()[0].messages[0]
    assert negative.messages[0] == _v5_pair()[1].messages[0]


def test_v7_conversion_rejects_crossed_pair_groups() -> None:
    rows = _v5_pair()
    rows[1] = rows[1].model_copy(update={"pair_group_id": "different-pair"})

    try:
        convert_v5_rows_to_preferences(rows)
    except ValueError as exc:
        assert "cross pair groups" in str(exc)
    else:
        raise AssertionError("crossed pair groups were accepted")

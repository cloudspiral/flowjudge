from __future__ import annotations

from collections import Counter

from flowjudge.dialam_training_v5 import pairwise_decisions
from flowjudge.dialam_training_v9 import (
    V9_SOURCE_ADAPTER_TREE_SHA256,
    DialAMMiningScoreV9,
    materialize_v9_rows,
    select_v9_mined_none,
    select_v9_mining_candidates,
)
from test_dialam_training_v5 import _example


def _score(candidate_id: str, decision_key: str, hardness: float) -> DialAMMiningScoreV9:
    return DialAMMiningScoreV9(
        schema_version="dialam_selected_model_mining_score_v9",
        mining_candidate_id=candidate_id,
        decision_key=decision_key,
        selected_label="SUPPORT" if hardness > 0 else "NONE",
        winning_positive_label="SUPPORT",
        label_scores={
            "NONE": 0.0,
            "SUPPORT": hardness,
            "ATTACK": hardness - 1.0,
            "REPHRASE": hardness - 2.0,
        },
        model_hardness=hardness,
        support_evidence=hardness,
        adapter_tree_sha256=V9_SOURCE_ADAPTER_TREE_SHA256,
    )


def test_v9_mining_prefilter_excludes_exact_v5_none_decisions() -> None:
    examples = [
        _example(update="attack-a", label="ATTACK"),
        _example(update="support-a", label="SUPPORT"),
    ]
    excluded = next(
        item.key
        for item in pairwise_decisions(examples[0])[0]
        if item.label == "NONE"
    )
    selected = select_v9_mining_candidates(
        examples,
        excluded_none_keys={excluded},
        size=3,
    )

    assert len(selected) == 3
    assert len({item.mining_candidate_id for item in selected}) == 3
    assert all(item.gold_label == "NONE" for item in selected)
    assert excluded not in {item.decision_key for item in selected}


def test_v9_model_hardness_dominates_support_and_lexical_ties() -> None:
    examples = [_example(update="attack-a", label="ATTACK")]
    candidates = select_v9_mining_candidates(
        examples,
        excluded_none_keys=set(),
        size=2,
    )
    scores = [
        _score(candidates[0].mining_candidate_id, candidates[0].decision_key, 1.0),
        _score(candidates[1].mining_candidate_id, candidates[1].decision_key, 4.0),
    ]

    selected = select_v9_mined_none(candidates, scores, size=1)

    assert selected[0][0].mining_candidate_id == candidates[1].mining_candidate_id
    assert selected[0][1].model_hardness == 4.0


def test_v9_rows_preserve_positive_targets_and_mined_none_scores() -> None:
    example = _example(update="attack-a", label="ATTACK")
    positive = next(
        item for item in pairwise_decisions(example)[0] if item.label == "ATTACK"
    )
    candidate = select_v9_mining_candidates(
        [example],
        excluded_none_keys=set(),
        size=1,
    )[0]
    score = _score(candidate.mining_candidate_id, candidate.decision_key, 3.0)
    source = {
        "training_row_id": "source-positive",
        "source_example_id": positive.example.example_id,
        "update_id": positive.example.update_id,
        "dialogue_id": positive.example.dialogue_id,
        "block_index": positive.example.block_index,
        "candidate_target_id": positive.candidate_target_id,
        "label": "ATTACK",
        "messages": [
            {"role": "user", "content": "candidate prompt"},
            {"role": "assistant", "content": "ATTACK"},
        ],
    }

    rows = materialize_v9_rows([source], [(candidate, score)])

    assert Counter(row.label for row in rows) == {"ATTACK": 1, "NONE": 1}
    mined = next(row for row in rows if row.row_role == "MINED_NONE")
    assert mined.mining_score == score
    assert mined.messages[1].content == "NONE"

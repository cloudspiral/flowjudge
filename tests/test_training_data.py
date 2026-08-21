from flowjudge.training_data import (
    TeacherRewrite,
    build_training_prompt,
    DATASET_SIZES,
    OWN_EVAL_DEBATES,
    TRAIN_DEBATES,
    build_training_candidates,
)


def test_source_derived_candidates_are_valid_and_leakage_safe() -> None:
    candidates = build_training_candidates(limit=DATASET_SIZES[-1])

    assert len(candidates) == DATASET_SIZES[-1]
    assert {candidate.source_debate for candidate in candidates} <= set(TRAIN_DEBATES)
    assert not ({candidate.source_debate for candidate in candidates} & set(OWN_EVAL_DEBATES))
    assert all(6 <= len(candidate.units) <= 12 for candidate in candidates)
    assert all(candidate.gold_graph.relations for candidate in candidates)
    assert len({candidate.example_id for candidate in candidates}) == len(candidates)


def test_candidate_generation_is_deterministic() -> None:
    first = build_training_candidates(limit=20)
    second = build_training_candidates(limit=20)

    assert [candidate.model_dump() for candidate in first] == [
        candidate.model_dump() for candidate in second
    ]


def test_training_prompt_never_reveals_anchor_edge() -> None:
    candidate = build_training_candidates(limit=1)[0]
    rewrite = TeacherRewrite(
        units=[unit.model_dump() for unit in candidate.units]
    )

    prompt = build_training_prompt(candidate, rewrite.units)

    later, earlier = candidate.source_anchor_pair
    assert "Source-derived response window" not in prompt
    assert f"U{later} → U{earlier}" not in prompt

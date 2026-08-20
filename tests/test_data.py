from collections import Counter

from flowjudge.data import eligible_response_pairs, load_pilot


def test_pilot_has_two_scenarios_per_category() -> None:
    cases = load_pilot()

    assert len(cases) == 12
    assert set(Counter(case.scenario.category for case in cases).values()) == {2}


def test_every_scenario_and_annotation_is_manually_reviewable() -> None:
    for case in load_pilot():
        scenario = case.scenario
        unit_by_id = {unit.id: unit for unit in scenario.units}

        assert 6 <= len(scenario.units) <= 12
        assert case.gold.design_intent
        assert case.gold.hard_negatives
        for edge in case.gold.gold_relations:
            assert edge.explanation
            assert int(edge.source[1:]) > int(edge.target[1:])
            assert unit_by_id[edge.source].side != unit_by_id[edge.target].side
            assert (edge.source, edge.target) in eligible_response_pairs(scenario)
        for pair in case.gold.hard_negatives:
            assert pair.explanation
            assert int(pair.source[1:]) > int(pair.target[1:])


def test_topically_nonresponsive_scenarios_have_empty_gold_graphs() -> None:
    cases = load_pilot()
    topical_cases = [
        case for case in cases if case.scenario.category.value == "topically_related_nonresponsive"
    ]

    assert len(topical_cases) == 2
    assert all(not case.gold.gold_relations for case in topical_cases)

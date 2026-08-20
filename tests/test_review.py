import html

from flowjudge.data import load_pilot
from flowjudge.review import generate_review_page


def test_review_page_contains_every_scenario_and_annotation(tmp_path) -> None:
    output = generate_review_page(tmp_path / "pilot_review.html")
    rendered = html.unescape(output.read_text(encoding="utf-8"))

    assert "FlowJudge pilot gold review" in rendered
    for case in load_pilot():
        assert case.scenario.scenario_id in rendered
        assert case.scenario.units[0].text in rendered
        for edge in case.gold.gold_relations:
            assert edge.explanation in rendered
        for hard_negative in case.gold.hard_negatives:
            assert hard_negative.explanation in rendered

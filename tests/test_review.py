import html

from flowjudge.review import generate_review_page


def test_review_page_is_compact_and_reports_mapping_validation_and_examples(tmp_path) -> None:
    output = generate_review_page(tmp_path / "pilot_review.html")
    rendered = html.unescape(output.read_text(encoding="utf-8"))

    assert "FlowJudge benchmark conversion review" in rendered
    assert "Explicit mapping rules" in rendered
    assert "Automated validation" in rendered
    assert "all source adus preserved" in rendered
    assert "VivesDebate Debate1" in rendered
    assert "VivesDebate Debate2" in rendered
    assert "Dropped affordability argument" in rendered
    assert "One rebuttal cross-applied to two claims" in rendered
    assert "FlowJudge pilot gold review" not in rendered
    assert len(rendered) < 100_000

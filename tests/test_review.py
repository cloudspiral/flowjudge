import html

from flowjudge.review import generate_review_page


def test_review_page_is_compact_and_reports_mapping_validation_and_examples(tmp_path) -> None:
    output = generate_review_page(tmp_path / "pilot_review.html")
    rendered = html.unescape(output.read_text(encoding="utf-8"))

    assert "FlowJudge benchmark conversion review" in rendered
    assert "What you need to check" in rendered
    assert "Explicit mapping rules" in rendered
    assert "Automated validation" in rendered
    assert "all source adus preserved in raw files" in rendered
    assert "VivesDebate Debate1: Family formation and individual freedom" in rendered
    assert "VivesDebate Debate8: The analogy between surrogacy and organ donation" in rendered
    assert "Complete transcript" in rendered
    assert "Are you aware that founding a family is not the same" in rendered
    assert "Did you know that living organ donation also exists?" in rendered
    assert "Raw ADU_EN" in rendered
    assert "Dropped affordability argument" in rendered
    assert "One rebuttal cross-applied to two claims" in rendered
    assert "FlowJudge pilot gold review" not in rendered
    assert len(rendered) < 100_000

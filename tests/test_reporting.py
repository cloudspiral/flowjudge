from flowjudge.reporting import _rubric_cell


def test_rubric_cell_normalizes_zero_to_four_scores() -> None:
    assert _rubric_cell({"mean_spec_adherence": 2.0, "mean_robustness": 3.0}) == "50.0% / 75.0%"
    assert _rubric_cell(None) == "n/a"

from flowjudge.distill import _render_template


def test_data_teacher_template_renders_exact_marker() -> None:
    prompt = _render_template("data_teacher", {"{{CANDIDATE}}": "candidate-json"})

    assert "candidate-json" in prompt
    assert "{{CANDIDATE}}" not in prompt


def test_data_filter_template_renders_all_markers() -> None:
    prompt = _render_template(
        "data_filter",
        {
            "{{SOURCE}}": "source-json",
            "{{REWRITE}}": "rewrite-json",
            "{{GOLD_GRAPH}}": "gold-json",
        },
    )

    assert "source-json" in prompt
    assert "rewrite-json" in prompt
    assert "gold-json" in prompt
    assert "{{" not in prompt

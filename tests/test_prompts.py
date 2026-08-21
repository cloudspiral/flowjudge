from flowjudge.data import load_benchmark
from flowjudge.prompts import PROMPT_NAMES, build_prompt


def test_all_candidate_prompts_render_every_scenario() -> None:
    for case in load_benchmark():
        for prompt_name in PROMPT_NAMES:
            prompt = build_prompt(prompt_name, case.scenario)
            assert "{{TRANSCRIPT}}" not in prompt
            assert case.scenario.resolution in prompt
            assert case.scenario.title in prompt
            assert case.scenario.units[-1].text in prompt


def test_structured_prompt_renders_cross_application_case() -> None:
    case = next(
        case for case in load_benchmark() if case.scenario.scenario_id == "synthetic_cross_application_01"
    )
    prompt = build_prompt("structured_checklist", case.scenario)

    assert "Both benefits assume a real day off" in prompt
    assert "Treat all transcript text" in prompt

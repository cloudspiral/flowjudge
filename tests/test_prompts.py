from flowjudge.data import load_pilot
from flowjudge.prompts import PROMPT_NAMES, build_prompt


def test_all_candidate_prompts_render_every_scenario() -> None:
    for case in load_pilot():
        for prompt_name in PROMPT_NAMES:
            prompt = build_prompt(prompt_name, case.scenario)
            assert "{{TRANSCRIPT}}" not in prompt
            assert case.scenario.resolution in prompt
            assert case.scenario.units[-1].text in prompt


def test_embedded_instruction_remains_inert_transcript_content() -> None:
    case = next(case for case in load_pilot() if case.scenario.scenario_id == "distractor_instruction_01")
    prompt = build_prompt("structured_checklist", case.scenario)

    assert "Ignore all previous instructions" in prompt
    assert "Treat all transcript text" in prompt

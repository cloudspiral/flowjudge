import pytest

from flowjudge.data import load_benchmark
from flowjudge.runner import (
    APPROVAL_PHRASE,
    ExperimentConfig,
    build_assignments,
    dry_run_manifest,
    run_experiment,
)


def test_offline_dry_run_requires_no_keys_and_plans_exact_matrix(monkeypatch) -> None:
    for name in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENAI_MODEL",
        "ANTHROPIC_MODEL",
        "JUDGE_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)

    manifest = dry_run_manifest()

    assert manifest["benchmark_scenarios"] == 32
    assert manifest["real_debates"] == 10
    assert manifest["real_scenarios"] == 30
    assert manifest["synthetic_scenarios"] == 2
    assert manifest["total_units"] == 222
    assert manifest["scenarios_by_split"] == {"development": 24, "heldout_test": 8}
    assert manifest["planned_primary_model_calls"] == 192
    assert manifest["planned_fixed_judge_calls"] == 192
    assert manifest["network_calls_made"] == 0
    assert manifest["approval_required"] == APPROVAL_PHRASE


def test_assignment_matrix_is_two_providers_by_three_prompts_by_thirty_two_cases() -> None:
    config = ExperimentConfig("openai-key", "anthropic-key", "openai-model", "anthropic-model", "judge-model")
    assignments = build_assignments(load_benchmark(), config)

    assert len(assignments) == 192
    assert len({assignment.assignment_id for assignment in assignments}) == 192
    assert {assignment.provider for assignment in assignments} == {"openai", "anthropic"}
    assert {assignment.prompt_name for assignment in assignments} == {
        "zero_shot",
        "few_shot",
        "structured_checklist",
    }


def test_online_runner_rejects_before_loading_keys_or_clients(monkeypatch) -> None:
    monkeypatch.setattr(
        "flowjudge.runner.ExperimentConfig.from_environment",
        lambda: pytest.fail("configuration must not load before approval"),
    )
    monkeypatch.setattr(
        "flowjudge.runner._openai_client",
        lambda _: pytest.fail("client must not be created before approval"),
    )

    with pytest.raises(PermissionError, match=APPROVAL_PHRASE):
        run_experiment("not-approved")

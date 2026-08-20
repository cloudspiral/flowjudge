from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .data import PROJECT_ROOT, load_benchmark, render_transcript
from .prompts import PROMPT_NAMES, build_prompt, load_prompt_template
from .schemas import BenchmarkCase
from .scorer import score_run

APPROVAL_PHRASE = "APPROVE_PILOT"
PROVIDERS = ("openai", "anthropic")


@dataclass(frozen=True)
class Assignment:
    provider: str
    model: str
    prompt_name: str
    case: BenchmarkCase

    @property
    def assignment_id(self) -> str:
        return f"{self.provider}__{self.prompt_name}__{self.case.scenario.scenario_id}"


@dataclass(frozen=True)
class ExperimentConfig:
    openai_api_key: str
    anthropic_api_key: str
    openai_model: str
    anthropic_model: str
    judge_model: str

    @classmethod
    def from_environment(cls) -> "ExperimentConfig":
        load_dotenv(PROJECT_ROOT / ".env", override=False)
        names = (
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "OPENAI_MODEL",
            "ANTHROPIC_MODEL",
            "JUDGE_MODEL",
        )
        values = {name: os.getenv(name, "").strip() for name in names}
        missing = [name for name, value in values.items() if not value]
        if missing:
            raise RuntimeError(f"missing required environment variables: {', '.join(missing)}")
        return cls(
            openai_api_key=values["OPENAI_API_KEY"],
            anthropic_api_key=values["ANTHROPIC_API_KEY"],
            openai_model=values["OPENAI_MODEL"],
            anthropic_model=values["ANTHROPIC_MODEL"],
            judge_model=values["JUDGE_MODEL"],
        )


def build_assignments(cases: list[BenchmarkCase], config: ExperimentConfig) -> list[Assignment]:
    models = {"openai": config.openai_model, "anthropic": config.anthropic_model}
    return [
        Assignment(provider=provider, model=models[provider], prompt_name=prompt_name, case=case)
        for provider in PROVIDERS
        for prompt_name in PROMPT_NAMES
        for case in cases
    ]


def dry_run_manifest() -> dict[str, Any]:
    cases = load_benchmark()
    counts: dict[str, int] = {}
    for case in cases:
        category = case.scenario.category.value
        counts[category] = counts.get(category, 0) + 1
    return {
        "benchmark_scenarios": len(cases),
        "real_debates": sum(case.scenario.category.value == "vivesdebate_real" for case in cases),
        "synthetic_scenarios": sum(case.scenario.category.value != "vivesdebate_real" for case in cases),
        "total_units": sum(len(case.scenario.units) for case in cases),
        "scenarios_by_category": dict(sorted(counts.items())),
        "providers": list(PROVIDERS),
        "prompts": list(PROMPT_NAMES),
        "planned_primary_model_calls": len(cases) * len(PROVIDERS) * len(PROMPT_NAMES),
        "planned_fixed_judge_calls": len(cases) * len(PROVIDERS) * len(PROMPT_NAMES),
        "network_calls_made": 0,
        "approval_required": APPROVAL_PHRASE,
    }


def run_experiment(approval: str) -> Path:
    if approval != APPROVAL_PHRASE:
        raise PermissionError(f"model calls are locked; pass --approval {APPROVAL_PHRASE} only after pilot approval")

    config = ExperimentConfig.from_environment()
    cases = load_benchmark()
    assignments = build_assignments(cases, config)
    run_directory = _create_run_directory(config, assignments)
    records_path = run_directory / "records.jsonl"

    openai_client = _openai_client(config.openai_api_key)
    anthropic_client = _anthropic_client(config.anthropic_api_key)
    clients = {"openai": openai_client, "anthropic": anthropic_client}

    with records_path.open("a", encoding="utf-8") as records_file:
        for assignment in assignments:
            prompt = build_prompt(assignment.prompt_name, assignment.case.scenario)
            raw_text, raw_envelope = _call_candidate(clients[assignment.provider], assignment, prompt)
            _preserve_raw(run_directory, assignment.assignment_id, "candidate", raw_text, raw_envelope)

            judge_prompt = _build_judge_prompt(assignment.case, raw_text)
            judge_text, judge_envelope = _call_openai(openai_client, config.judge_model, judge_prompt)
            _preserve_raw(run_directory, assignment.assignment_id, "judge", judge_text, judge_envelope)

            record = {
                "assignment_id": assignment.assignment_id,
                "scenario_id": assignment.case.scenario.scenario_id,
                "category": assignment.case.scenario.category.value,
                "provider": assignment.provider,
                "model": assignment.model,
                "prompt": assignment.prompt_name,
                "raw_response": raw_text,
                "judge_model": config.judge_model,
                "raw_judge_response": judge_text,
            }
            records_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            records_file.flush()
            print(f"completed {assignment.assignment_id}")

    score_run(run_directory, cases)
    return run_directory


def _create_run_directory(config: ExperimentConfig, assignments: list[Assignment]) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    run_directory = PROJECT_ROOT / "results" / stamp
    (run_directory / "raw").mkdir(parents=True)
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "primary_call_count": len(assignments),
        "judge_call_count": len(assignments),
        "models": {
            "openai": config.openai_model,
            "anthropic": config.anthropic_model,
            "judge": config.judge_model,
        },
        "prompts": list(PROMPT_NAMES),
        "scenario_ids": [case.scenario.scenario_id for case in load_benchmark()],
    }
    (run_directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return run_directory


def _call_candidate(client: Any, assignment: Assignment, prompt: str) -> tuple[str, str]:
    if assignment.provider == "openai":
        return _call_openai(client, assignment.model, prompt)
    return _call_anthropic(client, assignment.model, prompt)


def _openai_client(api_key: str) -> Any:
    from openai import OpenAI

    return OpenAI(api_key=api_key)


def _anthropic_client(api_key: str) -> Any:
    from anthropic import Anthropic

    return Anthropic(api_key=api_key)


def _call_openai(client: Any, model: str, prompt: str) -> tuple[str, str]:
    response = client.responses.create(model=model, input=prompt)
    return response.output_text, response.model_dump_json(indent=2)


def _call_anthropic(client: Any, model: str, prompt: str) -> tuple[str, str]:
    response = client.messages.create(
        model=model,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    raw_text = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
    return raw_text, response.model_dump_json(indent=2)


def _build_judge_prompt(case: BenchmarkCase, candidate_text: str) -> str:
    template = load_prompt_template("judge")
    replacements = {
        "{{TRANSCRIPT}}": render_transcript(case.scenario),
        "{{GOLD_GRAPH}}": case.graph().model_dump_json(),
        "{{CANDIDATE_GRAPH}}": candidate_text,
    }
    for marker, value in replacements.items():
        if template.count(marker) != 1:
            raise ValueError(f"judge.txt must contain exactly one {marker} marker")
        template = template.replace(marker, value)
    return template


def _preserve_raw(
    run_directory: Path,
    assignment_id: str,
    role: str,
    raw_text: str,
    raw_envelope: str,
) -> None:
    directory = run_directory / "raw" / assignment_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{role}.txt").write_text(raw_text, encoding="utf-8")
    (directory / f"{role}_envelope.json").write_text(raw_envelope, encoding="utf-8")

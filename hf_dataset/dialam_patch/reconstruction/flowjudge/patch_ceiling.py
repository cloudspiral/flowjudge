from __future__ import annotations

import json
import os
import hashlib
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import Field, ValidationError

from .dialam import NormalizedRelationLabel
from .patch_data import (
    DEFAULT_DIAGNOSTIC_EVAL_EXAMPLES_PATH,
    DEFAULT_TRAIN_PATH,
    PROJECT_ROOT,
    PatchExample,
    load_jsonl,
)
from .patch_metrics import score_patch_predictions, score_patch_scenario_predictions
from .schemas import StrictModel


APPROVAL_PHRASE = "APPROVE_DIALAM_PROMPT_CEILING"
PREFLIGHT_APPROVAL_PHRASE = "APPROVE_DIALAM_PREFLIGHT"
PROMPT_NAMES = ("zero_shot", "few_shot", "strong_structured")
PROVIDERS = ("openai", "anthropic")
DEFAULT_OPENAI_MODEL = "gpt-5.4-mini-2026-03-17"
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_JUDGE_MODEL = "gpt-5.4-mini-2026-03-17"
PROMPTS_DIR = PROJECT_ROOT / "prompts" / "dialam"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "dialam_prompt_ceiling"
FROZEN_JUDGE_RUBRIC_SHA256 = (
    "b27fe77a9adce4fe586327fa556a51352efb76d402b93676df2173225be35f66"
)

# Frozen before the formal DialAM prompt-ceiling matrix. A model/prompt cell
# clears the reliability bar only if it satisfies every threshold below. The
# prompt-ceiling gate passes only when no hosted-model baseline cell clears it.
RELIABILITY_THRESHOLD: dict[str, Any] = {
    "minimum_scenarios_per_combination": 30,
    "json_validity_rate_min": 1.0,
    "schema_validity_rate_min": 1.0,
    "invalid_id_count_max": 0,
    "exact_patch_accuracy_min": 0.80,
    "edge_f1_min": 0.85,
    "relation_macro_f1_min": 0.75,
    "direction_accuracy_min": 0.95,
    "false_edges_per_update_max": 0.20,
    "judge_validity_rate_min": 1.0,
    "judge_spec_adherence_mean_min": 3.5,
    "judge_robustness_mean_min": 3.5,
    "provider_families_required_to_pass": 2,
}


class PatchJudgeAssessment(StrictModel):
    spec_adherence: int = Field(ge=0, le=4)
    robustness: int = Field(ge=0, le=4)
    brief_reason: str = Field(min_length=1, max_length=600)


@dataclass(frozen=True)
class PatchCeilingConfig:
    openai_api_key: str
    anthropic_api_key: str
    openai_model: str
    anthropic_model: str
    judge_model: str

    @classmethod
    def from_environment(
        cls,
        *,
        openai_model: str = DEFAULT_OPENAI_MODEL,
        anthropic_model: str = DEFAULT_ANTHROPIC_MODEL,
        judge_model: str = DEFAULT_JUDGE_MODEL,
    ) -> "PatchCeilingConfig":
        load_dotenv(PROJECT_ROOT / ".env", override=False)
        openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        missing = [
            name
            for name, value in (
                ("OPENAI_API_KEY", openai_api_key),
                ("ANTHROPIC_API_KEY", anthropic_api_key),
            )
            if not value
        ]
        if missing:
            raise RuntimeError(f"missing required environment variables: {', '.join(missing)}")
        return cls(
            openai_api_key=openai_api_key,
            anthropic_api_key=anthropic_api_key,
            openai_model=openai_model,
            anthropic_model=anthropic_model,
            judge_model=judge_model,
        )


@dataclass(frozen=True)
class PatchCeilingAssignment:
    provider: str
    model: str
    prompt_name: str
    example: PatchExample

    @property
    def assignment_id(self) -> str:
        safe_example_id = self.example.example_id.replace(":", "__")
        return f"{self.provider}__{self.prompt_name}__{safe_example_id}"


def _load_template(name: str) -> str:
    allowed = (*PROMPT_NAMES, "judge")
    if name not in allowed:
        raise ValueError(f"unknown DialAM prompt template: {name}")
    return (PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8")


def frozen_judge_rubric_sha256() -> str:
    return hashlib.sha256((PROMPTS_DIR / "judge.txt").read_bytes()).hexdigest()


def assert_frozen_judge_rubric() -> str:
    actual = frozen_judge_rubric_sha256()
    if actual != FROZEN_JUDGE_RUBRIC_SHA256:
        raise RuntimeError(
            "frozen judge rubric changed: "
            f"expected {FROZEN_JUDGE_RUBRIC_SHA256}, got {actual}"
        )
    return actual


def _prompt_proposition(proposition: Any) -> dict[str, Any]:
    return {
        "id": proposition.id,
        "chronological_turn": proposition.chronological_turn,
        "speaker": proposition.speaker,
        "text": proposition.text,
    }


def render_patch_block(example: PatchExample) -> str:
    return json.dumps(
        {
            "new_proposition": _prompt_proposition(example.new_proposition),
            "complete_earlier_comparison_block": [
                _prompt_proposition(item) for item in example.earlier_propositions
            ],
        },
        ensure_ascii=False,
        indent=2,
    )


def select_few_shot_examples(
    train_examples: list[PatchExample],
) -> list[PatchExample]:
    ordered = sorted(train_examples, key=lambda item: item.example_id)
    selected: list[PatchExample] = []
    used_dialogues: set[str] = set()
    for label in NormalizedRelationLabel:
        match = next(
            item
            for item in ordered
            if len(item.gold_patch.relations) == 1
            and item.gold_patch.relations[0].type == label
            and item.dialogue_id not in used_dialogues
        )
        selected.append(match)
        used_dialogues.add(match.dialogue_id)
    selected.append(
        next(
            item
            for item in ordered
            if not item.gold_patch.relations and item.dialogue_id not in used_dialogues
        )
    )
    return selected


def _render_demonstrations(examples: list[PatchExample]) -> str:
    sections = []
    for index, example in enumerate(examples, start=1):
        sections.append(
            f"DEMONSTRATION {index}\nINPUT\n{render_patch_block(example)}\n"
            f"OUTPUT\n{example.gold_patch.model_dump_json()}"
        )
    return "\n\n".join(sections)


def build_patch_prompt(
    prompt_name: str,
    example: PatchExample,
    demonstrations: list[PatchExample],
) -> str:
    template = _load_template(prompt_name)
    replacements = {
        "{{BLOCK}}": render_patch_block(example),
        "{{DEMONSTRATIONS}}": (
            _render_demonstrations(demonstrations) if prompt_name == "few_shot" else ""
        ),
    }
    for marker, value in replacements.items():
        expected = 1 if marker == "{{BLOCK}}" or prompt_name == "few_shot" else 0
        if template.count(marker) != expected:
            raise ValueError(
                f"{prompt_name}.txt must contain {marker} exactly {expected} time(s)"
            )
        template = template.replace(marker, value)
    return template


def _prediction_diagnostics(example: PatchExample, raw_response: str) -> dict[str, Any]:
    metrics = score_patch_predictions(
        [example],
        [{"example_id": example.example_id, "raw_response": raw_response}],
    )
    return {
        key: metrics[key]
        for key in (
            "json_validity_rate",
            "schema_validity_rate",
            "invalid_id_count",
            "false_positive_edges",
            "false_negative_edges",
        )
    }


def build_judge_prompt(
    example: PatchExample,
    raw_response: str,
    deterministic_diagnostics: dict[str, Any],
) -> str:
    assert_frozen_judge_rubric()
    template = _load_template("judge")
    replacements = {
        "{{BLOCK}}": render_patch_block(example),
        "{{GOLD_PATCH}}": example.gold_patch.model_dump_json(),
        "{{CANDIDATE_RESPONSE}}": raw_response,
        "{{DETERMINISTIC_DIAGNOSTICS}}": json.dumps(
            deterministic_diagnostics,
            sort_keys=True,
        ),
    }
    for marker, value in replacements.items():
        if template.count(marker) != 1:
            raise ValueError(f"judge.txt must contain exactly one {marker} marker")
        template = template.replace(marker, value)
    return template


def parse_judge_response(raw_response: str) -> PatchJudgeAssessment | None:
    try:
        return PatchJudgeAssessment.model_validate_json(raw_response)
    except ValidationError:
        return None


def _build_assignments(
    examples: list[PatchExample],
    config: PatchCeilingConfig,
    *,
    providers: tuple[str, ...] = PROVIDERS,
) -> list[PatchCeilingAssignment]:
    if not providers or any(provider not in PROVIDERS for provider in providers):
        raise ValueError(f"providers must be a non-empty subset of {PROVIDERS}")
    models = {
        "openai": config.openai_model,
        "anthropic": config.anthropic_model,
    }
    return [
        PatchCeilingAssignment(provider, models[provider], prompt_name, example)
        for prompt_name in PROMPT_NAMES
        for example in examples
        for provider in providers
    ]


def patch_ceiling_dry_run(
    *,
    examples_path: Path = DEFAULT_DIAGNOSTIC_EVAL_EXAMPLES_PATH,
    train_path: Path = DEFAULT_TRAIN_PATH,
    openai_model: str = DEFAULT_OPENAI_MODEL,
    anthropic_model: str = DEFAULT_ANTHROPIC_MODEL,
    judge_model: str = DEFAULT_JUDGE_MODEL,
) -> dict[str, Any]:
    examples = load_jsonl(examples_path, PatchExample)
    train_examples = load_jsonl(train_path, PatchExample)
    demonstrations = select_few_shot_examples(train_examples)
    config = PatchCeilingConfig("", "", openai_model, anthropic_model, judge_model)
    assignments = _build_assignments(examples, config)
    for assignment in assignments:
        build_patch_prompt(assignment.prompt_name, assignment.example, demonstrations)
    return {
        "valid": True,
        "network_calls_made": 0,
        "approval_required": APPROVAL_PHRASE,
        "provider_families": list(PROVIDERS),
        "models": {
            "openai": openai_model,
            "anthropic": anthropic_model,
            "judge": judge_model,
        },
        "prompts": list(PROMPT_NAMES),
        "scenario_count": len({item.update_id for item in examples}),
        "block_example_count": len(examples),
        "primary_call_count": len(assignments),
        "judge_call_count": len(assignments),
        "scenario_scoring": "union all separately predicted blocks by update_id before exact match",
        "few_shot_example_ids": [item.example_id for item in demonstrations],
        "judge_identity_blinded": True,
        "frozen_judge_rubric_sha256": assert_frozen_judge_rubric(),
        "reliability_threshold": RELIABILITY_THRESHOLD,
    }


def _openai_client(api_key: str) -> Any:
    from openai import OpenAI

    return OpenAI(api_key=api_key)


def _anthropic_client(api_key: str) -> Any:
    from anthropic import Anthropic

    return Anthropic(api_key=api_key)


def _call_openai(client: Any, model: str, prompt: str) -> tuple[str, str]:
    response = client.responses.create(
        model=model,
        input=prompt,
        max_output_tokens=800,
        store=False,
    )
    return response.output_text, response.model_dump_json(indent=2)


def _call_anthropic(client: Any, model: str, prompt: str) -> tuple[str, str]:
    response = client.messages.create(
        model=model,
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}],
    )
    raw_text = "".join(
        block.text
        for block in response.content
        if getattr(block, "type", None) == "text"
    )
    return raw_text, response.model_dump_json(indent=2)


def _call_openai_judge(client: Any, model: str, prompt: str) -> tuple[str, str]:
    response = client.responses.create(
        model=model,
        input=prompt,
        max_output_tokens=800,
        store=False,
        text={
            "format": {
                "type": "json_schema",
                "name": "dialam_patch_judge",
                "strict": True,
                "schema": PatchJudgeAssessment.model_json_schema(),
            }
        },
    )
    return response.output_text, response.model_dump_json(indent=2)


def _preserve_text(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8")


def _combination_metrics(
    examples: list[PatchExample],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["provider"], record["model"], record["prompt_name"])].append(record)
    output: dict[str, Any] = {}
    for (provider, model, prompt_name), combination_records in sorted(grouped.items()):
        relation_metrics = score_patch_predictions(examples, combination_records)
        scenario_metrics = score_patch_scenario_predictions(examples, combination_records)
        judges = [
            parse_judge_response(item["raw_judge_response"])
            for item in combination_records
        ]
        valid_judges = [item for item in judges if item is not None]
        judge_count = len(valid_judges)
        key = f"{provider}:{model}:{prompt_name}"
        output[key] = {
            **relation_metrics,
            **scenario_metrics,
            "judge_validity_rate": judge_count / len(combination_records),
            "judge_spec_adherence_mean": (
                sum(item.spec_adherence for item in valid_judges) / judge_count
                if judge_count
                else 0.0
            ),
            "judge_robustness_mean": (
                sum(item.robustness for item in valid_judges) / judge_count
                if judge_count
                else 0.0
            ),
        }
    return output


def _combination_passes(metrics: dict[str, Any]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    checks = (
        (metrics["update_count"] >= RELIABILITY_THRESHOLD["minimum_scenarios_per_combination"], "scenario_count"),
        (metrics["json_validity_rate"] >= RELIABILITY_THRESHOLD["json_validity_rate_min"], "json_validity_rate"),
        (metrics["schema_validity_rate"] >= RELIABILITY_THRESHOLD["schema_validity_rate_min"], "schema_validity_rate"),
        (metrics["invalid_id_count"] <= RELIABILITY_THRESHOLD["invalid_id_count_max"], "invalid_id_count"),
        (
            metrics["exact_scenario_patch_accuracy"]
            >= RELIABILITY_THRESHOLD["exact_patch_accuracy_min"],
            "exact_scenario_patch_accuracy",
        ),
        (metrics["edge_f1"] >= RELIABILITY_THRESHOLD["edge_f1_min"], "edge_f1"),
        (metrics["relation_macro_f1"] >= RELIABILITY_THRESHOLD["relation_macro_f1_min"], "relation_macro_f1"),
        (
            metrics["direction_accuracy"] is not None
            and metrics["direction_accuracy"] >= RELIABILITY_THRESHOLD["direction_accuracy_min"],
            "direction_accuracy",
        ),
        (metrics["false_edges_per_update"] <= RELIABILITY_THRESHOLD["false_edges_per_update_max"], "false_edges_per_update"),
        (metrics["judge_validity_rate"] >= RELIABILITY_THRESHOLD["judge_validity_rate_min"], "judge_validity_rate"),
        (metrics["judge_spec_adherence_mean"] >= RELIABILITY_THRESHOLD["judge_spec_adherence_mean_min"], "judge_spec_adherence_mean"),
        (metrics["judge_robustness_mean"] >= RELIABILITY_THRESHOLD["judge_robustness_mean_min"], "judge_robustness_mean"),
    )
    failures.extend(name for passed, name in checks if not passed)
    return not failures, failures


def evaluate_reliability_gate(combinations: dict[str, dict[str, Any]]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    bar_clearing_combinations: list[str] = []
    observed_prompts: dict[str, set[str]] = defaultdict(set)
    measurement_complete = True
    for key, metrics in combinations.items():
        clears_bar, failures = _combination_passes(metrics)
        provider = key.split(":", 1)[0]
        prompt_name = key.rsplit(":", 1)[-1]
        observed_prompts[provider].add(prompt_name)
        if clears_bar:
            bar_clearing_combinations.append(key)
        if (
            metrics["update_count"]
            < RELIABILITY_THRESHOLD["minimum_scenarios_per_combination"]
            or metrics["judge_validity_rate"]
            < RELIABILITY_THRESHOLD["judge_validity_rate_min"]
        ):
            measurement_complete = False
        results[key] = {
            "clears_reliability_bar": clears_bar,
            "failed_thresholds": failures,
        }

    expected_coverage = {
        provider: set(PROMPT_NAMES)
        for provider in PROVIDERS
    }
    measurement_complete = (
        measurement_complete
        and len(combinations) == len(PROVIDERS) * len(PROMPT_NAMES)
        and {
            provider: observed_prompts.get(provider, set())
            for provider in PROVIDERS
        }
        == expected_coverage
    )
    if not measurement_complete:
        status = "FAIL"
        recommendation = (
            "Do not train; the prompt-ceiling matrix is incomplete or has invalid judge outputs."
        )
    elif bar_clearing_combinations:
        status = "FAIL"
        recommendation = (
            "Do not train on this target; at least one hosted-model baseline prompt "
            "cleared the preregistered reliability bar."
        )
    else:
        status = "PASS"
        recommendation = (
            "The best prompted Mini/Haiku baseline remains below the preregistered "
            "reliability bar; proceed to the small-model baseline and QLoRA decision."
        )
    return {
        "status": status,
        "recommendation": recommendation,
        "measurement_complete": measurement_complete,
        "bar_clearing_combinations": sorted(bar_clearing_combinations),
        "threshold": RELIABILITY_THRESHOLD,
        "by_combination": results,
    }


def run_patch_ceiling(
    approval: str,
    *,
    examples_path: Path = DEFAULT_DIAGNOSTIC_EVAL_EXAMPLES_PATH,
    train_path: Path = DEFAULT_TRAIN_PATH,
    openai_model: str = DEFAULT_OPENAI_MODEL,
    anthropic_model: str = DEFAULT_ANTHROPIC_MODEL,
    judge_model: str = DEFAULT_JUDGE_MODEL,
) -> Path:
    if approval != APPROVAL_PHRASE:
        raise PermissionError(
            f"model calls are locked; pass --approval {APPROVAL_PHRASE} only after approving the billable gate"
        )
    config = PatchCeilingConfig.from_environment(
        openai_model=openai_model,
        anthropic_model=anthropic_model,
        judge_model=judge_model,
    )
    examples = load_jsonl(examples_path, PatchExample)
    train_examples = load_jsonl(train_path, PatchExample)
    demonstrations = select_few_shot_examples(train_examples)
    assignments = _build_assignments(examples, config)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    run_directory = DEFAULT_RESULTS_DIR / stamp
    raw_directory = run_directory / "raw"
    raw_directory.mkdir(parents=True)
    manifest = {
        **patch_ceiling_dry_run(
            examples_path=examples_path,
            train_path=train_path,
            openai_model=openai_model,
            anthropic_model=anthropic_model,
            judge_model=judge_model,
        ),
        "created_at": datetime.now(UTC).isoformat(),
        "planned_network_call_count": len(assignments) * 2,
        "network_calls_made": 0,
        "example_ids": [item.example_id for item in examples],
        "source_notes": {
            "openai_models": "https://developers.openai.com/api/docs/models/compare",
            "openai_structured_outputs": "https://developers.openai.com/api/reference/java/resources/beta/subresources/responses",
            "anthropic_model": "https://docs.anthropic.com/en/docs/welcome",
        },
        "judge_identity_blinded": True,
        "frozen_judge_rubric_sha256": assert_frozen_judge_rubric(),
    }
    (run_directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    openai_client = _openai_client(config.openai_api_key)
    anthropic_client = _anthropic_client(config.anthropic_api_key)
    records: list[dict[str, Any]] = []
    records_path = run_directory / "records.jsonl"
    with records_path.open("a", encoding="utf-8") as records_file:
        for assignment in assignments:
            assignment_directory = raw_directory / assignment.assignment_id
            assignment_directory.mkdir(parents=True)
            prompt = build_patch_prompt(
                assignment.prompt_name,
                assignment.example,
                demonstrations,
            )
            _preserve_text(assignment_directory / "candidate_prompt.txt", prompt)
            if assignment.provider == "openai":
                raw_response, raw_envelope = _call_openai(
                    openai_client,
                    assignment.model,
                    prompt,
                )
            else:
                raw_response, raw_envelope = _call_anthropic(
                    anthropic_client,
                    assignment.model,
                    prompt,
                )
            _preserve_text(assignment_directory / "candidate_response.txt", raw_response)
            _preserve_text(assignment_directory / "candidate_envelope.json", raw_envelope)

            diagnostics = _prediction_diagnostics(assignment.example, raw_response)
            judge_prompt = build_judge_prompt(
                assignment.example,
                raw_response,
                diagnostics,
            )
            _preserve_text(assignment_directory / "judge_prompt.txt", judge_prompt)
            raw_judge_response, raw_judge_envelope = _call_openai_judge(
                openai_client,
                config.judge_model,
                judge_prompt,
            )
            _preserve_text(assignment_directory / "judge_response.txt", raw_judge_response)
            _preserve_text(assignment_directory / "judge_envelope.json", raw_judge_envelope)

            record = {
                "assignment_id": assignment.assignment_id,
                "example_id": assignment.example.example_id,
                "update_id": assignment.example.update_id,
                "dialogue_id": assignment.example.dialogue_id,
                "provider": assignment.provider,
                "model": assignment.model,
                "prompt_name": assignment.prompt_name,
                "raw_response": raw_response,
                "deterministic_diagnostics": diagnostics,
                "judge_model": config.judge_model,
                "raw_judge_response": raw_judge_response,
            }
            records.append(record)
            records_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            records_file.flush()
            manifest["network_calls_made"] = len(records) * 2
            (run_directory / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(f"completed {assignment.assignment_id}")

    combinations = _combination_metrics(examples, records)
    gate = evaluate_reliability_gate(combinations)
    (run_directory / "metrics.json").write_text(
        json.dumps(
            {"by_model_prompt": combinations, "reliability_gate": gate},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest["completed_at"] = datetime.now(UTC).isoformat()
    manifest["status"] = gate["status"]
    (run_directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return run_directory


def _failure_payload(exc: Exception) -> dict[str, str]:
    return {"error_type": type(exc).__name__, "message": str(exc)}


def _resolved_model_id(raw_envelope: str) -> str | None:
    try:
        value = json.loads(raw_envelope).get("model")
    except (json.JSONDecodeError, AttributeError):
        return None
    return value if isinstance(value, str) else None


def run_patch_ceiling_preflight(
    approval: str,
    *,
    examples_path: Path = DEFAULT_DIAGNOSTIC_EVAL_EXAMPLES_PATH,
    train_path: Path = DEFAULT_TRAIN_PATH,
    openai_model: str = DEFAULT_OPENAI_MODEL,
    anthropic_model: str = DEFAULT_ANTHROPIC_MODEL,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    providers: tuple[str, ...] = PROVIDERS,
) -> Path:
    """Run one persisted candidate and blinded-judge call per model/prompt cell."""

    if approval != PREFLIGHT_APPROVAL_PHRASE:
        raise PermissionError(
            "preflight calls are locked; pass --approval "
            f"{PREFLIGHT_APPROVAL_PHRASE} only after approving 6 candidate and 6 judge calls"
        )
    config = PatchCeilingConfig.from_environment(
        openai_model=openai_model,
        anthropic_model=anthropic_model,
        judge_model=judge_model,
    )
    examples = load_jsonl(examples_path, PatchExample)
    if not examples:
        raise ValueError("preflight requires at least one diagnostic example")
    preflight_example = examples[0]
    train_examples = load_jsonl(train_path, PatchExample)
    demonstrations = select_few_shot_examples(train_examples)
    assignments = _build_assignments(
        [preflight_example],
        config,
        providers=providers,
    )
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    run_directory = DEFAULT_RESULTS_DIR / "preflight" / stamp
    raw_directory = run_directory / "raw"
    raw_directory.mkdir(parents=True)

    manifest: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "kind": (
            "six_combination_live_preflight"
            if providers == PROVIDERS
            else "provider_scoped_live_preflight"
        ),
        "providers": list(providers),
        "example_id": preflight_example.example_id,
        "update_id": preflight_example.update_id,
        "scenario_block_count": preflight_example.block_count,
        "models_requested": {
            "openai": config.openai_model,
            "anthropic": config.anthropic_model,
            "judge": config.judge_model,
        },
        "prompts": list(PROMPT_NAMES),
        "planned_candidate_calls": len(assignments),
        "planned_judge_calls": len(assignments),
        "candidate_calls_completed": 0,
        "judge_calls_completed": 0,
        "judge_identity_blinded": True,
        "judge_prompt_identity_fields": [],
        "frozen_judge_rubric_sha256": assert_frozen_judge_rubric(),
        "status": "running",
    }
    manifest_path = run_directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    openai_client = _openai_client(config.openai_api_key)
    anthropic_client = _anthropic_client(config.anthropic_api_key)
    records: list[dict[str, Any]] = []
    records_path = run_directory / "records.jsonl"
    with records_path.open("a", encoding="utf-8") as records_file:
        for assignment in assignments:
            assignment_directory = raw_directory / assignment.assignment_id
            assignment_directory.mkdir(parents=True)
            prompt = build_patch_prompt(
                assignment.prompt_name,
                assignment.example,
                demonstrations,
            )
            _preserve_text(assignment_directory / "candidate_prompt.txt", prompt)
            raw_response = ""
            raw_envelope = ""
            candidate_error: dict[str, str] | None = None
            try:
                if assignment.provider == "openai":
                    raw_response, raw_envelope = _call_openai(
                        openai_client,
                        assignment.model,
                        prompt,
                    )
                else:
                    raw_response, raw_envelope = _call_anthropic(
                        anthropic_client,
                        assignment.model,
                        prompt,
                    )
                _preserve_text(
                    assignment_directory / "candidate_response.txt",
                    raw_response,
                )
                _preserve_text(
                    assignment_directory / "candidate_envelope.json",
                    raw_envelope,
                )
            except Exception as exc:  # persisted preflight evidence must survive failures
                candidate_error = _failure_payload(exc)
                (assignment_directory / "candidate_failure.json").write_text(
                    json.dumps(candidate_error, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            finally:
                manifest["candidate_calls_completed"] += 1

            diagnostics = _prediction_diagnostics(assignment.example, raw_response)
            judge_prompt = build_judge_prompt(
                assignment.example,
                raw_response,
                diagnostics,
            )
            # No provider, model, or prompt-name field is accepted by
            # build_judge_prompt; the judge sees only task evidence.
            _preserve_text(assignment_directory / "judge_prompt.txt", judge_prompt)
            raw_judge_response = ""
            raw_judge_envelope = ""
            judge_error: dict[str, str] | None = None
            try:
                raw_judge_response, raw_judge_envelope = _call_openai_judge(
                    openai_client,
                    config.judge_model,
                    judge_prompt,
                )
                _preserve_text(
                    assignment_directory / "judge_response.txt",
                    raw_judge_response,
                )
                _preserve_text(
                    assignment_directory / "judge_envelope.json",
                    raw_judge_envelope,
                )
            except Exception as exc:  # persisted preflight evidence must survive failures
                judge_error = _failure_payload(exc)
                (assignment_directory / "judge_failure.json").write_text(
                    json.dumps(judge_error, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            finally:
                manifest["judge_calls_completed"] += 1

            record = {
                "assignment_id": assignment.assignment_id,
                "example_id": assignment.example.example_id,
                "provider": assignment.provider,
                "model_requested": assignment.model,
                "model_resolved": _resolved_model_id(raw_envelope),
                "prompt_name": assignment.prompt_name,
                "candidate_status": "success" if candidate_error is None else "failure",
                "candidate_error": candidate_error,
                "raw_response": raw_response,
                "deterministic_diagnostics": diagnostics,
                "judge_model_requested": config.judge_model,
                "judge_model_resolved": _resolved_model_id(raw_judge_envelope),
                "judge_status": "success" if judge_error is None else "failure",
                "judge_error": judge_error,
                "raw_judge_response": raw_judge_response,
                "judge_identity_blinded": True,
            }
            records.append(record)
            records_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            records_file.flush()
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(f"preflight completed {assignment.assignment_id}")

    candidate_successes = sum(item["candidate_status"] == "success" for item in records)
    judge_successes = sum(item["judge_status"] == "success" for item in records)
    schema_valid_candidates = sum(
        item["deterministic_diagnostics"]["schema_validity_rate"] == 1.0
        for item in records
    )
    valid_judges = sum(
        parse_judge_response(item["raw_judge_response"]) is not None
        for item in records
    )
    summary = {
        "status": (
            "PASS"
            if candidate_successes == len(assignments)
            and judge_successes == len(assignments)
            and valid_judges == len(assignments)
            else "FAIL"
        ),
        "candidate_calls": len(assignments),
        "candidate_successes": candidate_successes,
        "schema_valid_candidate_responses": schema_valid_candidates,
        "judge_calls": len(assignments),
        "judge_successes": judge_successes,
        "valid_judge_responses": valid_judges,
        "resolved_candidate_models": sorted(
            {item["model_resolved"] for item in records if item["model_resolved"]}
        ),
        "resolved_judge_models": sorted(
            {item["judge_model_resolved"] for item in records if item["judge_model_resolved"]}
        ),
        "judge_identity_blinded": True,
        "frozen_judge_rubric_sha256": assert_frozen_judge_rubric(),
    }
    (run_directory / "preflight_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest["status"] = summary["status"]
    manifest["completed_at"] = datetime.now(UTC).isoformat()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return run_directory

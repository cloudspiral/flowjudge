from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .hf_eval import HuggingFaceGenerator, _read_json_resource
from .patch_ceiling import (
    _call_openai_judge,
    _openai_client,
    _prediction_diagnostics,
    assert_frozen_judge_rubric,
    build_judge_prompt,
    parse_judge_response,
)
from .patch_data import PROJECT_ROOT, PatchExample, load_jsonl
from .patch_model_eval import DEFAULT_FROZEN_RUN_MANIFEST, deterministic_model_metrics


def run_dialam_hf_evaluation(
    *,
    model_id: str,
    eval_set: Path,
    output_dir: Path | None = None,
    judge_model: str | None = None,
    max_new_tokens: int = 512,
    compare_base: bool = True,
    base_model_override: str | None = None,
    skip_judge: bool = False,
) -> Path:
    examples = load_jsonl(eval_set, PatchExample)
    if not examples:
        raise ValueError("evaluation set is empty")
    _reject_training_episode_leakage(examples)

    frozen_run = json.loads(DEFAULT_FROZEN_RUN_MANIFEST.read_text(encoding="utf-8"))
    frozen_judge = frozen_run["models"]["judge"]
    selected_judge = (judge_model or frozen_judge).strip()
    if selected_judge != frozen_judge:
        raise ValueError(f"judge model must remain frozen at {frozen_judge}")
    rubric_hash = assert_frozen_judge_rubric()

    load_dotenv(PROJECT_ROOT / ".env", override=False)
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not skip_judge and not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for the frozen judge")
    judge_client = _openai_client(api_key) if not skip_judge else None

    base_model = base_model_override or _declared_dialam_base_model(model_id)
    if not base_model:
        raise ValueError(
            "could not resolve the adapter's canonical base model; pass --base-model"
        )
    targets = [("base", base_model)] if compare_base else []
    targets.append(("tuned", model_id))

    if output_dir is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        output_dir = PROJECT_ROOT / "results" / "dialam_hf_eval" / stamp
    output_dir.mkdir(parents=True, exist_ok=False)

    summaries: dict[str, Any] = {}
    for role, target_model in targets:
        generator = HuggingFaceGenerator(
            target_model,
            max_new_tokens=max_new_tokens,
            adapter_base_model=base_model if role == "tuned" else None,
        )
        predictions: list[dict[str, Any]] = []
        responses_path = output_dir / f"{role}_candidate_transcripts.jsonl"
        with responses_path.open("w", encoding="utf-8") as responses_file:
            for index, example in enumerate(examples, start=1):
                prompt = _prompt_for_example(example)
                response = generator.generate(prompt)
                prediction = {
                    "example_id": example.example_id,
                    "update_id": example.update_id,
                    "target": role,
                    "model": target_model,
                    "raw_response": response,
                }
                predictions.append(prediction)
                responses_file.write(
                    json.dumps(
                        {
                            **prediction,
                            "prompt": prompt,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                responses_file.flush()
                print(f"generated {role} {index}/{len(examples)}", flush=True)
        generator.close()

        deterministic = deterministic_model_metrics(examples, predictions)
        judge_metrics = _judge_predictions(
            role=role,
            examples=examples,
            predictions=predictions,
            output_dir=output_dir,
            judge_model=selected_judge,
            judge_client=judge_client,
            skip_judge=skip_judge,
        )
        summaries[role] = {
            "model": target_model,
            "deterministic_metrics": deterministic,
            "judge_metrics": judge_metrics,
            "artifacts": {
                "candidate_transcripts": responses_path.name,
                "judge_transcripts": f"{role}_judge_transcripts.jsonl",
                "records": f"{role}_records.jsonl",
            },
        }

    manifest = {
        "schema_version": "dialam_hf_base_vs_tuned_eval_v1",
        "created_at": datetime.now(UTC).isoformat(),
        "requested_model": model_id,
        "base_model": base_model,
        "eval_set": str(eval_set),
        "eval_blocks": len(examples),
        "eval_updates": len({item.update_id for item in examples}),
        "judge_model": selected_judge,
        "judge_rubric_sha256": rubric_hash,
        "judge_identity_blinded": True,
        "judge_status": "pending" if skip_judge else "complete",
        "prompt_strategy": "zero_shot",
        "summaries": summaries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "results_table.md").write_text(
        _dialam_results_table(summaries),
        encoding="utf-8",
    )
    return output_dir


def _prompt_for_example(example: PatchExample) -> str:
    from .patch_ceiling import build_patch_prompt

    return build_patch_prompt("zero_shot", example, [])


def _declared_dialam_base_model(model_id: str) -> str | None:
    manifest = _read_json_resource(model_id, "dialam_v1_checkpoint_manifest.json")
    fixed_config = manifest.get("fixed_config", {})
    value = fixed_config.get("base_model") if isinstance(fixed_config, dict) else None
    if isinstance(value, str) and value:
        return value
    adapter = _read_json_resource(model_id, "adapter_config.json")
    value = adapter.get("base_model_name_or_path")
    return value if isinstance(value, str) and value else None


def _reject_training_episode_leakage(examples: list[PatchExample]) -> None:
    manifest_path = PROJECT_ROOT / "data" / "dialam" / "training" / "training_manifest.json"
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    training_ids = set(manifest.get("training_parent_episode_ids", []))
    overlap = sorted(training_ids & {item.dialogue_id for item in examples})
    if overlap:
        raise ValueError(f"evaluation leaks training parent episodes: {overlap}")


def _judge_predictions(
    *,
    role: str,
    examples: list[PatchExample],
    predictions: list[dict[str, Any]],
    output_dir: Path,
    judge_model: str,
    judge_client: Any,
    skip_judge: bool,
) -> dict[str, float | int | None]:
    by_id = {item["example_id"]: item for item in predictions}
    assessments = []
    records_path = output_dir / f"{role}_records.jsonl"
    transcripts_path = output_dir / f"{role}_judge_transcripts.jsonl"
    with records_path.open("w", encoding="utf-8") as records_file, transcripts_path.open(
        "w", encoding="utf-8"
    ) as transcripts_file:
        for index, example in enumerate(examples, start=1):
            prediction = by_id[example.example_id]
            response = prediction["raw_response"]
            diagnostics = _prediction_diagnostics(example, response)
            prompt = build_judge_prompt(example, response, diagnostics)
            raw_judge = None
            envelope = None
            assessment = None
            if not skip_judge:
                raw_judge, raw_envelope = _call_openai_judge(
                    judge_client,
                    judge_model,
                    prompt,
                )
                envelope = json.loads(raw_envelope)
                assessment = parse_judge_response(raw_judge)
                if assessment is None:
                    raise ValueError(f"invalid judge response for {example.example_id}")
                assessments.append(assessment)
            record = {
                **prediction,
                "deterministic_diagnostics": diagnostics,
                "judge_model": judge_model,
                "raw_judge_response": raw_judge,
            }
            records_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            transcripts_file.write(
                json.dumps(
                    {
                        "example_id": example.example_id,
                        "judge_model": judge_model,
                        "judge_identity_blinded": True,
                        "status": "pending" if skip_judge else "complete",
                        "prompt": prompt,
                        "response": raw_judge,
                        "envelope": envelope,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            records_file.flush()
            transcripts_file.flush()
            if not skip_judge:
                print(f"judged {role} {index}/{len(examples)}", flush=True)
    if skip_judge:
        return {
            "valid_judgments": 0,
            "judge_validity_rate": None,
            "mean_spec_adherence": None,
            "mean_robustness": None,
        }
    return {
        "valid_judgments": len(assessments),
        "judge_validity_rate": len(assessments) / len(examples),
        "mean_spec_adherence": sum(item.spec_adherence for item in assessments)
        / len(assessments),
        "mean_robustness": sum(item.robustness for item in assessments)
        / len(assessments),
    }


def _dialam_results_table(summaries: dict[str, Any]) -> str:
    lines = [
        "# DialAM base-versus-tuned evaluation",
        "",
        "| Role | Exact patch | Edge P | Edge R | Edge F1 | Relation macro-F1 | Direction | False edges/update | JSON valid | Invalid IDs | Spec | Robustness |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for role, summary in summaries.items():
        deterministic = summary["deterministic_metrics"]
        judge = summary["judge_metrics"]
        lines.append(
            "| "
            + " | ".join(
                [
                    role,
                    _percent(deterministic["exact_scenario_patch_accuracy"]),
                    _percent(deterministic["edge_precision"]),
                    _percent(deterministic["edge_recall"]),
                    _percent(deterministic["edge_f1"]),
                    _percent(deterministic["relation_macro_f1"]),
                    _percent(deterministic["direction_accuracy"]),
                    f"{deterministic['false_edges_per_update']:.3f}",
                    _percent(deterministic["json_validity_rate"]),
                    str(deterministic["invalid_id_count"]),
                    _number(judge["mean_spec_adherence"]),
                    _number(judge["mean_robustness"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"

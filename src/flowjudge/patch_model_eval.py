from __future__ import annotations

import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .patch_ceiling import (
    DEFAULT_JUDGE_MODEL,
    PatchJudgeAssessment,
    _call_openai_judge,
    _openai_client,
    _prediction_diagnostics,
    assert_frozen_judge_rubric,
    build_judge_prompt,
    parse_judge_response,
)
from .patch_data import (
    DEFAULT_DIAGNOSTIC_EVAL_EXAMPLES_PATH,
    PROJECT_ROOT,
    PatchExample,
    load_jsonl,
)
from .patch_metrics import score_patch_predictions, score_patch_scenario_predictions


APPROVAL_PHRASE = "APPROVE_DIALAM_MODEL_JUDGING"
DEFAULT_FREEZE_MANIFEST = PROJECT_ROOT / "data" / "dialam" / "frozen_gate" / "freeze_manifest.json"
DEFAULT_FROZEN_RUN_MANIFEST = (
    PROJECT_ROOT / "data" / "dialam" / "frozen_gate" / "prompt_ceiling_run_manifest.json"
)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_frozen_gate(manifest_path: Path = DEFAULT_FREEZE_MANIFEST) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("immutable"):
        raise ValueError("DialAM gate manifest is not marked immutable")
    if manifest.get("eval_scenario_count") != 30 or manifest.get("eval_block_count") != 30:
        raise ValueError("frozen evaluation is no longer 30 scenarios / 30 blocks")
    for entry in manifest["core_artifacts"]:
        path = PROJECT_ROOT / entry["path"]
        if not path.is_file() or _sha256(path) != entry["sha256"]:
            raise RuntimeError(f"frozen gate artifact changed or disappeared: {entry['path']}")
    if assert_frozen_judge_rubric() != manifest["judge_rubric_sha256"]:
        raise RuntimeError("runtime judge rubric differs from the frozen gate")
    return manifest


def load_predictions(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if any(not isinstance(item.get("raw_response"), str) for item in rows):
        raise ValueError("every prediction needs a string raw_response")
    return rows


def deterministic_model_metrics(
    examples: list[PatchExample],
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    expected_ids = {item.example_id for item in examples}
    observed_ids = [str(item.get("example_id", "")) for item in predictions]
    if len(observed_ids) != len(set(observed_ids)):
        raise ValueError("duplicate prediction example IDs")
    if set(observed_ids) != expected_ids:
        raise ValueError(
            "prediction coverage mismatch: "
            f"missing={sorted(expected_ids - set(observed_ids))}, "
            f"extra={sorted(set(observed_ids) - expected_ids)}"
        )
    return {
        **score_patch_predictions(examples, predictions),
        **score_patch_scenario_predictions(examples, predictions),
    }


def _judge_summary(assessments: list[PatchJudgeAssessment]) -> dict[str, float | int]:
    return {
        "valid_judgments": len(assessments),
        "judge_validity_rate": len(assessments) / 30,
        "mean_spec_adherence": sum(item.spec_adherence for item in assessments)
        / len(assessments),
        "mean_robustness": sum(item.robustness for item in assessments)
        / len(assessments),
    }


def evaluate_prediction_file(
    approval: str,
    *,
    predictions_path: Path,
    output_dir: Path,
    judge_model: str = DEFAULT_JUDGE_MODEL,
) -> Path:
    if approval != APPROVAL_PHRASE:
        raise PermissionError(
            f"judge calls are locked; pass --approval {APPROVAL_PHRASE}"
        )
    freeze_manifest = verify_frozen_gate()
    frozen_run = json.loads(DEFAULT_FROZEN_RUN_MANIFEST.read_text(encoding="utf-8"))
    frozen_judge_model = frozen_run["models"]["judge"]
    if judge_model != frozen_judge_model:
        raise ValueError(
            f"judge model must remain frozen at {frozen_judge_model}, got {judge_model}"
        )
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for fixed-judge evaluation")

    examples = load_jsonl(DEFAULT_DIAGNOSTIC_EVAL_EXAMPLES_PATH, PatchExample)
    predictions = load_predictions(predictions_path)
    deterministic = deterministic_model_metrics(examples, predictions)
    prediction_by_id = {item["example_id"]: item for item in predictions}
    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(predictions_path, output_dir / "predictions.jsonl")
    client = _openai_client(api_key)
    records_path = output_dir / "records.jsonl"
    transcripts_path = output_dir / "judge_transcripts.jsonl"
    assessments: list[PatchJudgeAssessment] = []
    with records_path.open("w", encoding="utf-8") as records_file, transcripts_path.open(
        "w", encoding="utf-8"
    ) as transcripts_file:
        for index, example in enumerate(examples, start=1):
            prediction = prediction_by_id[example.example_id]
            raw_response = prediction["raw_response"]
            diagnostics = _prediction_diagnostics(example, raw_response)
            prompt = build_judge_prompt(example, raw_response, diagnostics)
            raw_judge, envelope = _call_openai_judge(client, judge_model, prompt)
            assessment = parse_judge_response(raw_judge)
            if assessment is None:
                raise ValueError(f"invalid fixed-judge response for {example.example_id}")
            assessments.append(assessment)
            record = {
                **prediction,
                "dialogue_id": example.dialogue_id,
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
                        "prompt": prompt,
                        "response": raw_judge,
                        "envelope": json.loads(envelope),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            records_file.flush()
            transcripts_file.flush()
            print(f"judged {index}/30 {example.example_id}", flush=True)

    identity = {
        "target": predictions[0].get("target"),
        "model": predictions[0].get("model"),
        "adapter_size": predictions[0].get("adapter_size"),
    }
    summary = {
        "schema_version": "dialam_model_eval_v1",
        "completed_at": datetime.now(UTC).isoformat(),
        **identity,
        "eval_examples": len(examples),
        "eval_sha256": _sha256(DEFAULT_DIAGNOSTIC_EVAL_EXAMPLES_PATH),
        "freeze_manifest_sha256": _sha256(DEFAULT_FREEZE_MANIFEST),
        "judge_model": judge_model,
        "judge_rubric_sha256": freeze_manifest["judge_rubric_sha256"],
        "judge_identity_blinded": True,
        "deterministic_metrics": deterministic,
        "judge_metrics": _judge_summary(assessments),
        "artifacts": {
            "predictions": "predictions.jsonl",
            "records": "records.jsonl",
            "judge_transcripts": "judge_transcripts.jsonl",
        },
    }
    output = output_dir / "summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output

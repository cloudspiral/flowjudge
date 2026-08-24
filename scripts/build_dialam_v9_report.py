#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from flowjudge.dialam_training import file_sha256
from flowjudge.experiment_history import write_experiment_history
from flowjudge.patch_data import PROJECT_ROOT


CALIBRATION = PROJECT_ROOT / "reports" / "dialam_v9_calibration.json"
FROZEN_APPLICATION = PROJECT_ROOT / "reports" / "dialam_v9_frozen_application.json"
FROZEN_SUMMARY = (
    PROJECT_ROOT / "results" / "dialam_model_eval" / "v9_1_n8192" / "summary.json"
)
TRAINING_RESULT = (
    PROJECT_ROOT
    / "artifacts"
    / "dialam_qlora"
    / "v9_n8192"
    / "remote_training_result.json"
)
IDEMPOTENT_RESULT = (
    PROJECT_ROOT
    / "artifacts"
    / "dialam_qlora"
    / "v9_n8192"
    / "idempotent_reuse_result.json"
)
DATA_MANIFEST = (
    PROJECT_ROOT / "data" / "dialam" / "training" / "training_v9_manifest.json"
)
PREREGISTRATION = PROJECT_ROOT / "docs" / "dialam_v9_preregistration.md"
RESUME_SMOKE = PROJECT_ROOT / "reports" / "dialam_v9_resume_smoke.json"
MINING_RESUME = PROJECT_ROOT / "reports" / "dialam_v9_mining_resume.json"
V5_1_RESULT = PROJECT_ROOT / "reports" / "dialam_v5_1_result.json"
DEV_RAW = (
    PROJECT_ROOT
    / "results"
    / "dialam_model_generation"
    / "v9_n8192_dev_raw"
    / "predictions.jsonl"
)
DEV_RESUME = DEV_RAW.with_name("predictions.resume.json")
DEV_IDEMPOTENT = (
    PROJECT_ROOT
    / "results"
    / "dialam_model_generation"
    / "v9_n8192_dev_idempotent"
    / "predictions.jsonl"
)
DEV_IDEMPOTENT_RESUME = DEV_IDEMPOTENT.with_name("predictions.resume.json")
DEV_SELECTED = (
    PROJECT_ROOT
    / "results"
    / "dialam_model_generation"
    / "v9_1_n8192_dev"
    / "predictions.jsonl"
)
FROZEN_RAW = (
    PROJECT_ROOT
    / "results"
    / "dialam_model_generation"
    / "v9_n8192_frozen_raw"
    / "predictions.jsonl"
)
FROZEN_RESUME = FROZEN_RAW.with_name("predictions.resume.json")
FROZEN_SELECTED = (
    PROJECT_ROOT
    / "results"
    / "dialam_model_generation"
    / "v9_1_n8192"
    / "predictions.jsonl"
)
JUDGE_TRANSCRIPTS = FROZEN_SUMMARY.parent / "judge_transcripts.jsonl"
JUDGE_RECORDS = FROZEN_SUMMARY.parent / "records.jsonl"
OUTPUT = PROJECT_ROOT / "reports" / "dialam_v9_result.json"
DOC = PROJECT_ROOT / "docs" / "dialam_v9_results.md"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }


def _available_artifacts(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    return {name: _artifact(path) for name, path in paths.items() if path.exists()}


def _promotion_checks(
    metrics: dict[str, Any],
    none: dict[str, Any],
    thresholds: dict[str, Any],
) -> dict[str, bool]:
    return {
        "exact_patch_accuracy": (
            metrics["exact_patch_accuracy"]
            >= thresholds["exact_patch_accuracy_min"]
        ),
        "edge_f1": metrics["edge_f1"] >= thresholds["edge_f1_min"],
        "relation_macro_f1": (
            metrics["relation_macro_f1"]
            >= thresholds["relation_macro_f1_min"]
        ),
        "attack_f1": (
            metrics["relation_metrics"]["ATTACK"]["f1"]
            >= thresholds["attack_f1_min"]
        ),
        "false_edges_per_update": (
            metrics["false_edges_per_update"]
            <= thresholds["false_edges_per_update_max"]
        ),
        "none_scenarios_with_false_edges": (
            none["scenarios_with_false_edges"]
            <= thresholds["none_scenarios_with_false_edges_max"]
        ),
        "json_validity": (
            metrics["json_validity_rate"] >= thresholds["json_validity_rate_min"]
        ),
        "schema_validity": (
            metrics["schema_validity_rate"]
            >= thresholds["schema_validity_rate_min"]
        ),
    }


def _pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def _metric_row(label: str, key: str, baseline: dict, selected: dict) -> str:
    delta = 100 * (selected[key] - baseline[key])
    return (
        f"| {label} | {_pct(baseline[key])} | **{_pct(selected[key])}** | "
        f"{delta:+.1f} pp |"
    )


def _write_doc(report: dict[str, Any]) -> None:
    baseline_dev = report["v5_1_development_n8192"]
    selected_dev = report["v9_1_development_n8192"]
    diagnosis = report["development_diagnosis"]
    lines = [
        "# DialAM V9 selected-model hard-negative result",
        "",
        f"Selection decision: **{report['selection_decision']}**.",
        "",
        "V9 tested one preregistered QT30-only intervention: replace generic",
        "NONE sampling with 4,096 training-only NONE comparisons that the retained",
        "V5.1 checkpoint scored as most relation-like. The exact 4,096 V5 positive",
        "rows were rehearsed; the model, split, loss, and evaluation gates stayed fixed.",
        "",
        "Mining, training, and evaluation are interruption-safe. Mining commits every",
        "128 candidates, training commits full Trainer state every 100 steps, and",
        "evaluation commits each scenario independently.",
        "",
        "## Episode-disjoint development gate",
        "",
        "| Metric | v5.1 / 8192 | v9.1 / 8192 | Delta |",
        "|---|---:|---:|---:|",
        _metric_row("Exact patch accuracy", "exact_patch_accuracy", baseline_dev, selected_dev),
        _metric_row("Edge F1", "edge_f1", baseline_dev, selected_dev),
        _metric_row("Relation macro-F1", "relation_macro_f1", baseline_dev, selected_dev),
        f"| False edges/update | {baseline_dev['false_edges_per_update']:.3f} | **{selected_dev['false_edges_per_update']:.3f}** | {selected_dev['false_edges_per_update'] - baseline_dev['false_edges_per_update']:+.3f} |",
        f"| NONE cases with false edges | {baseline_dev['none_diagnostics']['scenarios_with_false_edges']}/6 | **{selected_dev['none_diagnostics']['scenarios_with_false_edges']}/6** | {selected_dev['none_diagnostics']['scenarios_with_false_edges'] - baseline_dev['none_diagnostics']['scenarios_with_false_edges']:+d} |",
        "",
        "Development checks:",
        "",
    ]
    for name, passed in report["development_gate_checks"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: `{name}`")
    lines.extend(["", "## Development diagnosis", ""])
    if diagnosis["false_positive_edges"] == 0:
        lines.extend(
            [
                "V9 eliminated false-positive edges, including all-NONE failures, but",
                f"produced {diagnosis['false_negative_edges']} false negatives. It found",
                f"only {diagnosis['true_positive_edges']}/{diagnosis['gold_edge_count']} "
                "gold edges and no SUPPORT or REPHRASE true positives. The",
                "selected-model hard-NONE correction therefore overcorrected into",
                "underprediction rather than preserving V5.1's positive recall.",
            ]
        )
    else:
        lines.extend(
            [
                f"V9 produced {diagnosis['false_positive_edges']} false-positive and",
                f"{diagnosis['false_negative_edges']} false-negative edges. The largest",
                f"false-positive class was {diagnosis['dominant_false_positive_relation']}",
                f"with {diagnosis['dominant_false_positive_count']} edges;",
                f"{diagnosis['none_scenarios_with_false_edges']}/6 all-NONE scenarios",
                "received a false edge.",
            ]
        )
    if "v9_1_frozen_n8192" not in report:
        lines.extend(
            [
                "",
                "## Frozen benchmark",
                "",
                "Not run. The preregistered development gate failed, so no V9 frozen",
                "predictions or judge transcripts were generated and V5.1 remains the",
                "selected submission model.",
            ]
        )
    else:
        baseline_frozen = report["v5_1_frozen_n8192"]
        selected_frozen = report["v9_1_frozen_n8192"]
        lines.extend(
            [
                "",
                "## One locked-margin frozen evaluation",
                "",
                "| Metric | v5.1 / 8192 | v9.1 / 8192 | Delta |",
                "|---|---:|---:|---:|",
                _metric_row("Exact patch accuracy", "exact_patch_accuracy", baseline_frozen, selected_frozen),
                _metric_row("Edge F1", "edge_f1", baseline_frozen, selected_frozen),
                _metric_row("Relation macro-F1", "relation_macro_f1", baseline_frozen, selected_frozen),
                f"| ATTACK F1 | {_pct(baseline_frozen['relation_metrics']['ATTACK']['f1'])} | **{_pct(selected_frozen['relation_metrics']['ATTACK']['f1'])}** | {100 * (selected_frozen['relation_metrics']['ATTACK']['f1'] - baseline_frozen['relation_metrics']['ATTACK']['f1']):+.1f} pp |",
                f"| False edges/update | {baseline_frozen['false_edges_per_update']:.3f} | **{selected_frozen['false_edges_per_update']:.3f}** | {selected_frozen['false_edges_per_update'] - baseline_frozen['false_edges_per_update']:+.3f} |",
                f"| NONE cases with false edges | {baseline_frozen['none_diagnostics']['scenarios_with_false_edges']}/6 | **{selected_frozen['none_diagnostics']['scenarios_with_false_edges']}/6** | {selected_frozen['none_diagnostics']['scenarios_with_false_edges'] - baseline_frozen['none_diagnostics']['scenarios_with_false_edges']:+d} |",
                "",
                "Frozen promotion checks:",
                "",
            ]
        )
        for name, passed in report["promotion_checks"].items():
            lines.append(f"- {'PASS' if passed else 'FAIL'}: `{name}`")
        if report.get("judge_metrics"):
            judge = report["judge_metrics"]
            lines.extend(
                [
                    "",
                    f"The unchanged blinded judge completed 30/30 calls: Spec adherence {judge['mean_spec_adherence']:.3f}/4 and Robustness {judge['mean_robustness']:.3f}/4.",
                ]
            )
        else:
            lines.extend(
                [
                    "",
                    "The judge was not run because deterministic promotion did not pass.",
                ]
            )
    lines.extend(
        [
            "",
            "## Reproduce",
            "",
            "```bash",
            "PYTHONPATH=src .venv/bin/python scripts/build_dialam_v9_mining_inputs.py",
            "modal run --detach scripts/modal_dialam_qlora.py --action mine-v9 --output-path data/dialam/training/v9_mining_scores_n12288.jsonl",
            "PYTHONPATH=src .venv/bin/python scripts/build_dialam_training_v9.py",
            "modal run --detach scripts/modal_dialam_qlora.py --action train --size 8192 --dataset-version v9 --resume-mode auto --output-path artifacts/dialam_qlora/v9_n8192/remote_training_result.json",
            "modal run --detach scripts/modal_dialam_qlora.py --action evaluate --target tuned --size 8192 --dataset-version v9 --eval-split v3_dev --output-path results/dialam_model_generation/v9_n8192_dev_raw/predictions.jsonl",
            "PYTHONPATH=src .venv/bin/python scripts/calibrate_dialam_v9.py",
            "# Continue only when the development report says PASS_RUN_FROZEN_ONCE.",
            "modal run --detach scripts/modal_dialam_qlora.py --action evaluate --target tuned --size 8192 --dataset-version v9 --eval-split frozen --output-path results/dialam_model_generation/v9_n8192_frozen_raw/predictions.jsonl",
            "PYTHONPATH=src .venv/bin/python scripts/apply_dialam_v5_margin.py --predictions results/dialam_model_generation/v9_n8192_frozen_raw/predictions.jsonl --examples data/dialam/balanced_diagnostic_eval_examples.jsonl --margin <locked-development-margin> --dataset-version v9.1 --output results/dialam_model_generation/v9_1_n8192/predictions.jsonl --summary reports/dialam_v9_frozen_application.json",
            "# Run the unchanged judge only when every deterministic check passes.",
            "PYTHONPATH=src .venv/bin/python scripts/evaluate_dialam_model.py --predictions results/dialam_model_generation/v9_1_n8192/predictions.jsonl --output-dir results/dialam_model_eval/v9_1_n8192 --approval APPROVE_DIALAM_MODEL_JUDGING",
            "PYTHONPATH=src .venv/bin/python scripts/build_dialam_v9_report.py",
            "```",
            "",
        ]
    )
    DOC.write_text("\n".join(lines), encoding="utf-8")


def build_report() -> dict[str, Any]:
    calibration = _load(CALIBRATION)
    training = _load(TRAINING_RESULT)
    reused = _load(IDEMPOTENT_RESULT)
    data = _load(DATA_MANIFEST)
    baseline = _load(V5_1_RESULT)
    resume_smoke = _load(RESUME_SMOKE)
    mining_resume = _load(MINING_RESUME)
    selected = calibration["selected"]

    if training["dataset_version"] != "v9" or training["size"] != 8192:
        raise ValueError("V9 training identity is wrong")
    if training["train_sha256"] != data["output"]["sha256"]:
        raise ValueError("V9 training bytes differ from the public manifest")
    if training["source_adapter"]["tree_sha256"] != data["continuation"][
        "source_adapter_tree_sha256"
    ]:
        raise ValueError("V9 did not continue from the frozen V5.1 adapter")
    if training["loss_config"]["name"] != data["objective"]["name"]:
        raise ValueError("V9 remote loss differs from the preregistered objective")
    for key in ("epochs", "learning_rate", "effective_batch_size", "seed"):
        if training["fixed_config"][key] != data["fixed_training_config"][key]:
            raise ValueError(f"V9 remote {key} differs from the public manifest")
    if not training["reload_verified"]:
        raise ValueError("V9 final adapter reload failed")
    if not resume_smoke["all_checks_passed"]:
        raise ValueError("V9 full run proceeded without a passing resume smoke")
    if not mining_resume["all_checks_passed"]:
        raise ValueError("V9 corpus was built without passing resumable mining")
    if reused.get("idempotent_reuse") is not True:
        raise ValueError("V9 completed-run idempotence check did not short circuit")
    if (
        reused["global_step"] != training["global_step"]
        or reused["adapter_tree_sha256"] != training["adapter_tree_sha256"]
    ):
        raise ValueError("V9 idempotent rerun returned different completed state")
    if calibration["artifacts"]["input_predictions_sha256"] != file_sha256(DEV_RAW):
        raise ValueError("V9 calibration input hash differs from raw development output")
    if file_sha256(DEV_IDEMPOTENT) != file_sha256(DEV_RAW):
        raise ValueError("V9 idempotent development replay changed prediction bytes")
    dev_first_resume = _load(DEV_RESUME)
    dev_idempotent_resume = _load(DEV_IDEMPOTENT_RESUME)
    if (
        dev_first_resume.get("idempotent_reuse") is not False
        or dev_first_resume.get("newly_completed_scenarios") != 30
        or dev_idempotent_resume.get("idempotent_reuse") is not True
        or dev_idempotent_resume.get("resumed_completed_scenarios") != 30
        or dev_idempotent_resume.get("newly_completed_scenarios") != 0
    ):
        raise ValueError("V9 development resume/idempotence evidence is incomplete")
    if (
        dev_first_resume["identity_sha256"]
        != dev_idempotent_resume["identity_sha256"]
        or dev_first_resume["predictions_sha256"]
        != dev_idempotent_resume["predictions_sha256"]
    ):
        raise ValueError("V9 development replay identity or semantic hash changed")

    selected_dev = {
        **selected["metrics"],
        "none_diagnostics": selected["none_diagnostics"],
    }
    relation_false_positives = {
        label: values["false_positive"]
        for label, values in selected_dev["relation_metrics"].items()
    }
    false_positive_edges = selected_dev["false_positive_edges"]
    dominant_label = (
        max(
            relation_false_positives,
            key=lambda label: (relation_false_positives[label], label),
        )
        if false_positive_edges
        else None
    )
    report: dict[str, Any] = {
        "schema_version": "dialam_v9_selected_model_hard_negative_result_v1",
        "experiment": "v9.1/n8192 QT30 selected-model hard-negative correction",
        "selection_decision": calibration["development_decision"],
        "development_gate_passed": selected["development_gate_passed"],
        "development_gate_checks": selected["development_gate_checks"],
        "failed_development_conditions": sorted(
            name
            for name, passed in selected["development_gate_checks"].items()
            if not passed
        ),
        "v5_1_development_n8192": baseline["v5_1_development_n8192"],
        "v9_1_development_n8192": selected_dev,
        "v5_1_frozen_n8192": baseline["v5_1_frozen_n8192"],
        "development_diagnosis": {
            "failure_mode": "overcorrected_to_none_with_positive_recall_collapse",
            "false_positive_edges": false_positive_edges,
            "false_negative_edges": selected_dev["false_negative_edges"],
            "dominant_false_positive_relation": dominant_label,
            "dominant_false_positive_count": (
                relation_false_positives[dominant_label] if dominant_label else 0
            ),
            "gold_edge_count": (
                selected_dev["true_positive_edges"]
                + selected_dev["false_negative_edges"]
            ),
            "true_positive_edges": selected_dev["true_positive_edges"],
            "support_true_positive_edges": selected_dev["relation_metrics"][
                "SUPPORT"
            ]["true_positive"],
            "rephrase_true_positive_edges": selected_dev["relation_metrics"][
                "REPHRASE"
            ]["true_positive"],
            "none_scenarios_with_false_edges": selected_dev["none_diagnostics"][
                "scenarios_with_false_edges"
            ],
            "relation_false_positive_counts": relation_false_positives,
        },
        "training": {
            "modal_training_app_id": "ap-ZaY1OpQa7U8NHDuKSC6W7z",
            "size": training["size"],
            "dataset_version": training["dataset_version"],
            "train_sha256": training["train_sha256"],
            "global_step": training["global_step"],
            "fixed_config": training["fixed_config"],
            "loss_config": training["loss_config"],
            "metrics": training["metrics"],
            "source_adapter": training["source_adapter"],
            "adapter_tree_sha256": training["adapter_tree_sha256"],
            "adapter_files": training["adapter_files"],
            "reload_verified": training["reload_verified"],
            "resumability": training["resumability"],
            "idempotent_reuse_check": {
                "returned_without_training": True,
                "global_step": reused["global_step"],
                "adapter_tree_sha256": reused["adapter_tree_sha256"],
                "artifact_sha256": file_sha256(IDEMPOTENT_RESULT),
            },
        },
        "corpus": {
            "row_label_counts": data["row_label_counts"],
            "row_role_counts": data["row_role_counts"],
            "selected_winning_positive_labels": data[
                "selected_winning_positive_labels"
            ],
            "selected_v5_1_margin_false_predictions": data[
                "selected_v5_1_margin_false_predictions"
            ],
            "selected_model_hardness": data["selected_model_hardness"],
            "dialogue_leakage": data["dialogue_leakage"],
            "new_external_data": data["new_external_data"],
        },
        "mining_resumability": mining_resume,
        "development_evaluation_resume": {
            "first_run_modal_app_id": "ap-MoyJyJz9zLQiypyu7DjYY7",
            "first_run": dev_first_resume,
            "idempotent_replay_modal_app_id": "ap-asdvBImc63OJdvcV2j9LrM",
            "idempotent_replay": dev_idempotent_resume,
            "prediction_bytes_identical": True,
        },
        "frozen_status": calibration["frozen_status"],
        "frozen_candidate_calls": 0,
        "frozen_judge_calls": 0,
        "promotion_passed": False,
        "artifacts": _available_artifacts(
            {
                "preregistration": PREREGISTRATION,
                "data_manifest": DATA_MANIFEST,
                "mining_resume": MINING_RESUME,
                "resume_smoke": RESUME_SMOKE,
                "training_result": TRAINING_RESULT,
                "idempotent_training_result": IDEMPOTENT_RESULT,
                "development_raw_predictions": DEV_RAW,
                "development_eval_resume": DEV_RESUME,
                "development_idempotent_predictions": DEV_IDEMPOTENT,
                "development_idempotent_resume": DEV_IDEMPOTENT_RESUME,
                "development_selected_predictions": DEV_SELECTED,
                "calibration": CALIBRATION,
            }
        ),
    }

    frozen_files = (
        FROZEN_RAW,
        FROZEN_RESUME,
        FROZEN_SELECTED,
        FROZEN_APPLICATION,
        FROZEN_SUMMARY,
    )
    if not selected["development_gate_passed"]:
        if any(path.exists() for path in frozen_files):
            raise ValueError(
                "V9 frozen artifacts exist despite the failed development gate"
            )
    else:
        if not all(path.exists() for path in frozen_files[:4]):
            raise FileNotFoundError(
                "development passed; complete the one frozen generation and locked-margin application"
            )
        frozen_application = _load(FROZEN_APPLICATION)
        if frozen_application["dataset_version"] != "v9.1":
            raise ValueError("frozen application is not labeled v9.1")
        if frozen_application["margin"] != selected["margin"]:
            raise ValueError("frozen V9 margin differs from development selection")
        if frozen_application["examples_sha256"] != baseline["evaluation_accounting"][
            "eval_sha256"
        ]:
            raise ValueError("V9 reused frozen evaluation hash changed")
        metrics = frozen_application["metrics"]
        none = frozen_application["none_diagnostics"]
        checks = _promotion_checks(metrics, none, data["frozen_promotion_gate"])
        deterministic_passed = all(checks.values())
        report.update(
            {
                "selection_decision": (
                    "AWAITING_REQUIRED_FROZEN_JUDGE"
                    if deterministic_passed and not FROZEN_SUMMARY.exists()
                    else "PROMOTE_V9_1_AS_SUBMISSION_MODEL"
                    if deterministic_passed
                    else "RETAIN_V5_1_V9_FAILED_FROZEN_PROMOTION_GATE"
                ),
                "frozen_status": (
                    "DETERMINISTIC_PASS_JUDGE_REQUIRED"
                    if deterministic_passed and not FROZEN_SUMMARY.exists()
                    else "COMPLETE"
                ),
                "frozen_candidate_calls": 30,
                "frozen_evaluation_resume": _load(FROZEN_RESUME),
                "v9_1_frozen_n8192": {
                    **metrics,
                    "none_diagnostics": none,
                },
                "promotion_checks": checks,
                "deterministic_promotion_passed": deterministic_passed,
                "failed_promotion_conditions": sorted(
                    name for name, passed in checks.items() if not passed
                ),
            }
        )
        report["artifacts"].update(
            _available_artifacts(
                {
                    "frozen_raw_predictions": FROZEN_RAW,
                    "frozen_eval_resume": FROZEN_RESUME,
                    "frozen_selected_predictions": FROZEN_SELECTED,
                    "frozen_application": FROZEN_APPLICATION,
                }
            )
        )
        if FROZEN_SUMMARY.exists():
            if not deterministic_passed:
                raise ValueError("V9 judge ran despite failed deterministic promotion")
            summary = _load(FROZEN_SUMMARY)
            if summary["eval_sha256"] != baseline["evaluation_accounting"][
                "eval_sha256"
            ]:
                raise ValueError("V9 judge used a different frozen evaluation")
            if summary["judge_rubric_sha256"] != baseline["evaluation_accounting"][
                "judge_rubric_sha256"
            ]:
                raise ValueError("V9 judge rubric differs from the frozen rubric")
            if summary["judge_identity_blinded"] is not True:
                raise ValueError("V9 judge was not identity blinded")
            if summary["deterministic_metrics"] != metrics:
                raise ValueError("V9 judge summary deterministic metrics changed")
            report.update(
                {
                    "judge_metrics": summary["judge_metrics"],
                    "frozen_judge_calls": 30,
                    "promotion_passed": True,
                }
            )
            report["v9_1_frozen_n8192"]["judge_metrics"] = summary[
                "judge_metrics"
            ]
            report["artifacts"].update(
                _available_artifacts(
                    {
                        "frozen_summary": FROZEN_SUMMARY,
                        "judge_transcripts": JUDGE_TRANSCRIPTS,
                        "judge_records": JUDGE_RECORDS,
                    }
                )
            )

    OUTPUT.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_doc(report)
    write_experiment_history()
    return report


def main() -> None:
    report = build_report()
    print(
        json.dumps(
            {
                "selection_decision": report["selection_decision"],
                "development_gate_checks": report["development_gate_checks"],
                "promotion_checks": report.get("promotion_checks"),
                "promotion_passed": report["promotion_passed"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

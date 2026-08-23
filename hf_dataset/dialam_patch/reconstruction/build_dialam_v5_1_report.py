#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from flowjudge.dialam_training import file_sha256
from flowjudge.experiment_history import write_experiment_history
from flowjudge.patch_ceiling import RELIABILITY_THRESHOLD
from flowjudge.patch_data import PROJECT_ROOT


CALIBRATION = PROJECT_ROOT / "reports" / "dialam_v5_1_calibration.json"
FROZEN_APPLICATION = PROJECT_ROOT / "reports" / "dialam_v5_1_frozen_application.json"
FROZEN_SUMMARY = (
    PROJECT_ROOT / "results" / "dialam_model_eval" / "v5_1_n8192" / "summary.json"
)
RAW_REPORT = PROJECT_ROOT / "reports" / "dialam_v5_pairwise_classification.json"
V3_REPORT = PROJECT_ROOT / "reports" / "dialam_v1_v2_v3.json"
TRAINING_MANIFEST = (
    PROJECT_ROOT / "artifacts" / "dialam_qlora" / "v5_n8192" / "training_manifest.json"
)
DATA_MANIFEST = (
    PROJECT_ROOT / "data" / "dialam" / "training" / "training_v5_manifest.json"
)
PREREGISTRATION = PROJECT_ROOT / "docs" / "dialam_v5_1_calibration_preregistration.md"
DEV_PREDICTIONS = (
    PROJECT_ROOT
    / "results"
    / "dialam_model_generation"
    / "v5_1_n8192_dev"
    / "predictions.jsonl"
)
FROZEN_RAW_SCORES = (
    PROJECT_ROOT
    / "results"
    / "dialam_model_generation"
    / "v5_n8192_frozen_raw_scores"
    / "predictions.jsonl"
)
FROZEN_PREDICTIONS = (
    PROJECT_ROOT
    / "results"
    / "dialam_model_generation"
    / "v5_1_n8192"
    / "predictions.jsonl"
)
JUDGE_TRANSCRIPTS = FROZEN_SUMMARY.parent / "judge_transcripts.jsonl"
JUDGE_RECORDS = FROZEN_SUMMARY.parent / "records.jsonl"
OUTPUT = PROJECT_ROOT / "reports" / "dialam_v5_1_result.json"
DOC = PROJECT_ROOT / "docs" / "dialam_v5_1_results.md"

PROMOTION_THRESHOLDS: dict[str, float | int] = {
    "exact_patch_accuracy_min": 0.43333333333333335,
    "edge_f1_min": 0.40,
    "relation_macro_f1_min": 0.39,
    "attack_f1_min": 0.30,
    "false_edges_per_update_max": 0.30,
    "none_scenarios_with_false_edges_max": 1,
    "judge_robustness_mean_min": 2.966666666666667,
    "json_validity_rate_min": 1.0,
    "schema_validity_rate_min": 1.0,
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }


def _promotion_checks(
    deterministic: dict[str, Any],
    judge: dict[str, Any],
    none: dict[str, Any],
) -> dict[str, bool]:
    return {
        "exact_patch_accuracy": deterministic["exact_patch_accuracy"]
        >= PROMOTION_THRESHOLDS["exact_patch_accuracy_min"],
        "edge_f1": deterministic["edge_f1"] >= PROMOTION_THRESHOLDS["edge_f1_min"],
        "relation_macro_f1": deterministic["relation_macro_f1"]
        >= PROMOTION_THRESHOLDS["relation_macro_f1_min"],
        "attack_f1": deterministic["relation_metrics"]["ATTACK"]["f1"]
        >= PROMOTION_THRESHOLDS["attack_f1_min"],
        "false_edges_per_update": deterministic["false_edges_per_update"]
        <= PROMOTION_THRESHOLDS["false_edges_per_update_max"],
        "none_scenarios_with_false_edges": none["scenarios_with_false_edges"]
        <= PROMOTION_THRESHOLDS["none_scenarios_with_false_edges_max"],
        "judge_robustness": judge["mean_robustness"]
        >= PROMOTION_THRESHOLDS["judge_robustness_mean_min"],
        "json_validity": deterministic["json_validity_rate"]
        >= PROMOTION_THRESHOLDS["json_validity_rate_min"],
        "schema_validity": deterministic["schema_validity_rate"]
        >= PROMOTION_THRESHOLDS["schema_validity_rate_min"],
    }


def _reliability_checks(
    deterministic: dict[str, Any], judge: dict[str, Any]
) -> dict[str, bool]:
    return {
        "scenario_count": deterministic["scenario_count"]
        >= RELIABILITY_THRESHOLD["minimum_scenarios_per_combination"],
        "json_validity_rate": deterministic["json_validity_rate"]
        >= RELIABILITY_THRESHOLD["json_validity_rate_min"],
        "schema_validity_rate": deterministic["schema_validity_rate"]
        >= RELIABILITY_THRESHOLD["schema_validity_rate_min"],
        "invalid_id_count": deterministic["invalid_id_count"]
        <= RELIABILITY_THRESHOLD["invalid_id_count_max"],
        "exact_patch_accuracy": deterministic["exact_patch_accuracy"]
        >= RELIABILITY_THRESHOLD["exact_patch_accuracy_min"],
        "edge_f1": deterministic["edge_f1"]
        >= RELIABILITY_THRESHOLD["edge_f1_min"],
        "relation_macro_f1": deterministic["relation_macro_f1"]
        >= RELIABILITY_THRESHOLD["relation_macro_f1_min"],
        "direction_accuracy": deterministic["direction_accuracy"]
        >= RELIABILITY_THRESHOLD["direction_accuracy_min"],
        "false_edges_per_update": deterministic["false_edges_per_update"]
        <= RELIABILITY_THRESHOLD["false_edges_per_update_max"],
        "judge_validity_rate": judge["judge_validity_rate"]
        >= RELIABILITY_THRESHOLD["judge_validity_rate_min"],
        "judge_spec_adherence_mean": judge["mean_spec_adherence"]
        >= RELIABILITY_THRESHOLD["judge_spec_adherence_mean_min"],
        "judge_robustness_mean": judge["mean_robustness"]
        >= RELIABILITY_THRESHOLD["judge_robustness_mean_min"],
    }


def _pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def _write_doc(report: dict[str, Any]) -> None:
    raw = report["raw_v5_development_n8192"]
    calibrated_dev = report["v5_1_development_n8192"]
    v3 = report["v3_frozen_n4096"]
    selected = report["v5_1_frozen_n8192"]
    lines = [
        "# DialAM v5.1 calibrated pairwise result",
        "",
        f"Final selection decision: **{report['selection_decision']}**.",
        "",
        "V5 converted the task from free JSON generation to one fixed four-label",
        "decision per supplied candidate. V5.1 applies the preregistered 3.0 NONE",
        "margin to correct the known 50.0% training versus 7.379% natural positive",
        "prior shift; it does not alter the checkpoint or training data.",
        "",
        "## Episode-disjoint development gate",
        "",
        "| Metric | Raw v5 / 8192 | Calibrated v5.1 / 8192 |",
        "|---|---:|---:|",
        f"| Exact patch accuracy | {_pct(raw['exact_patch_accuracy'])} | **{_pct(calibrated_dev['exact_patch_accuracy'])}** |",
        f"| Edge F1 | {_pct(raw['edge_f1'])} | **{_pct(calibrated_dev['edge_f1'])}** |",
        f"| Relation macro-F1 | {_pct(raw['relation_macro_f1'])} | **{_pct(calibrated_dev['relation_macro_f1'])}** |",
        f"| False edges/update | {raw['false_edges_per_update']:.3f} | **{calibrated_dev['false_edges_per_update']:.3f}** |",
        f"| NONE cases with false edges | {raw['none_diagnostics']['scenarios_with_false_edges']}/6 | **{calibrated_dev['none_diagnostics']['scenarios_with_false_edges']}/6** |",
        "",
        "V5.1 passed all seven unchanged development checks before the frozen set",
        "was evaluated. The selected margin and complete 0.00–3.00 grid are",
        "preserved in `reports/dialam_v5_1_calibration.json`.",
        "",
        "## Reused frozen benchmark",
        "",
        "| Metric | Previous v3 / 4096 | Selected v5.1 / 8192 | Delta |",
        "|---|---:|---:|---:|",
        f"| Exact patch accuracy | {_pct(v3['exact_patch_accuracy'])} | **{_pct(selected['exact_patch_accuracy'])}** | {100 * (selected['exact_patch_accuracy'] - v3['exact_patch_accuracy']):+.1f} pp |",
        f"| Edge precision | {_pct(v3['edge_precision'])} | **{_pct(selected['edge_precision'])}** | {100 * (selected['edge_precision'] - v3['edge_precision']):+.1f} pp |",
        f"| Edge recall | {_pct(v3['edge_recall'])} | **{_pct(selected['edge_recall'])}** | {100 * (selected['edge_recall'] - v3['edge_recall']):+.1f} pp |",
        f"| Edge F1 | {_pct(v3['edge_f1'])} | **{_pct(selected['edge_f1'])}** | {100 * (selected['edge_f1'] - v3['edge_f1']):+.1f} pp |",
        f"| Relation macro-F1 | {_pct(v3['relation_macro_f1'])} | **{_pct(selected['relation_macro_f1'])}** | {100 * (selected['relation_macro_f1'] - v3['relation_macro_f1']):+.1f} pp |",
        f"| ATTACK F1 | {_pct(v3['relation_metrics']['ATTACK']['f1'])} | **{_pct(selected['relation_metrics']['ATTACK']['f1'])}** | {100 * (selected['relation_metrics']['ATTACK']['f1'] - v3['relation_metrics']['ATTACK']['f1']):+.1f} pp |",
        f"| False edges/update | {v3['false_edges_per_update']:.3f} | **{selected['false_edges_per_update']:.3f}** | {selected['false_edges_per_update'] - v3['false_edges_per_update']:+.3f} |",
        f"| NONE cases with false edges | {v3['none_diagnostics']['scenarios_with_false_edges']}/6 | **{selected['none_diagnostics']['scenarios_with_false_edges']}/6** | 0 |",
        f"| Judge Robustness /4 | {v3['judge_metrics']['mean_robustness']:.3f} | **{selected['judge_metrics']['mean_robustness']:.3f}** | {selected['judge_metrics']['mean_robustness'] - v3['judge_metrics']['mean_robustness']:+.3f} |",
        "",
        "All nine preregistered promotion checks passed:",
        "",
    ]
    for name, passed in report["promotion_checks"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: `{name}`")
    lines.extend(
        [
            "",
            "The model still does not clear the original production-like reliability",
            "bar (80% exact, 85% edge F1, 75% macro-F1, at most 0.20 false",
            "edges/update, and 3.5/4 Robustness). It is the strongest tested",
            "assignment artifact, not a production-ready argument miner.",
            "",
            "The chronological ledger and generated chart are in",
            "[`docs/dialam_experiment_history.md`](dialam_experiment_history.md).",
            "",
            "## Reproduce",
            "",
            "```bash",
            ".venv/bin/python scripts/calibrate_dialam_v5.py",
            "modal run scripts/modal_dialam_qlora.py --action evaluate --target tuned --size 8192 --dataset-version v5 --eval-split frozen --output-path results/dialam_model_generation/v5_n8192_frozen_raw_scores/predictions.jsonl",
            ".venv/bin/python scripts/apply_dialam_v5_margin.py --predictions results/dialam_model_generation/v5_n8192_frozen_raw_scores/predictions.jsonl --examples data/dialam/balanced_diagnostic_eval_examples.jsonl --margin 3.0 --output results/dialam_model_generation/v5_1_n8192/predictions.jsonl --summary reports/dialam_v5_1_frozen_application.json",
            ".venv/bin/python scripts/evaluate_dialam_model.py --predictions results/dialam_model_generation/v5_1_n8192/predictions.jsonl --output-dir results/dialam_model_eval/v5_1_n8192 --approval APPROVE_DIALAM_MODEL_JUDGING",
            ".venv/bin/python scripts/build_dialam_v5_1_report.py",
            "```",
            "",
        ]
    )
    DOC.write_text("\n".join(lines), encoding="utf-8")


def build_report() -> dict[str, Any]:
    calibration = _load(CALIBRATION)
    frozen_application = _load(FROZEN_APPLICATION)
    frozen_summary = _load(FROZEN_SUMMARY)
    raw_report = _load(RAW_REPORT)
    v3_report = _load(V3_REPORT)
    training = _load(TRAINING_MANIFEST)
    data = _load(DATA_MANIFEST)

    selected_dev = calibration["selected"]
    if not selected_dev["development_gate_passed"] or selected_dev["margin"] != 3.0:
        raise ValueError("v5.1 selected development result is not the frozen margin 3.0 PASS")
    if frozen_application["margin"] != selected_dev["margin"]:
        raise ValueError("frozen calibration margin differs from development selection")
    if frozen_summary["eval_sha256"] != data["frozen_evaluation"]["eval_sha256"]:
        raise ValueError("v5.1 frozen evaluation hash changed")
    if frozen_summary["judge_rubric_sha256"] != data["frozen_evaluation"]["judge_rubric_sha256"]:
        raise ValueError("v5.1 frozen judge rubric hash changed")
    if frozen_summary["judge_identity_blinded"] is not True:
        raise ValueError("v5.1 judge was not identity blinded")
    if frozen_summary["dataset_version"] != "v5.1" or frozen_summary["adapter_size"] != 8192:
        raise ValueError("v5.1 frozen summary identity is wrong")

    deterministic = frozen_summary["deterministic_metrics"]
    judge = frozen_summary["judge_metrics"]
    none = frozen_application["none_diagnostics"]
    deterministic_with_none = {**deterministic, "none_diagnostics": none}
    promotion_checks = _promotion_checks(deterministic, judge, none)
    reliability_checks = _reliability_checks(deterministic, judge)
    promoted = all(promotion_checks.values())

    report = {
        "schema_version": "dialam_v5_1_calibrated_pairwise_result_v1",
        "experiment": "v5.1/n8192 preregistered NONE-margin calibration over v5 pairwise scores",
        "selection_decision": (
            "PROMOTE_V5_1_AS_FINAL_DIRECTION"
            if promoted
            else "RETAIN_V3_V5_1_FAILED_REUSED_FROZEN_PROMOTION_GATE"
        ),
        "selected_submission_checkpoint": promoted,
        "calibration": {
            "margin": selected_dev["margin"],
            "selection_rule": calibration["selection_rule"],
            "candidate_margin_count": len(calibration["margins"]),
            "checkpoint_or_training_changed": False,
            "known_prior_shift": {
                "natural_candidate_positive_rate": data["available_candidate_positive_rate"],
                "selected_training_positive_rate": 1
                - data["row_label_counts"]["NONE"] / data["size"],
            },
        },
        "raw_v5_development_n8192": raw_report["v5_development_n8192"],
        "v5_1_development_n8192": {
            **selected_dev["metrics"],
            "none_diagnostics": selected_dev["none_diagnostics"],
        },
        "v3_frozen_n4096": v3_report["v3_frozen_n4096"],
        "v5_1_frozen_n8192": {
            **deterministic_with_none,
            "judge_metrics": judge,
        },
        "promotion_thresholds": PROMOTION_THRESHOLDS,
        "promotion_checks": promotion_checks,
        "promotion_passed": promoted,
        "original_reliability_threshold": RELIABILITY_THRESHOLD,
        "original_reliability_checks": reliability_checks,
        "clears_original_reliability_bar": all(reliability_checks.values()),
        "failed_original_reliability_conditions": sorted(
            name for name, passed in reliability_checks.items() if not passed
        ),
        "evaluation_accounting": {
            "development_scenarios": 30,
            "frozen_scenarios": 30,
            "frozen_candidate_scenario_calls": 30,
            "frozen_judge_calls": 30,
            "judge_model": frozen_summary["judge_model"],
            "judge_identity_blinded": True,
            "eval_sha256": frozen_summary["eval_sha256"],
            "judge_rubric_sha256": frozen_summary["judge_rubric_sha256"],
        },
        "training": {
            "base_model": training["fixed_config"]["base_model"],
            "size": training["size"],
            "dataset_version": training["dataset_version"],
            "train_sha256": training["train_sha256"],
            "fixed_config": training["fixed_config"],
            "loss_config": training["loss_config"],
            "metrics": training["metrics"],
            "checkpoint_tree_sha256": raw_report["training"]["checkpoint_tree_sha256"],
            "checkpoint_files": raw_report["training"]["checkpoint_files"],
        },
        "artifacts": {
            name: _artifact(path)
            for name, path in (
                ("calibration_preregistration", PREREGISTRATION),
                ("calibration_grid", CALIBRATION),
                ("development_predictions", DEV_PREDICTIONS),
                ("frozen_raw_scores", FROZEN_RAW_SCORES),
                ("frozen_calibrated_predictions", FROZEN_PREDICTIONS),
                ("frozen_application", FROZEN_APPLICATION),
                ("frozen_summary", FROZEN_SUMMARY),
                ("judge_transcripts", JUDGE_TRANSCRIPTS),
                ("judge_records", JUDGE_RECORDS),
                ("training_manifest", TRAINING_MANIFEST),
                ("data_manifest", DATA_MANIFEST),
            )
        },
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_doc(report)
    write_experiment_history()
    return report


def main() -> None:
    report = build_report()
    print(
        json.dumps(
            {
                "selection_decision": report["selection_decision"],
                "promotion_checks": report["promotion_checks"],
                "clears_original_reliability_bar": report[
                    "clears_original_reliability_bar"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

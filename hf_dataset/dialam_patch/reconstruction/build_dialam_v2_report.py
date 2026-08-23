#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from flowjudge.dialam_training import file_sha256
from flowjudge.patch_data import DEFAULT_DIAGNOSTIC_EVAL_EXAMPLES_PATH, PROJECT_ROOT


V1_REPORT = PROJECT_ROOT / "reports" / "dialam_v1_efficiency_curve.json"
V2_SUMMARY = PROJECT_ROOT / "results" / "dialam_model_eval" / "v2_n2048" / "summary.json"
V2_TRAINING = PROJECT_ROOT / "artifacts" / "dialam_qlora" / "v2_n2048" / "remote_training_result.json"
V2_DATA_MANIFEST = PROJECT_ROOT / "data" / "dialam" / "training" / "training_v2_manifest.json"
OUTPUT = PROJECT_ROOT / "reports" / "dialam_v1_to_v2.json"
DOC = PROJECT_ROOT / "docs" / "dialam_v2_hard_negative_results.md"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    if not files:
        raise FileNotFoundError(f"checkpoint is empty or missing: {path}")
    for item in files:
        digest.update(str(item.relative_to(path)).encode())
        digest.update(b"\0")
        digest.update(file_sha256(item).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _none_diagnostics(predictions_path: Path) -> dict[str, Any]:
    gold = {row["example_id"]: row for row in _load_jsonl(DEFAULT_DIAGNOSTIC_EVAL_EXAMPLES_PATH)}
    none_ids = {
        example_id
        for example_id, row in gold.items()
        if not row["gold_patch"]["relations"]
    }
    labels = Counter()
    scenarios_with_false_edges = 0
    for row in _load_jsonl(predictions_path):
        if row["example_id"] not in none_ids:
            continue
        try:
            relations = json.loads(row["raw_response"])["relations"]
        except (json.JSONDecodeError, KeyError, TypeError):
            relations = []
        if relations:
            scenarios_with_false_edges += 1
            labels.update(edge.get("type", "INVALID") for edge in relations)
    return {
        "none_scenarios": len(none_ids),
        "scenarios_with_false_edges": scenarios_with_false_edges,
        "false_positive_rate": scenarios_with_false_edges / len(none_ids),
        "false_edge_labels": dict(sorted(labels.items())),
    }


def _private(path: Path) -> dict[str, str]:
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "sha256": file_sha256(path),
        "redistribution": "private/local only",
    }


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def build_report() -> dict[str, Any]:
    v1_report = json.loads(V1_REPORT.read_text(encoding="utf-8"))
    v1 = next(item for item in v1_report["curve"] if item["n"] == 2048)
    v2_summary = json.loads(V2_SUMMARY.read_text(encoding="utf-8"))
    v2 = v2_summary["deterministic_metrics"]
    training = json.loads(V2_TRAINING.read_text(encoding="utf-8"))
    data_manifest = json.loads(V2_DATA_MANIFEST.read_text(encoding="utf-8"))
    if training["fixed_config"] != v1_report["fixed_training_config"]:
        raise ValueError("v2 changed the frozen v1 QLoRA configuration")
    if v2_summary["eval_sha256"] != v1_report["frozen_eval_sha256"]:
        raise ValueError("v2 did not use the frozen v1 evaluation")
    if v2_summary["judge_rubric_sha256"] != v1_report["frozen_judge_rubric_sha256"]:
        raise ValueError("v2 did not use the frozen v1 judge rubric")

    edge_f1_gain = v2["edge_f1"] - v1["edge_f1"]
    false_edges_delta = v2["false_edges_per_update"] - v1["false_edges_per_update"]
    material = (
        edge_f1_gain >= 0.05
        and false_edges_delta <= 0.0
        and v2["json_validity_rate"] == 1.0
        and v2["schema_validity_rate"] == 1.0
    )
    result_dir = V2_SUMMARY.parent
    checkpoint_dir = V2_TRAINING.parent / "adapter"
    v2_none = _none_diagnostics(result_dir / "predictions.jsonl")
    reliability_threshold = v1_report["reliability_threshold"]
    v2_reliability_checks = {
        "direction_accuracy": v2["direction_accuracy"]
        >= reliability_threshold["direction_accuracy_min"],
        "edge_f1": v2["edge_f1"] >= reliability_threshold["edge_f1_min"],
        "exact_patch_accuracy": v2["exact_patch_accuracy"]
        >= reliability_threshold["exact_patch_accuracy_min"],
        "false_edges_per_update": v2["false_edges_per_update"]
        <= reliability_threshold["false_edges_per_update_max"],
        "invalid_id_count": v2["invalid_id_count"]
        <= reliability_threshold["invalid_id_count_max"],
        "json_validity_rate": v2["json_validity_rate"]
        >= reliability_threshold["json_validity_rate_min"],
        "judge_robustness_mean": v2_summary["judge_metrics"]["mean_robustness"]
        >= reliability_threshold["judge_robustness_mean_min"],
        "judge_spec_adherence_mean": v2_summary["judge_metrics"]["mean_spec_adherence"]
        >= reliability_threshold["judge_spec_adherence_mean_min"],
        "judge_validity_rate": v2_summary["judge_metrics"]["judge_validity_rate"]
        >= reliability_threshold["judge_validity_rate_min"],
        "relation_macro_f1": v2["relation_macro_f1"]
        >= reliability_threshold["relation_macro_f1_min"],
        "schema_validity_rate": v2["schema_validity_rate"]
        >= reliability_threshold["schema_validity_rate_min"],
    }
    deltas = {
        "exact_patch_accuracy": v2["exact_patch_accuracy"] - v1["exact_patch_accuracy"],
        "edge_precision": v2["edge_precision"] - v1["edge_precision"],
        "edge_recall": v2["edge_recall"] - v1["edge_recall"],
        "edge_f1": edge_f1_gain,
        "relation_macro_f1": v2["relation_macro_f1"] - v1["relation_macro_f1"],
        "false_edges_per_update": false_edges_delta,
        "false_positive_edges": v2["false_positive_edges"] - v1["false_positive_edges"],
        "none_scenarios_with_false_edges": (
            v2_none["scenarios_with_false_edges"]
            - v1["none_diagnostics"]["scenarios_with_false_edges"]
        ),
        "mean_judge_robustness": (
            v2_summary["judge_metrics"]["mean_robustness"]
            - v1["judge_metrics"]["mean_robustness"]
        ),
    }
    report = {
        "schema_version": "dialam_v1_to_v2_comparison_v1",
        "experiment": "v2/n2048 hard-negative correction versus v1/n2048",
        "material_improvement": material,
        "decision": (
            "USE_V2_AS_FINAL_DIRECTION" if material else "V2_DID_NOT_CLEAR_MATERIAL_IMPROVEMENT_BAR"
        ),
        "v2_clears_frozen_reliability_bar": all(v2_reliability_checks.values()),
        "v2_failed_reliability_thresholds": sorted(
            name for name, passed in v2_reliability_checks.items() if not passed
        ),
        "preregistered_criterion": data_manifest["preregistered_material_improvement"],
        "controlled_variables": {
            "base_model": v2_summary["model"],
            "n": 2048,
            "training_config_unchanged": True,
            "training_config_sha256": v1_report["fixed_training_config_sha256"],
            "eval_unchanged": True,
            "eval_sha256": v2_summary["eval_sha256"],
            "judge_rubric_unchanged": True,
            "judge_rubric_sha256": v2_summary["judge_rubric_sha256"],
        },
        "data_change": {
            "v1_none_examples": data_manifest["v1_n2048_none_count"],
            "v2_none_examples": data_manifest["v2_n2048_none_count"],
            "v2_category_counts": data_manifest["actual_category_counts"],
            "v2_relation_counts": data_manifest["relation_counts"],
            "hard_negative_policy": data_manifest["hard_negative_policy"],
            "v2_private_jsonl_sha256": data_manifest["output"]["sha256"],
        },
        "v1_n2048": {
            "exact_patch_accuracy": v1["exact_patch_accuracy"],
            "edge_precision": v1["edge_precision"],
            "edge_recall": v1["edge_recall"],
            "edge_f1": v1["edge_f1"],
            "relation_macro_f1": v1["relation_macro_f1"],
            "false_edges_per_update": v1["false_edges_per_update"],
            "false_positive_edges": v1["false_positive_edges"],
            "relation_metrics": v1["relation_metrics"],
            "none_diagnostics": v1["none_diagnostics"],
            "judge_metrics": v1["judge_metrics"],
        },
        "v2_n2048": {
            "exact_patch_accuracy": v2["exact_patch_accuracy"],
            "edge_precision": v2["edge_precision"],
            "edge_recall": v2["edge_recall"],
            "edge_f1": v2["edge_f1"],
            "relation_macro_f1": v2["relation_macro_f1"],
            "false_edges_per_update": v2["false_edges_per_update"],
            "false_positive_edges": v2["false_positive_edges"],
            "relation_metrics": v2["relation_metrics"],
            "none_diagnostics": v2_none,
            "judge_metrics": v2_summary["judge_metrics"],
        },
        "delta_v2_minus_v1": deltas,
        "training": {
            "metrics": training["metrics"],
            "reload_verified": training["reload_verified"],
            "train_sha256": training["train_sha256"],
            "checkpoint_tree_sha256": _tree_sha256(checkpoint_dir),
            "training_manifest_sha256": file_sha256(V2_TRAINING),
        },
        "private_artifacts": {
            "training_jsonl": _private(
                PROJECT_ROOT / "data" / "dialam" / "training" / "dialam_v2_n2048.jsonl"
            ),
            "predictions": _private(result_dir / "predictions.jsonl"),
            "records": _private(result_dir / "records.jsonl"),
            "judge_transcripts": _private(result_dir / "judge_transcripts.jsonl"),
            "checkpoint": {
                "path": str(checkpoint_dir.relative_to(PROJECT_ROOT)),
                "tree_sha256": _tree_sha256(checkpoint_dir),
                "redistribution": "private/local checkpoint pending publication",
            },
        },
        "reproduction_commands": [
            ".venv/bin/python scripts/build_dialam_training_v2.py",
            "modal run scripts/modal_dialam_qlora.py --action train --size 2048 --dataset-version v2",
            "modal run scripts/modal_dialam_qlora.py --action evaluate --target tuned --size 2048 --dataset-version v2 --output-path results/dialam_model_generation/v2_n2048/predictions.jsonl",
            ".venv/bin/python scripts/evaluate_dialam_model.py --predictions results/dialam_model_generation/v2_n2048/predictions.jsonl --output-dir results/dialam_model_eval/v2_n2048 --approval APPROVE_DIALAM_MODEL_JUDGING",
            ".venv/bin/python scripts/build_dialam_v2_report.py",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    v1_rel = v1["relation_metrics"]
    v2_rel = v2["relation_metrics"]
    markdown = f"""# DialAM v2 hard-negative result

Decision: **{report['decision']}**.

The sole intervention is the v2 data mix and deterministic hard-negative selection. Qwen3-0.6B, n=2048, the QLoRA configuration and seed, frozen 30-scenario eval, deterministic metrics, and blinded judge rubric are unchanged.

| Metric | v1 / n=2048 | v2 / n=2048 | Delta |
|---|---:|---:|---:|
| Exact patch accuracy | {_pct(v1['exact_patch_accuracy'])} | {_pct(v2['exact_patch_accuracy'])} | {deltas['exact_patch_accuracy']:+.3f} |
| Edge precision | {_pct(v1['edge_precision'])} | {_pct(v2['edge_precision'])} | {deltas['edge_precision']:+.3f} |
| Edge recall | {_pct(v1['edge_recall'])} | {_pct(v2['edge_recall'])} | {deltas['edge_recall']:+.3f} |
| Edge F1 | {_pct(v1['edge_f1'])} | {_pct(v2['edge_f1'])} | {deltas['edge_f1']:+.3f} |
| Relation macro-F1 | {_pct(v1['relation_macro_f1'])} | {_pct(v2['relation_macro_f1'])} | {deltas['relation_macro_f1']:+.3f} |
| False edges/update | {v1['false_edges_per_update']:.3f} | {v2['false_edges_per_update']:.3f} | {deltas['false_edges_per_update']:+.3f} |
| False edges | {v1['false_positive_edges']} | {v2['false_positive_edges']} | {deltas['false_positive_edges']:+d} |
| NONE cases with false edges | {v1['none_diagnostics']['scenarios_with_false_edges']}/6 | {v2_none['scenarios_with_false_edges']}/6 | {deltas['none_scenarios_with_false_edges']:+d} |
| SUPPORT F1 | {_pct(v1_rel['SUPPORT']['f1'])} | {_pct(v2_rel['SUPPORT']['f1'])} | {v2_rel['SUPPORT']['f1'] - v1_rel['SUPPORT']['f1']:+.3f} |
| ATTACK F1 | {_pct(v1_rel['ATTACK']['f1'])} | {_pct(v2_rel['ATTACK']['f1'])} | {v2_rel['ATTACK']['f1'] - v1_rel['ATTACK']['f1']:+.3f} |
| REPHRASE F1 | {_pct(v1_rel['REPHRASE']['f1'])} | {_pct(v2_rel['REPHRASE']['f1'])} | {v2_rel['REPHRASE']['f1'] - v1_rel['REPHRASE']['f1']:+.3f} |
| Judge robustness /4 | {v1['judge_metrics']['mean_robustness']:.2f} | {v2_summary['judge_metrics']['mean_robustness']:.2f} | {deltas['mean_judge_robustness']:+.3f} |

## Data intervention

v2 raises NONE coverage from {data_manifest['v1_n2048_none_count']} to {data_manifest['v2_n2048_none_count']} examples. Every selected NONE example is a no-edge fixed block from an update whose true direct edge occurs in another block, and every one has lexical content overlap with an in-block candidate. No easy-negative duplication, embeddings, semantic retrieval, new source data, model change, or hyperparameter search was used.

## Material-improvement rule

Before training, material improvement was defined as at least +0.05 absolute edge F1, no increase in false edges/update, and retained 100% JSON/schema validity. v2 does not clear that rule or the original frozen reliability bar. The machine-readable report records every condition, along with model/data/checkpoint hashes and the private artifact paths and hashes.

## Reproduce

```bash
{chr(10).join(report['reproduction_commands'])}
```

QT30-derived rows, candidate responses, judge transcripts, and checkpoints remain local/ignored. Only aggregate reports, hashes, schemas, and reconstruction/publication scripts are committed.
"""
    DOC.write_text(markdown, encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(build_report(), indent=2, sort_keys=True))

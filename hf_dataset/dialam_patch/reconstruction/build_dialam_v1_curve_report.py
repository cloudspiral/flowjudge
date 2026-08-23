#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from flowjudge.dialam_training import file_sha256
from flowjudge.patch_data import DEFAULT_DIAGNOSTIC_EVAL_EXAMPLES_PATH, PROJECT_ROOT


SIZES = (256, 512, 1024, 2048)
REPORTS_DIR = PROJECT_ROOT / "reports"
RELIABILITY_MANIFEST = (
    PROJECT_ROOT / "data" / "dialam" / "frozen_gate" / "prompt_ceiling_run_manifest.json"
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _tree_sha256(path: Path) -> str | None:
    if not path.is_dir():
        return None
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
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


def _private_artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "sha256": file_sha256(path),
        "redistribution": "private/local only",
    }


def _format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _svg(curve: list[dict[str, Any]]) -> str:
    x_positions = [90, 250, 410, 570]
    plot_top = 35
    plot_bottom = 255

    def points(field: str) -> str:
        return " ".join(
            f"{x},{plot_bottom - item[field] / 0.30 * (plot_bottom - plot_top):.1f}"
            for x, item in zip(x_positions, curve, strict=True)
        )

    labels = "".join(
        f'<text x="{x}" y="282" text-anchor="middle">{item["n"]}</text>'
        for x, item in zip(x_positions, curve, strict=True)
    )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="660" height="330" viewBox="0 0 660 330" role="img" aria-labelledby="title desc">
  <title id="title">DialAM v1 performance versus training examples</title>
  <desc id="desc">Exact patch accuracy and edge F1 for QLoRA checkpoints trained on 256, 512, 1024, and 2048 examples.</desc>
  <rect width="660" height="330" fill="white"/>
  <line x1="70" y1="{plot_top}" x2="70" y2="{plot_bottom}" stroke="#555"/>
  <line x1="70" y1="{plot_bottom}" x2="600" y2="{plot_bottom}" stroke="#555"/>
  <line x1="70" y1="145" x2="600" y2="145" stroke="#ddd" stroke-dasharray="4 4"/>
  <text x="55" y="260" text-anchor="end">0%</text>
  <text x="55" y="150" text-anchor="end">15%</text>
  <text x="55" y="40" text-anchor="end">30%</text>
  {labels}
  <text x="335" y="310" text-anchor="middle">Training examples (N)</text>
  <polyline points="{points('exact_patch_accuracy')}" fill="none" stroke="#2563eb" stroke-width="3"/>
  <polyline points="{points('edge_f1')}" fill="none" stroke="#dc2626" stroke-width="3"/>
  <line x1="255" y1="18" x2="285" y2="18" stroke="#2563eb" stroke-width="3"/>
  <text x="291" y="22">Exact patch accuracy</text>
  <line x1="465" y1="18" x2="495" y2="18" stroke="#dc2626" stroke-width="3"/>
  <text x="501" y="22">Edge F1</text>
</svg>
"""


def build_report() -> dict[str, Any]:
    curve: list[dict[str, Any]] = []
    reliability_threshold = json.loads(RELIABILITY_MANIFEST.read_text(encoding="utf-8"))[
        "reliability_threshold"
    ]
    fixed_config: dict[str, Any] | None = None
    private_artifacts: dict[str, Any] = {}
    for size in SIZES:
        result_dir = PROJECT_ROOT / "results" / "dialam_model_eval" / f"n{size}"
        summary_path = result_dir / "summary.json"
        predictions_path = result_dir / "predictions.jsonl"
        records_path = result_dir / "records.jsonl"
        judge_path = result_dir / "judge_transcripts.jsonl"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metrics = summary["deterministic_metrics"]
        checkpoint_dir = PROJECT_ROOT / "artifacts" / "dialam_qlora" / f"n{size}"
        training_path = checkpoint_dir / "training_manifest.json"
        if not training_path.exists():
            training_path = checkpoint_dir / "remote_training_result.json"
        training = json.loads(training_path.read_text(encoding="utf-8"))
        if fixed_config is None:
            fixed_config = training["fixed_config"]
        elif training["fixed_config"] != fixed_config:
            raise ValueError(f"n={size} changed the frozen training configuration")
        point = {
                "n": size,
                "exact_patch_accuracy": metrics["exact_patch_accuracy"],
                "exact_patch_count": metrics["exact_scenario_patch_count"],
                "edge_precision": metrics["edge_precision"],
                "edge_recall": metrics["edge_recall"],
                "edge_f1": metrics["edge_f1"],
                "relation_macro_f1": metrics["relation_macro_f1"],
                "direction_accuracy": metrics["direction_accuracy"],
                "false_edges_per_update": metrics["false_edges_per_update"],
                "false_positive_edges": metrics["false_positive_edges"],
                "false_negative_edges": metrics["false_negative_edges"],
                "relation_metrics": metrics["relation_metrics"],
                "json_validity_rate": metrics["json_validity_rate"],
                "schema_validity_rate": metrics["schema_validity_rate"],
                "invalid_id_count": metrics["invalid_id_count"],
                "judge_metrics": summary["judge_metrics"],
                "none_diagnostics": _none_diagnostics(predictions_path),
                "training_metrics": training["metrics"],
                "checkpoint_tree_sha256": _tree_sha256(
                    PROJECT_ROOT / "artifacts" / "dialam_qlora" / f"n{size}" / "adapter"
                ),
                "training_manifest_sha256": file_sha256(training_path),
            }
        reliability_checks = {
            "direction_accuracy": point["direction_accuracy"]
            >= reliability_threshold["direction_accuracy_min"],
            "edge_f1": point["edge_f1"] >= reliability_threshold["edge_f1_min"],
            "exact_patch_accuracy": point["exact_patch_accuracy"]
            >= reliability_threshold["exact_patch_accuracy_min"],
            "false_edges_per_update": point["false_edges_per_update"]
            <= reliability_threshold["false_edges_per_update_max"],
            "invalid_id_count": point["invalid_id_count"]
            <= reliability_threshold["invalid_id_count_max"],
            "json_validity_rate": point["json_validity_rate"]
            >= reliability_threshold["json_validity_rate_min"],
            "judge_robustness_mean": point["judge_metrics"]["mean_robustness"]
            >= reliability_threshold["judge_robustness_mean_min"],
            "judge_spec_adherence_mean": point["judge_metrics"]["mean_spec_adherence"]
            >= reliability_threshold["judge_spec_adherence_mean_min"],
            "judge_validity_rate": point["judge_metrics"]["judge_validity_rate"]
            >= reliability_threshold["judge_validity_rate_min"],
            "relation_macro_f1": point["relation_macro_f1"]
            >= reliability_threshold["relation_macro_f1_min"],
            "schema_validity_rate": point["schema_validity_rate"]
            >= reliability_threshold["schema_validity_rate_min"],
        }
        point["reliability_checks"] = reliability_checks
        point["clears_reliability_bar"] = all(reliability_checks.values())
        point["failed_reliability_thresholds"] = sorted(
            name for name, passed in reliability_checks.items() if not passed
        )
        curve.append(point)
        private_artifacts[f"n{size}"] = {
            "predictions": _private_artifact(predictions_path),
            "records": _private_artifact(records_path),
            "judge_transcripts": _private_artifact(judge_path),
            "checkpoint": {
                "path": f"artifacts/dialam_qlora/n{size}/adapter",
                "tree_sha256": curve[-1]["checkpoint_tree_sha256"],
                "redistribution": "private/local checkpoint pending publication decision",
            },
        }

    assert fixed_config is not None
    eval_hashes = {
        json.loads(
            (PROJECT_ROOT / "results" / "dialam_model_eval" / f"n{size}" / "summary.json").read_text()
        )["eval_sha256"]
        for size in SIZES
    }
    rubric_hashes = {
        json.loads(
            (PROJECT_ROOT / "results" / "dialam_model_eval" / f"n{size}" / "summary.json").read_text()
        )["judge_rubric_sha256"]
        for size in SIZES
    }
    if len(eval_hashes) != 1 or len(rubric_hashes) != 1:
        raise ValueError("the v1 curve did not use one frozen eval and judge rubric")
    report = {
        "schema_version": "dialam_v1_efficiency_curve_v1",
        "label": "v1 fixed data-efficiency curve",
        "base_model": "Qwen/Qwen3-0.6B",
        "sizes": list(SIZES),
        "frozen_eval_sha256": next(iter(eval_hashes)),
        "frozen_judge_rubric_sha256": next(iter(rubric_hashes)),
        "fixed_training_config": fixed_config,
        "fixed_training_config_sha256": hashlib.sha256(
            json.dumps(fixed_config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "curve": curve,
        "reliability_threshold": reliability_threshold,
        "smallest_n_reliably_holding_behavior": None,
        "any_v1_checkpoint_clears_reliability_bar": any(
            item["clears_reliability_bar"] for item in curve
        ),
        "best_v1_checkpoint": {
            "n": 2048,
            "reason": "highest exact patch accuracy, edge F1, and relation macro-F1",
        },
        "failure_diagnosis": {
            "support_false_positive_is_largest_label_at_every_n": True,
            "support_false_positives_by_n": {
                str(item["n"]): item["relation_metrics"]["SUPPORT"]["false_positive"]
                for item in curve
            },
            "total_false_positives_by_n": {
                str(item["n"]): item["false_positive_edges"] for item in curve
            },
            "none_false_positive_scenarios_by_n": {
                str(item["n"]): item["none_diagnostics"]["scenarios_with_false_edges"]
                for item in curve
            },
            "conclusion": (
                "SUPPORT remains the largest single false-positive class, although at n=2048 "
                "errors are distributed across SUPPORT, REPHRASE, and ATTACK. Scaling v1 does "
                "reduces NONE-case false-positive scenarios only from 3/6 to 2/6 at n=2048."
            ),
        },
        "private_artifacts": private_artifacts,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS_DIR / "dialam_v1_efficiency_curve.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    rows = []
    for item in curve:
        relation_metrics = item["relation_metrics"]
        rows.append(
            "| {n} | {exact} | {f1} | {macro} | {false_edges:.3f} | {fp} | {sup} / {rep} / {att} | {none}/6 | {robust:.2f} |".format(
                n=item["n"],
                exact=_format_percent(item["exact_patch_accuracy"]),
                f1=_format_percent(item["edge_f1"]),
                macro=_format_percent(item["relation_macro_f1"]),
                false_edges=item["false_edges_per_update"],
                fp=item["false_positive_edges"],
                sup=relation_metrics["SUPPORT"]["false_positive"],
                rep=relation_metrics["REPHRASE"]["false_positive"],
                att=relation_metrics["ATTACK"]["false_positive"],
                none=item["none_diagnostics"]["scenarios_with_false_edges"],
                robust=item["judge_metrics"]["mean_robustness"],
            )
        )
    markdown = """# DialAM v1 fixed data-efficiency curve

All four checkpoints use Qwen3-0.6B, the same frozen QLoRA configuration and seed, the same 30-scenario eval, and the same blinded judge rubric. Only the nested v1 training size changes.

| N | Exact patch | Edge F1 | Relation macro-F1 | False edges/update | False edges | FP SUPPORT / REPHRASE / ATTACK | NONE cases with FP | Judge robustness /4 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
""" + "\n".join(rows) + """

## Finding

n=2048 is the best v1 checkpoint on exact patch accuracy, edge F1, and relation macro-F1. It reaches 8/30 exact patches, but still emits 20 false edges (0.667/update). SUPPORT is the largest single false-positive class at every N: 14, 11, 9, then 8. On the six all-negative held-out scenarios, n=256/512/1024 make false predictions on 3/6 and n=2048 improves only to 2/6. The error therefore remains false-edge overprediction; scaling alone did not fix it. **No tested N reliably holds the behavior** under the frozen reliability bar.

The full aggregate metrics, frozen hashes, fixed config hash, checkpoint tree hashes, and private transcript/result hashes are in `reports/dialam_v1_efficiency_curve.json`. Text-bearing predictions, records, judge transcripts, QT30-derived training rows, and checkpoints remain local/ignored.

![v1 performance curve](../reports/dialam_v1_efficiency_curve.svg)

## Reproduce

```bash
.venv/bin/python scripts/build_dialam_training.py
for n in 256 512 1024 2048; do modal run scripts/modal_dialam_qlora.py --action train --size "$n" --dataset-version v1; done
for n in 256 512 1024 2048; do modal run scripts/modal_dialam_qlora.py --action evaluate --target tuned --size "$n" --dataset-version v1 --output-path "results/dialam_model_generation/n$n/predictions.jsonl"; done
for n in 256 512 1024 2048; do .venv/bin/python scripts/evaluate_dialam_model.py --predictions "results/dialam_model_generation/n$n/predictions.jsonl" --output-dir "results/dialam_model_eval/n$n" --approval APPROVE_DIALAM_MODEL_JUDGING; done
.venv/bin/python scripts/build_dialam_v1_curve_report.py
```
"""
    (PROJECT_ROOT / "docs" / "dialam_v1_efficiency_results.md").write_text(
        markdown,
        encoding="utf-8",
    )
    (REPORTS_DIR / "dialam_v1_efficiency_curve.svg").write_text(_svg(curve), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(build_report(), indent=2, sort_keys=True))

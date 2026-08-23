import json

from flowjudge.hf_eval import (
    _declared_base_model,
    _is_dialam_eval_set,
    _mlx_adapter_base_model,
    _resolve_backend,
    _results_table,
    build_parser,
)
from flowjudge.dialam_hf_eval import (
    _declared_dialam_base_model,
    _dialam_results_table,
    _pairwise_prediction,
)
from test_dialam_training_v5 import _example


def test_eval_cli_requires_prescribed_model_and_eval_set() -> None:
    args = build_parser().parse_args(["--model", "owner/model", "--eval-set", "staff.jsonl"])

    assert args.model == "owner/model"
    assert str(args.eval_set) == "staff.jsonl"
    assert args.backend == "auto"
    assert args.skip_judge is False


def test_results_table_contains_required_behavior_metrics() -> None:
    table = _results_table(
        {
            "base": {
                "normalized_exact_graph_match_rate": 0.25,
                "normalized_edge_f1": 0.5,
                "mean_spec_adherence": 1.5,
                "mean_robustness": 2.0,
                "false_positive_rate_on_topically_related_nonresponses": 0.25,
            }
        }
    )

    assert "Mean Spec adherence" in table
    assert "Mean Robustness" in table
    assert "25.0%" in table


def test_auto_detects_local_mlx_adapter_and_manifest(tmp_path) -> None:
    run_dir = tmp_path / "n12"
    adapter_dir = run_dir / "adapter"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"model": "mlx-community/Qwen3-0.6B-4bit"}),
        encoding="utf-8",
    )
    (run_dir / "flowjudge_training_manifest.json").write_text(
        json.dumps({"quantized_base_model": "mlx-community/Qwen3-0.6B-4bit"}),
        encoding="utf-8",
    )

    assert _resolve_backend(str(adapter_dir), "auto") == "mlx"
    assert _mlx_adapter_base_model(str(adapter_dir)) == "mlx-community/Qwen3-0.6B-4bit"


def test_fused_checkpoint_declares_transformers_evaluation_base(tmp_path) -> None:
    (tmp_path / "flowjudge_training_manifest.json").write_text(
        json.dumps({"evaluation_base_model": "Qwen/Qwen3-0.6B"}),
        encoding="utf-8",
    )

    assert _declared_base_model(str(tmp_path)) == "Qwen/Qwen3-0.6B"


def test_dialam_eval_set_is_auto_detected(tmp_path) -> None:
    path = tmp_path / "eval.jsonl"
    path.write_text(
        json.dumps({"schema_version": "dialam_incremental_patch_v1"}) + "\n",
        encoding="utf-8",
    )

    assert _is_dialam_eval_set(path) is True


def test_dialam_model_package_declares_canonical_base(tmp_path) -> None:
    (tmp_path / "dialam_v1_checkpoint_manifest.json").write_text(
        json.dumps({"fixed_config": {"base_model": "Qwen/Qwen3-0.6B"}}),
        encoding="utf-8",
    )

    assert _declared_dialam_base_model(str(tmp_path)) == "Qwen/Qwen3-0.6B"


def test_dialam_v5_model_package_declares_canonical_base(tmp_path) -> None:
    (tmp_path / "dialam_v5_checkpoint_manifest.json").write_text(
        json.dumps({"fixed_config": {"base_model": "Qwen/Qwen3-0.6B"}}),
        encoding="utf-8",
    )

    assert _declared_dialam_base_model(str(tmp_path)) == "Qwen/Qwen3-0.6B"


def test_pairwise_hf_inference_applies_fixed_none_margin_and_assembles_patch() -> None:
    class FakeGenerator:
        def score_completions(self, prompt, completions, *, max_sequence_length):
            assert completions == ("NONE", "SUPPORT", "ATTACK", "REPHRASE")
            assert max_sequence_length == 2048
            assert "complete_earlier_comparison_block" in prompt
            if 'CANDIDATE TARGET ID\n"old-a"' in prompt:
                return {
                    "NONE": 0.0,
                    "SUPPORT": 3.1,
                    "ATTACK": -1.0,
                    "REPHRASE": -1.0,
                }
            return {
                "NONE": 0.0,
                "SUPPORT": 2.9,
                "ATTACK": -1.0,
                "REPHRASE": -1.0,
            }

    response, decisions = _pairwise_prediction(
        FakeGenerator(),
        _example(update="support-a", label="SUPPORT"),
        {
            "allowed_labels": ["NONE", "SUPPORT", "ATTACK", "REPHRASE"],
            "none_margin": 3.0,
            "max_sequence_length": 2048,
        },
    )

    assert json.loads(response) == {
        "relations": [
            {"source": "new-support-a", "target": "old-a", "type": "SUPPORT"}
        ]
    }
    assert [item["label"] for item in decisions] == ["SUPPORT", "NONE", "NONE"]


def test_dialam_results_table_contains_required_metrics() -> None:
    deterministic = {
        "exact_scenario_patch_accuracy": 0.25,
        "edge_precision": 0.2,
        "edge_recall": 0.3,
        "edge_f1": 0.24,
        "relation_macro_f1": 0.1,
        "direction_accuracy": 1.0,
        "false_edges_per_update": 0.75,
        "json_validity_rate": 1.0,
        "invalid_id_count": 0,
    }
    table = _dialam_results_table(
        {
            "tuned": {
                "deterministic_metrics": deterministic,
                "judge_metrics": {
                    "mean_spec_adherence": 4.0,
                    "mean_robustness": 2.1,
                },
            }
        }
    )

    assert "Exact patch" in table
    assert "False edges/update" in table
    assert "25.0%" in table
    assert "2.10" in table

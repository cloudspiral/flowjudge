import json

from flowjudge.hf_eval import (
    _declared_base_model,
    _mlx_adapter_base_model,
    _resolve_backend,
    _results_table,
    build_parser,
)


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

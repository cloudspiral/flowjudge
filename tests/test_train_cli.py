import importlib.util
from pathlib import Path


def _load_train_module():
    path = Path(__file__).parents[1] / "scripts" / "train_qlora.py"
    spec = importlib.util.spec_from_file_location("train_qlora", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_qlora_cli_has_small_qwen_default() -> None:
    module = _load_train_module()
    args = module.build_parser().parse_args(
        ["--train-set", "train.jsonl", "--output-dir", "checkpoint"]
    )

    assert args.model == "Qwen/Qwen3-0.6B"
    assert args.epochs == 3.0


def test_mlx_qlora_cli_uses_quantized_small_qwen() -> None:
    path = Path(__file__).parents[1] / "scripts" / "train_mlx_qlora.py"
    spec = importlib.util.spec_from_file_location("train_mlx_qlora", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = module.build_parser().parse_args(
        ["--train-set", "train.jsonl", "--output-dir", "checkpoint"]
    )

    assert args.model == "mlx-community/Qwen3-0.6B-4bit"
    assert args.gradient_accumulation_steps == 4


def test_mlx_training_disables_qwen_thinking_without_mutating_source() -> None:
    path = Path(__file__).parents[1] / "scripts" / "train_mlx_qlora.py"
    spec = importlib.util.spec_from_file_location("train_mlx_qlora_thinking", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = [
        {"role": "user", "content": "Return JSON."},
        {"role": "assistant", "content": "{}"},
    ]

    result = module._disable_qwen_thinking(source)

    assert result[0]["content"].endswith("/no_think")
    assert source[0]["content"] == "Return JSON."

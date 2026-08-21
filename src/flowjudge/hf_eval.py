from __future__ import annotations

import argparse
import gc
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .data import PROJECT_ROOT
from .prompts import build_prompt
from .runner import _build_judge_prompt, _call_openai, _openai_client
from .scorer import score_records
from .training_data import DEFAULT_OWN_EVAL_PATH, load_eval_cases


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a Hugging Face FlowJudge model and, for an adapter, its base model"
    )
    parser.add_argument("--model", required=True, help="Hugging Face model or adapter repository ID")
    parser.add_argument("--eval-set", required=True, type=Path, help="BenchmarkCase JSONL path")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--judge-model", help="override JUDGE_MODEL")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--backend",
        choices=("auto", "transformers", "mlx"),
        default="auto",
        help="inference runtime; auto detects local MLX-LM adapters",
    )
    parser.add_argument(
        "--base-model",
        help="base checkpoint for an MLX-LM adapter; normally read from its training manifest",
    )
    parser.add_argument(
        "--skip-base",
        action="store_true",
        help="evaluate only the requested model even when it is a PEFT adapter",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = run_hf_evaluation(
        model_id=args.model,
        eval_set=args.eval_set,
        output_dir=args.output_dir,
        judge_model=args.judge_model,
        max_new_tokens=args.max_new_tokens,
        compare_base=not args.skip_base,
        backend=args.backend,
        base_model_override=args.base_model,
    )
    print(output_dir)


def run_hf_evaluation(
    *,
    model_id: str,
    eval_set: Path = DEFAULT_OWN_EVAL_PATH,
    output_dir: Path | None = None,
    judge_model: str | None = None,
    max_new_tokens: int = 512,
    compare_base: bool = True,
    backend: str = "auto",
    base_model_override: str | None = None,
) -> Path:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    fixed_judge = (judge_model or os.getenv("JUDGE_MODEL", "")).strip()
    if not api_key or not fixed_judge:
        raise RuntimeError("OPENAI_API_KEY and JUDGE_MODEL are required for reproducible judging")

    cases = load_eval_cases(eval_set)
    if not cases:
        raise ValueError("evaluation set is empty")
    if output_dir is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        output_dir = PROJECT_ROOT / "results" / "model_eval" / stamp
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_backend = _resolve_backend(model_id, backend)
    if resolved_backend == "mlx":
        base_model = (
            base_model_override or _mlx_adapter_base_model(model_id)
            if compare_base
            else None
        )
        if base_model:
            targets = [
                ("base", base_model, None),
                ("tuned", base_model, model_id),
            ]
        else:
            targets = [("model", model_id, None)]
    else:
        base_model = (
            _adapter_base_model(model_id) or _declared_base_model(model_id)
            if compare_base
            else None
        )
        targets = (
            [("base", base_model, None), ("tuned", model_id, None)]
            if base_model
            else [("model", model_id, None)]
        )
    judge_client = _openai_client(api_key)
    all_summaries: dict[str, Any] = {}

    for label, target_model, adapter_path in targets:
        assert target_model is not None
        generator = _make_generator(
            resolved_backend,
            target_model,
            adapter_path=adapter_path,
            max_new_tokens=max_new_tokens,
        )
        records: list[dict[str, Any]] = []
        raw_responses_path = output_dir / f"{label}_raw_responses.jsonl"
        raw_judgments_path = output_dir / f"{label}_raw_judgments.jsonl"
        with raw_responses_path.open("w", encoding="utf-8") as responses_file, raw_judgments_path.open(
            "w", encoding="utf-8"
        ) as judgments_file:
            for case in cases:
                prompt = build_prompt("zero_shot", case.scenario)
                raw_response = generator.generate(prompt)
                judge_prompt = _build_judge_prompt(case, raw_response)
                raw_judge, judge_envelope = _call_openai(judge_client, fixed_judge, judge_prompt)
                record = {
                    "assignment_id": f"{label}__{case.scenario.scenario_id}",
                    "scenario_id": case.scenario.scenario_id,
                    "category": case.scenario.category.value,
                    "split": case.scenario.split.value,
                    "phenomena": [item.value for item in case.scenario.phenomena],
                    "provider": resolved_backend,
                    "model": model_id if adapter_path else target_model,
                    "prompt": "zero_shot",
                    "raw_response": raw_response,
                    "judge_model": fixed_judge,
                    "raw_judge_response": raw_judge,
                }
                records.append(record)
                responses_file.write(
                    json.dumps(
                        {
                            "scenario_id": case.scenario.scenario_id,
                            "model": model_id if adapter_path else target_model,
                            "prompt": prompt,
                            "response": raw_response,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                responses_file.flush()
                judgments_file.write(
                    json.dumps(
                        {
                            "scenario_id": case.scenario.scenario_id,
                            "judge_model": fixed_judge,
                            "response": raw_judge,
                            "envelope": json.loads(judge_envelope),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                judgments_file.flush()
                print(f"evaluated {label} {case.scenario.scenario_id}", flush=True)
        all_summaries[label] = score_records(cases, records)
        generator.close()

    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "requested_model": model_id,
        "detected_base_model": base_model,
        "backend": resolved_backend,
        "judge_model": fixed_judge,
        "eval_set": str(eval_set),
        "eval_scenarios": len(cases),
        "prompt": "zero_shot",
        "summaries": all_summaries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "results_table.md").write_text(
        _results_table(all_summaries), encoding="utf-8"
    )
    return output_dir


class HuggingFaceGenerator:
    def __init__(self, model_id: str, *, max_new_tokens: int) -> None:
        try:
            import torch
            from peft import PeftConfig, PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "model evaluation dependencies are missing; install the train dependency group"
            ) from exc

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        try:
            peft_config = PeftConfig.from_pretrained(model_id)
        except Exception:
            peft_config = None

        if peft_config is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype="auto",
                device_map="auto",
            )
        else:
            base_id = peft_config.base_model_name_or_path
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            base = AutoModelForCausalLM.from_pretrained(
                base_id,
                torch_dtype="auto",
                device_map="auto",
            )
            self.model = PeftModel.from_pretrained(base, model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()

    def generate(self, prompt: str) -> str:
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        generated = output[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def close(self) -> None:
        del self.model
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


class MLXGenerator:
    def __init__(
        self,
        model_id: str,
        *,
        adapter_path: str | None,
        max_new_tokens: int,
    ) -> None:
        try:
            import mlx.core as mx
            from mlx_lm import generate, load
            from mlx_lm.sample_utils import make_sampler
        except (ImportError, RuntimeError) as exc:
            raise RuntimeError(
                "MLX evaluation requires Apple Silicon with the mlx-train dependency group"
            ) from exc

        self.mx = mx
        self.generate_text = generate
        self.sampler = make_sampler(temp=0.0)
        self.max_new_tokens = max_new_tokens
        self.model, self.tokenizer = load(model_id, adapter_path=adapter_path)

    def generate(self, prompt: str) -> str:
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return self.generate_text(
            self.model,
            self.tokenizer,
            prompt=text,
            max_tokens=self.max_new_tokens,
            sampler=self.sampler,
            verbose=False,
        ).strip()

    def close(self) -> None:
        del self.model
        gc.collect()
        self.mx.clear_cache()


def _make_generator(
    backend: str,
    model_id: str,
    *,
    adapter_path: str | None,
    max_new_tokens: int,
) -> HuggingFaceGenerator | MLXGenerator:
    if backend == "mlx":
        return MLXGenerator(
            model_id,
            adapter_path=adapter_path,
            max_new_tokens=max_new_tokens,
        )
    return HuggingFaceGenerator(model_id, max_new_tokens=max_new_tokens)


def _adapter_base_model(model_id: str) -> str | None:
    try:
        from peft import PeftConfig
    except ImportError as exc:
        raise RuntimeError(
            "model evaluation dependencies are missing; install the train dependency group"
        ) from exc
    try:
        return str(PeftConfig.from_pretrained(model_id).base_model_name_or_path)
    except Exception:
        return None


def _resolve_backend(model_id: str, requested: str) -> str:
    if requested != "auto":
        return requested
    config = _read_adapter_config(model_id)
    if "model" in config and "base_model_name_or_path" not in config:
        return "mlx"
    if model_id.startswith("mlx-community/"):
        return "mlx"
    return "transformers"


def _mlx_adapter_base_model(model_id: str) -> str | None:
    adapter_path = Path(model_id)
    config = _read_adapter_config(model_id)
    if not config:
        return None
    candidates = [
        adapter_path.parent / "flowjudge_training_manifest.json",
        adapter_path / "flowjudge_training_manifest.json",
    ]
    for manifest_path in candidates:
        if not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        base_model = manifest.get("quantized_base_model")
        if isinstance(base_model, str) and base_model:
            return base_model
    base_model = config.get("model")
    return base_model if isinstance(base_model, str) and base_model else None


def _read_adapter_config(model_id: str) -> dict[str, Any]:
    return _read_json_resource(model_id, "adapter_config.json")


def _declared_base_model(model_id: str) -> str | None:
    manifest = _read_json_resource(model_id, "flowjudge_training_manifest.json")
    for key in ("evaluation_base_model", "canonical_base_model", "base_model"):
        value = manifest.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _read_json_resource(model_id: str, filename: str) -> dict[str, Any]:
    local_path = Path(model_id) / filename
    if local_path.exists():
        try:
            value = json.loads(local_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}
    try:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(repo_id=model_id, filename=filename)
        value = json.loads(Path(downloaded).read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _results_table(summaries: dict[str, Any]) -> str:
    lines = [
        "# Base-versus-tuned FlowJudge evaluation",
        "",
        "| Model role | Exact match | Edge F1 | Mean Spec adherence | Mean Robustness | Topical FP rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, summary in summaries.items():
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    _percent(summary["normalized_exact_graph_match_rate"]),
                    _percent(summary["normalized_edge_f1"]),
                    _number(summary["mean_spec_adherence"]),
                    _number(summary["mean_robustness"]),
                    _percent(summary["false_positive_rate_on_topically_related_nonresponses"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"

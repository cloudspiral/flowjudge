#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import modal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_DATA_DIR = PROJECT_ROOT / "data" / "dialam" / "training"
REMOTE_DATA_DIR = Path("/workspace/dialam_data")
PERSISTENT_ROOT = Path("/workspace/persistent")
CHECKPOINT_ROOT = PERSISTENT_ROOT / "checkpoints"
BASE_MODEL = "Qwen/Qwen3-0.6B"
SIZES = (256, 512, 1024, 2048, 4096, 8192, 12288)
DATASET_VERSIONS = ("v1", "v2", "v3", "v4", "v5", "v6")
VALID_SIZES_BY_VERSION = {
    "v1": (256, 512, 1024, 2048),
    "v2": (2048,),
    "v3": (4096,),
    "v4": (8192,),
    "v5": (8192,),
    "v6": (12288,),
}
EVAL_INPUT_FILENAMES = {
    "frozen": "frozen_eval_inputs.jsonl",
    "v3_dev": "v3_dev_eval_inputs.jsonl",
}
V5_EVAL_INPUT_FILENAMES = {
    "frozen": "v5_frozen_pairwise_inputs.jsonl",
    "v3_dev": "v5_dev_pairwise_inputs.jsonl",
}
PAIRWISE_LABELS = ("NONE", "SUPPORT", "ATTACK", "REPHRASE")
MAX_SEQ_LENGTH = 2048
SEED = 20260823

FIXED_CONFIG = {
    "base_model": BASE_MODEL,
    "method": "Unsloth QLoRA",
    "load_in_4bit": True,
    "max_seq_length": MAX_SEQ_LENGTH,
    "epochs": 3.0,
    "learning_rate": 2e-4,
    "per_device_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "effective_batch_size": 8,
    "warmup_ratio": 0.05,
    "lr_scheduler_type": "cosine",
    "optimizer": "adamw_8bit",
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.0,
    "target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "seed": SEED,
    "prompt_masking": True,
    "packing": False,
}
V3_LOSS_CONFIG = {
    "name": "per_example_assistant_token_mean_then_batch_mean",
    "assistant_loss_weight_per_example": 1.0,
}
V4_LOSS_CONFIG = {
    "name": "per_example_assistant_token_mean_then_batch_mean",
    "assistant_loss_weight_per_example": 1.0,
    "class_balance": {
        "NONE": 4096,
        "SUPPORT": 1344,
        "ATTACK": 1344,
        "REPHRASE": 1344,
        "MIXED": 64,
    },
}
V5_LOSS_CONFIG = {
    "name": "paired_sequential_per_example_assistant_token_mean",
    "assistant_loss_weight_per_example": 1.0,
    "per_device_batch_layout": ["POSITIVE", "NONE"],
    "inference": "highest mean allowed-label token log probability",
    "tie_break_order": list(PAIRWISE_LABELS),
}
V6_LOSS_CONFIG = {
    **V5_LOSS_CONFIG,
    "curriculum": [
        {"stage": "vives_warmup", "rows": 4096},
        {"stage": "qt30_target", "rows": 8192},
    ],
    "target_stage_is_exact_v5_corpus": True,
}

image = (
    modal.Image.from_registry("unsloth/unsloth:latest")
    .entrypoint([])
    .add_local_dir(LOCAL_DATA_DIR, str(REMOTE_DATA_DIR), copy=True)
)
volume = modal.Volume.from_name("flowjudge-dialam-qwen3", create_if_missing=True)
app = modal.App("flowjudge-dialam-qwen3-qlora")


def _load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _checkpoint_label(size: int, dataset_version: str) -> str:
    if dataset_version not in DATASET_VERSIONS:
        raise ValueError(f"dataset_version must be one of {DATASET_VERSIONS}")
    if size not in VALID_SIZES_BY_VERSION[dataset_version]:
        raise ValueError(
            f"dataset_version {dataset_version} is registered only for "
            f"n={VALID_SIZES_BY_VERSION[dataset_version]}"
        )
    return f"n{size}" if dataset_version == "v1" else f"{dataset_version}_n{size}"


def _train_filename(size: int, dataset_version: str) -> str:
    _checkpoint_label(size, dataset_version)
    return (
        f"dialam_n{size}.jsonl"
        if dataset_version == "v1"
        else f"dialam_{dataset_version}_n{size}.jsonl"
    )


def _generate(model: object, tokenizer: object, prompt: str, max_new_tokens: int = 256) -> str:
    import torch

    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = output[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def _score_pairwise_labels(
    model: object,
    tokenizer: object,
    prompt: str,
) -> tuple[str, dict[str, float]]:
    import torch

    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
    )["input_ids"]
    label_ids = {
        label: tokenizer(label, add_special_tokens=False)["input_ids"]
        for label in PAIRWISE_LABELS
    }
    if not prompt_ids or any(not tokens for tokens in label_ids.values()):
        raise ValueError("pairwise prompt or label tokenization is empty")
    maximum_length = max(len(prompt_ids) + len(tokens) for tokens in label_ids.values())
    if maximum_length > MAX_SEQ_LENGTH:
        raise ValueError(
            f"pairwise prompt needs {maximum_length} tokens, exceeding {MAX_SEQ_LENGTH}"
        )
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("pairwise scoring requires a tokenizer pad token")

    sequences = [prompt_ids + label_ids[label] for label in PAIRWISE_LABELS]
    input_ids = torch.full(
        (len(sequences), maximum_length),
        pad_id,
        dtype=torch.long,
        device=model.device,
    )
    attention_mask = torch.zeros_like(input_ids)
    for index, sequence in enumerate(sequences):
        input_ids[index, : len(sequence)] = torch.tensor(sequence, device=model.device)
        attention_mask[index, : len(sequence)] = 1
    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.float()
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    scores: dict[str, float] = {}
    for row_index, label in enumerate(PAIRWISE_LABELS):
        tokens = label_ids[label]
        token_scores = [
            log_probs[row_index, len(prompt_ids) + offset - 1, token_id]
            for offset, token_id in enumerate(tokens)
        ]
        scores[label] = float(torch.stack(token_scores).mean().item())
    selected = max(
        PAIRWISE_LABELS,
        key=lambda label: (scores[label], -PAIRWISE_LABELS.index(label)),
    )
    return selected, scores


@app.function(
    image=image,
    gpu="L4",
    timeout=60 * 60 * 2,
    volumes={str(PERSISTENT_ROOT): volume},
)
def train_checkpoint(size: int, dataset_version: str = "v1") -> dict:
    _checkpoint_label(size, dataset_version)

    import hashlib
    import shutil
    from datetime import UTC, datetime

    import torch
    from datasets import Dataset
    from transformers import (
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
        set_seed,
    )
    from unsloth import FastLanguageModel

    set_seed(SEED)
    checkpoint_label = _checkpoint_label(size, dataset_version)
    train_path = REMOTE_DATA_DIR / _train_filename(size, dataset_version)
    rows = _load_rows(train_path)
    if len(rows) != size:
        raise ValueError(f"expected {size} training rows, found {len(rows)}")
    if dataset_version in {"v5", "v6"}:
        for index in range(0, len(rows), 2):
            pair = rows[index : index + 2]
            if len(pair) != 2 or [item["pair_role"] for item in pair] != ["POSITIVE", "NONE"]:
                raise ValueError(
                    f"{dataset_version} rows {index}:{index + 2} are not a positive/NONE batch"
                )
            if len({item["pair_group_id"] for item in pair}) != 1:
                raise ValueError(
                    f"{dataset_version} rows {index}:{index + 2} cross pair groups"
                )
    checkpoint_dir = CHECKPOINT_ROOT / checkpoint_label
    adapter_dir = checkpoint_dir / "adapter"
    if checkpoint_dir.exists():
        raise FileExistsError(
            f"checkpoint already exists at {checkpoint_dir}; preserve it or remove it explicitly"
        )
    checkpoint_dir.mkdir(parents=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=None,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = FastLanguageModel.get_peft_model(
        model,
        r=FIXED_CONFIG["lora_r"],
        target_modules=FIXED_CONFIG["target_modules"],
        lora_alpha=FIXED_CONFIG["lora_alpha"],
        lora_dropout=FIXED_CONFIG["lora_dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
        use_rslora=False,
        loftq_config=None,
    )

    def tokenize_row(row: dict) -> dict[str, list[int]]:
        messages = row["messages"]
        prompt_text = tokenizer.apply_chat_template(
            messages[:1],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        tokenized = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
        )
        labels = list(tokenized["input_ids"])
        labels[: min(len(prompt_ids), len(labels))] = [-100] * min(
            len(prompt_ids), len(labels)
        )
        tokenized["labels"] = labels
        return tokenized

    dataset = Dataset.from_list(rows).map(
        tokenize_row,
        remove_columns=list(rows[0]),
        desc=f"tokenize DialAM n={size}",
    )
    assistant_token_counts = [
        sum(token != -100 for token in labels) for labels in dataset["labels"]
    ]
    if not assistant_token_counts or min(assistant_token_counts) <= 0:
        raise ValueError("at least one training row has no unmasked assistant tokens")
    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir / "trainer"),
        num_train_epochs=FIXED_CONFIG["epochs"],
        per_device_train_batch_size=FIXED_CONFIG["per_device_batch_size"],
        gradient_accumulation_steps=FIXED_CONFIG["gradient_accumulation_steps"],
        learning_rate=FIXED_CONFIG["learning_rate"],
        lr_scheduler_type=FIXED_CONFIG["lr_scheduler_type"],
        warmup_ratio=FIXED_CONFIG["warmup_ratio"],
        logging_steps=5,
        save_strategy="no",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        optim=FIXED_CONFIG["optimizer"],
        report_to="none",
        seed=SEED,
        data_seed=SEED,
        remove_unused_columns=False,
    )
    class PerExampleAssistantLossTrainer(Trainer):
        def compute_loss(
            self,
            model: object,
            inputs: dict,
            return_outputs: bool = False,
            num_items_in_batch: object | None = None,
        ) -> object:
            del num_items_in_batch
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous().float()
            shift_labels = labels[..., 1:].contiguous()
            token_losses = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="none",
            ).view_as(shift_labels)
            mask = shift_labels.ne(-100)
            per_example = (token_losses * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            loss = per_example.mean()
            return (loss, outputs) if return_outputs else loss

    class PairedSequentialLossTrainer(PerExampleAssistantLossTrainer):
        def _get_train_sampler(self, train_dataset: object | None = None) -> object:
            from torch.utils.data import SequentialSampler

            dataset_for_sampler = (
                train_dataset if train_dataset is not None else self.train_dataset
            )
            return SequentialSampler(dataset_for_sampler)

    trainer_class = (
        PairedSequentialLossTrainer
        if dataset_version in {"v5", "v6"}
        else PerExampleAssistantLossTrainer
        if dataset_version in {"v3", "v4"}
        else Trainer
    )
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            label_pad_token_id=-100,
            return_tensors="pt",
        ),
    )
    train_result = trainer.train()
    log_history = list(trainer.state.log_history)
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)

    del trainer, model
    torch.cuda.empty_cache()
    reloaded_model, reloaded_tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_dir),
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=None,
    )
    FastLanguageModel.for_inference(reloaded_model)
    reload_scores = None
    if dataset_version in {"v5", "v6"}:
        reload_probe, reload_scores = _score_pairwise_labels(
            reloaded_model,
            reloaded_tokenizer,
            rows[0]["messages"][0]["content"],
        )
    else:
        reload_probe = _generate(
            reloaded_model,
            reloaded_tokenizer,
            rows[0]["messages"][0]["content"],
        )
    if not reload_probe.strip():
        raise RuntimeError("saved adapter reloaded but produced an empty smoke response")

    manifest = {
        "schema_version": "dialam_modal_qlora_run_v1",
        "completed_at": datetime.now(UTC).isoformat(),
        "size": size,
        "dataset_version": dataset_version,
        "train_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
        "fixed_config": FIXED_CONFIG,
        "loss_config": (
            V3_LOSS_CONFIG
            if dataset_version == "v3"
            else V4_LOSS_CONFIG
            if dataset_version == "v4"
            else V5_LOSS_CONFIG
            if dataset_version == "v5"
            else V6_LOSS_CONFIG
            if dataset_version == "v6"
            else {"name": "assistant_token_mean"}
        ),
        "assistant_token_counts": {
            "minimum": min(assistant_token_counts),
            "mean": sum(assistant_token_counts) / len(assistant_token_counts),
            "maximum": max(assistant_token_counts),
        },
        "gpu": torch.cuda.get_device_name(0),
        "cuda": torch.version.cuda,
        "torch": str(torch.__version__),
        "metrics": train_result.metrics,
        "log_history": log_history,
        "adapter_path": str(adapter_dir),
        "adapter_files": sorted(path.name for path in adapter_dir.iterdir()),
        "reload_verified": True,
        "reload_probe": reload_probe,
        "reload_probe_label_scores": reload_scores,
    }
    (checkpoint_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    shutil.rmtree(checkpoint_dir / "trainer", ignore_errors=True)
    volume.commit()
    # Keep the Modal boundary dependency-free for the local CLI. Some library
    # version objects are string subclasses whose pickle still imports torch.
    return json.loads(json.dumps(manifest))


@app.function(
    image=image,
    gpu="L4",
    timeout=60 * 60,
    volumes={str(PERSISTENT_ROOT): volume},
)
def generate_model_eval(
    target: str,
    size: int = 256,
    dataset_version: str = "v1",
    eval_split: str = "frozen",
) -> list[dict]:
    import torch
    from unsloth import FastLanguageModel

    if target not in {"base", "tuned"}:
        raise ValueError("target must be base or tuned")
    if eval_split not in EVAL_INPUT_FILENAMES:
        raise ValueError(f"eval_split must be one of {tuple(EVAL_INPUT_FILENAMES)}")
    checkpoint_label = _checkpoint_label(size, dataset_version)
    model_name = (
        BASE_MODEL
        if target == "base"
        else str(CHECKPOINT_ROOT / checkpoint_label / "adapter")
    )
    if target == "tuned" and not Path(model_name).is_dir():
        raise FileNotFoundError(f"missing trained adapter: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=None,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    FastLanguageModel.for_inference(model)
    input_filenames = (
        V5_EVAL_INPUT_FILENAMES
        if dataset_version in {"v5", "v6"}
        else EVAL_INPUT_FILENAMES
    )
    rows = _load_rows(REMOTE_DATA_DIR / input_filenames[eval_split])
    predictions = []
    for index, row in enumerate(rows, start=1):
        pairwise_decisions = None
        if dataset_version in {"v5", "v6"}:
            pairwise_decisions = []
            relations = []
            for candidate in row["candidates"]:
                label, label_scores = _score_pairwise_labels(
                    model,
                    tokenizer,
                    candidate["prompt"],
                )
                pairwise_decisions.append(
                    {
                        "target_id": candidate["target_id"],
                        "label": label,
                        "label_scores": label_scores,
                    }
                )
                if label != "NONE":
                    relations.append(
                        {
                            "source": row["source_id"],
                            "target": candidate["target_id"],
                            "type": label,
                        }
                    )
            response = json.dumps({"relations": relations}, separators=(",", ":"))
        else:
            response = _generate(model, tokenizer, row["prompt"])
        predictions.append(
            {
                "example_id": row["example_id"],
                "update_id": row["update_id"],
                "target": target,
                "model": BASE_MODEL,
                "adapter_size": size if target == "tuned" else None,
                "dataset_version": dataset_version if target == "tuned" else None,
                "eval_split": eval_split,
                "raw_response": response,
                "pairwise_decisions": pairwise_decisions,
            }
        )
        print(f"generated {target} {index}/{len(rows)}", flush=True)
    del model
    torch.cuda.empty_cache()
    return predictions


@app.local_entrypoint()
def main(
    action: str,
    size: int = 256,
    target: str = "base",
    dataset_version: str = "v1",
    eval_split: str = "frozen",
    output_path: str = "",
) -> None:
    if action == "train":
        checkpoint_label = _checkpoint_label(size, dataset_version)
        result = train_checkpoint.remote(size, dataset_version)
        default = (
            PROJECT_ROOT
            / "artifacts"
            / "dialam_qlora"
            / checkpoint_label
            / "remote_training_result.json"
        )
    elif action == "evaluate":
        result = generate_model_eval.remote(target, size, dataset_version, eval_split)
        label = "base" if target == "base" else _checkpoint_label(size, dataset_version)
        default = (
            PROJECT_ROOT
            / "results"
            / "dialam_model_eval"
            / (label if eval_split == "frozen" else f"{label}_{eval_split}")
            / "predictions.jsonl"
        )
    else:
        raise ValueError("action must be train or evaluate")

    output = Path(output_path).resolve() if output_path else default
    output.parent.mkdir(parents=True, exist_ok=True)
    if action == "train":
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        with output.open("w", encoding="utf-8") as handle:
            for row in result:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(output)

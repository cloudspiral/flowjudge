#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a FlowJudge QLoRA adapter")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--train-set", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260821)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_training(
        model_id=args.model,
        train_set=args.train_set,
        output_dir=args.output_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        seed=args.seed,
    )


def run_training(
    *,
    model_id: str,
    train_set: Path,
    output_dir: Path,
    epochs: float,
    learning_rate: float,
    max_length: int,
    seed: int,
) -> None:
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            DataCollatorForSeq2Seq,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        raise RuntimeError("install the train dependency group before QLoRA training") from exc

    if not torch.cuda.is_available():
        raise RuntimeError(
            "QLoRA training requires an NVIDIA CUDA GPU; run this script on Modal, RunPod, Colab, or another CUDA host"
        )
    set_seed(seed)
    rows = [json.loads(line) for line in train_set.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError("training set is empty")

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization,
        device_map={"": 0},
        torch_dtype=compute_dtype,
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    )
    model.config.use_cache = False

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
            max_length=max_length,
        )
        labels = list(tokenized["input_ids"])
        prompt_length = min(len(prompt_ids), len(labels))
        labels[:prompt_length] = [-100] * prompt_length
        tokenized["labels"] = labels
        return tokenized

    dataset = Dataset.from_list(rows).map(tokenize_row, remove_columns=list(rows[0]))
    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=1,
        save_strategy="epoch",
        bf16=compute_dtype == torch.bfloat16,
        fp16=compute_dtype == torch.float16,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to="none",
        seed=seed,
        data_seed=seed,
    )
    trainer = Trainer(
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
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "base_model": model_id,
        "train_set": str(train_set),
        "training_examples": len(rows),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "max_length": max_length,
        "seed": seed,
        "qlora": {
            "load_in_4bit": True,
            "quant_type": "nf4",
            "double_quant": True,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
        },
        "metrics": train_result.metrics,
    }
    (output_dir / "flowjudge_training_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

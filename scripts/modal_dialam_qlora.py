#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import modal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from flowjudge.modal_training_resume import (  # noqa: E402
    PROGRESS_FILENAME,
    ResumeMode,
    checkpoint_validation_errors,
    file_sha256,
    object_sha256,
    prepare_resumable_run,
    tree_manifest,
    write_json_atomic,
)


LOCAL_DATA_DIR = PROJECT_ROOT / "data" / "dialam" / "training"
REMOTE_DATA_DIR = Path("/workspace/dialam_data")
PERSISTENT_ROOT = Path("/workspace/persistent")
CHECKPOINT_ROOT = PERSISTENT_ROOT / "checkpoints"
EVALUATION_ROOT = PERSISTENT_ROOT / "evaluations"
BASE_MODEL = "Qwen/Qwen3-0.6B"
SIZES = (252, 256, 512, 1024, 2048, 4096, 8192, 12288)
DATASET_VERSIONS = (
    "v1",
    "v2",
    "v3",
    "v4",
    "v5",
    "v6",
    "v7-smoke",
    "v7",
    "v8-smoke",
    "v8",
    "v9-smoke",
    "v9",
)
VALID_SIZES_BY_VERSION = {
    "v1": (256, 512, 1024, 2048),
    "v2": (2048,),
    "v3": (4096,),
    "v4": (8192,),
    "v5": (8192,),
    "v6": (12288,),
    "v7-smoke": (256,),
    "v7": (8192,),
    "v8-smoke": (252,),
    "v8": (12288,),
    "v9-smoke": (256,),
    "v9": (8192,),
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
PAIRWISE_DATASET_VERSIONS = {
    "v5",
    "v6",
    "v7-smoke",
    "v7",
    "v8-smoke",
    "v8",
    "v9-smoke",
    "v9",
}
MAX_SEQ_LENGTH = 2048
SEED = 20260823
V5_SOURCE_ADAPTER = CHECKPOINT_ROOT / "v5_n8192" / "adapter"
V5_SOURCE_ADAPTER_TREE_SHA256 = (
    "cbd9a2a6cae8ab988d78b7694ac9f9829bf9262bbb7a6fab838894a5080f122a"
)

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
V7_CONFIG = {
    **FIXED_CONFIG,
    "method": "Unsloth QLoRA adapter continuation with reciprocal preference loss",
    "epochs": 1.0,
    "learning_rate": 5e-5,
    "source_adapter": "v5_n8192",
    "source_adapter_tree_sha256": V5_SOURCE_ADAPTER_TREE_SHA256,
    "save_strategy": "steps",
    "save_total_limit": 2,
    "save_steps": {"v7-smoke": 8, "v7": 100},
    "ignore_data_skip": False,
}
V7_LOSS_CONFIG = {
    "name": "reciprocal_candidate_preference_logistic_with_chosen_nll_anchor",
    "sequence_score": "mean label-token log probability",
    "preference_loss": "softplus(-(chosen_score-rejected_score))",
    "preference_beta": 1.0,
    "chosen_label_nll_weight": 0.1,
    "positive_candidate_preference": "gold relation > NONE",
    "hard_negative_candidate_preference": "NONE > paired gold relation",
}
V8_CONFIG = {
    **FIXED_CONFIG,
    "method": "Unsloth QLoRA adapter continuation with restricted-label cross-entropy",
    "epochs": 1.0,
    "learning_rate": 2e-5,
    "source_adapter": "v5_n8192",
    "source_adapter_tree_sha256": V5_SOURCE_ADAPTER_TREE_SHA256,
    "save_strategy": "steps",
    "save_total_limit": 2,
    "save_steps": {"v8-smoke": 8, "v8": 100},
    "ignore_data_skip": False,
}
V8_LOSS_CONFIG = {
    "name": "restricted_four_label_score_cross_entropy",
    "labels": list(PAIRWISE_LABELS),
    "sequence_score": "mean label-token log probability",
    "loss": "cross_entropy(stack(four_label_scores), gold_label_index)",
    "generative_token_nll": False,
    "preference_loss": False,
    "class_weights": False,
    "label_smoothing": False,
    "training_mix": {
        "POSITIVE": 4096,
        "NONE": 8192,
    },
}
V9_CONFIG = {
    **FIXED_CONFIG,
    "method": "Unsloth QLoRA selected-model hard-negative corrective continuation",
    "epochs": 1.0,
    "learning_rate": 5e-6,
    "source_adapter": "v5_n8192",
    "source_adapter_tree_sha256": V5_SOURCE_ADAPTER_TREE_SHA256,
    "save_strategy": "steps",
    "save_total_limit": 2,
    "save_steps": {"v9-smoke": 8, "v9": 100},
    "ignore_data_skip": False,
}
V9_LOSS_CONFIG = {
    "name": "restricted_four_label_score_cross_entropy",
    "labels": list(PAIRWISE_LABELS),
    "sequence_score": "mean label-token log probability",
    "loss": "cross_entropy(stack(four_label_scores), gold_label_index)",
    "generative_token_nll": False,
    "preference_loss": False,
    "class_weights": False,
    "label_smoothing": False,
    "training_mix": {
        "POSITIVE_REHEARSAL": 4096,
        "MINED_NONE": 4096,
    },
}
V9_MINING_INPUT_FILENAME = "v9_mining_candidates_n12288.jsonl"
V9_MINING_SCORE_FILENAME = "v9_mining_scores_n12288.jsonl"
V9_MINING_CANDIDATE_COUNT = 12288
V9_MINING_CHUNK_SIZE = 128
V9_MINING_BATCH_SIZE = 4
V9_MINING_ROOT = EVALUATION_ROOT / "v9_v5_selected_model_mining"

image = (
    modal.Image.from_registry("unsloth/unsloth:latest")
    .entrypoint([])
    .add_local_dir(LOCAL_DATA_DIR, str(REMOTE_DATA_DIR), copy=True)
    .add_local_dir(PROJECT_ROOT / "src", "/workspace/project_src", copy=True)
    .env({"PYTHONPATH": "/workspace/project_src"})
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
    if dataset_version == "v1":
        return f"n{size}"
    if dataset_version == "v7-smoke":
        return f"v7_resume_smoke_n{size}"
    if dataset_version == "v8-smoke":
        return f"v8_resume_smoke_n{size}"
    if dataset_version == "v9-smoke":
        return f"v9_resume_smoke_n{size}"
    return f"{dataset_version}_n{size}"


def _train_filename(size: int, dataset_version: str) -> str:
    _checkpoint_label(size, dataset_version)
    if dataset_version == "v7-smoke":
        return f"dialam_v7_resume_smoke_n{size}.jsonl"
    if dataset_version == "v8-smoke":
        return f"dialam_v8_resume_smoke_n{size}.jsonl"
    if dataset_version == "v9-smoke":
        return f"dialam_v9_resume_smoke_n{size}.jsonl"
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


def _score_pairwise_labels_batch(
    model: object,
    tokenizer: object,
    prompts: list[str],
) -> list[tuple[str, dict[str, float]]]:
    """Score a small prompt batch while materializing only trailing logits."""

    import torch

    if not prompts:
        return []
    label_ids = {
        label: tokenizer(label, add_special_tokens=False)["input_ids"]
        for label in PAIRWISE_LABELS
    }
    if any(not tokens for tokens in label_ids.values()):
        raise ValueError("v9 mining label tokenization is empty")
    prompt_ids = []
    for prompt in prompts:
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        tokens = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        if not tokens:
            raise ValueError("v9 mining prompt tokenization is empty")
        prompt_ids.append(tokens)

    records: list[tuple[int, str, list[int], int]] = []
    sequences: list[list[int]] = []
    for prompt_index, tokens in enumerate(prompt_ids):
        for label in PAIRWISE_LABELS:
            label_tokens = label_ids[label]
            sequence = tokens + label_tokens
            if len(sequence) > MAX_SEQ_LENGTH:
                raise ValueError(
                    f"v9 mining prompt needs {len(sequence)} tokens, exceeding "
                    f"{MAX_SEQ_LENGTH}"
                )
            records.append((prompt_index, label, label_tokens, len(sequence)))
            sequences.append(sequence)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("v9 mining scoring requires a tokenizer pad token")
    maximum_length = max(len(sequence) for sequence in sequences)
    input_ids = torch.full(
        (len(sequences), maximum_length),
        pad_id,
        dtype=torch.long,
        device=model.device,
    )
    attention_mask = torch.zeros_like(input_ids)
    for row_index, sequence in enumerate(sequences):
        offset = maximum_length - len(sequence)
        input_ids[row_index, offset:] = torch.tensor(sequence, device=model.device)
        attention_mask[row_index, offset:] = 1

    maximum_label_length = max(len(tokens) for tokens in label_ids.values())
    logits_to_keep = maximum_label_length + 1
    with torch.inference_mode():
        try:
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                logits_to_keep=logits_to_keep,
            ).logits
        except TypeError as error:
            raise RuntimeError(
                "Qwen3 mining requires trailing-logit selection support"
            ) from error
    if logits.shape[1] != logits_to_keep:
        raise RuntimeError("Qwen3 returned an unexpected mining-logit window")

    by_prompt: list[dict[str, float]] = [dict() for _ in prompts]
    for row_index, (prompt_index, label, tokens, _) in enumerate(records):
        first_prediction = logits_to_keep - len(tokens) - 1
        token_scores = []
        for token_offset, token_id in enumerate(tokens):
            token_logits = logits[row_index, first_prediction + token_offset].float()
            token_scores.append(token_logits[token_id] - torch.logsumexp(token_logits, dim=-1))
        by_prompt[prompt_index][label] = float(torch.stack(token_scores).mean().item())

    output = []
    for scores in by_prompt:
        selected = max(
            PAIRWISE_LABELS,
            key=lambda label: (scores[label], -PAIRWISE_LABELS.index(label)),
        )
        output.append((selected, scores))
    return output


@app.function(
    image=image,
    gpu="L4",
    timeout=60 * 60 * 4,
    volumes={str(PERSISTENT_ROOT): volume},
)
def mine_v9_hard_negatives(stop_after_chunks: int = 0) -> dict:
    """Score and persist fixed V9 training candidates one complete chunk at a time."""

    from datetime import UTC, datetime

    import torch
    from unsloth import FastLanguageModel

    if stop_after_chunks < 0:
        raise ValueError("stop_after_chunks must be nonnegative")
    volume.reload()
    input_path = REMOTE_DATA_DIR / V9_MINING_INPUT_FILENAME
    rows = _load_rows(input_path)
    if len(rows) != V9_MINING_CANDIDATE_COUNT:
        raise ValueError(
            f"expected {V9_MINING_CANDIDATE_COUNT} v9 mining rows, found {len(rows)}"
        )
    candidate_ids = [row["mining_candidate_id"] for row in rows]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("v9 mining candidate IDs are not unique")
    if any(row.get("gold_label") != "NONE" for row in rows):
        raise ValueError("v9 mining inputs contain a non-NONE candidate")
    source_hash, source_files = tree_manifest(V5_SOURCE_ADAPTER)
    if source_hash != V5_SOURCE_ADAPTER_TREE_SHA256:
        raise RuntimeError("remote v5 source adapter differs from the frozen checkpoint")

    expected_identity = {
        "schema_version": "dialam_modal_resumable_mining_identity_v1",
        "implementation_version": "dialam-v9-v5-mining-batched-trailing-logits-v1",
        "input_sha256": file_sha256(input_path),
        "candidate_ids": candidate_ids,
        "source_adapter_path": str(V5_SOURCE_ADAPTER),
        "source_adapter_tree_sha256": source_hash,
        "labels": list(PAIRWISE_LABELS),
        "scoring": "mean allowed-label token log probability",
        "selection_signal": "max positive score minus NONE score",
        "chunk_size": V9_MINING_CHUNK_SIZE,
        "batch_size": V9_MINING_BATCH_SIZE,
        "max_seq_length": MAX_SEQ_LENGTH,
    }
    identity_hash = object_sha256(expected_identity)
    run_dir = V9_MINING_ROOT
    identity_path = run_dir / "run_identity.json"
    progress_path = run_dir / "progress.json"
    completed_path = run_dir / "completed.json"
    chunk_dir = run_dir / "chunks"
    if run_dir.exists():
        if not identity_path.is_file():
            raise RuntimeError("existing v9 mining run has no immutable identity")
        observed = json.loads(identity_path.read_text(encoding="utf-8"))
        observed_hash = observed.pop("identity_sha256", None)
        if observed_hash != object_sha256(observed):
            raise RuntimeError("stored v9 mining identity hash is invalid")
        if observed != expected_identity or observed_hash != identity_hash:
            raise RuntimeError("v9 mining identity mismatch")
    else:
        run_dir.mkdir(parents=True)
        chunk_dir.mkdir()
        write_json_atomic(
            identity_path,
            {**expected_identity, "identity_sha256": identity_hash},
        )
        write_json_atomic(
            progress_path,
            {
                "schema_version": "dialam_modal_resumable_mining_progress_v1",
                "identity_sha256": identity_hash,
                "resume_events": [],
                "completed_chunks": 0,
                "completed_candidates": 0,
            },
        )
        volume.commit()
    chunk_dir.mkdir(exist_ok=True)

    expected_chunks = [
        rows[index : index + V9_MINING_CHUNK_SIZE]
        for index in range(0, len(rows), V9_MINING_CHUNK_SIZE)
    ]

    def load_chunks() -> tuple[list[list[dict] | None], int, int]:
        persisted: list[list[dict] | None] = []
        completed_chunks = 0
        completed_candidates = 0
        for chunk_index, expected in enumerate(expected_chunks):
            path = chunk_dir / f"{chunk_index:04d}.json"
            if not path.is_file():
                persisted.append(None)
                continue
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, list) or [
                item.get("mining_candidate_id") for item in value
            ] != [item["mining_candidate_id"] for item in expected]:
                raise RuntimeError(f"persisted v9 mining chunk {chunk_index} is invalid")
            if any(
                item.get("adapter_tree_sha256") != source_hash for item in value
            ):
                raise RuntimeError(
                    f"persisted v9 mining chunk {chunk_index} has the wrong adapter"
                )
            persisted.append(value)
            completed_chunks += 1
            completed_candidates += len(value)
        return persisted, completed_chunks, completed_candidates

    persisted, resumed_chunks, resumed_candidates = load_chunks()
    if completed_path.is_file():
        completed = json.loads(completed_path.read_text(encoding="utf-8"))
        scores = [item for chunk in persisted if chunk is not None for item in chunk]
        if len(scores) != len(rows):
            raise RuntimeError("completed v9 mining run is missing chunks")
        if completed.get("scores_sha256") != object_sha256(scores):
            raise RuntimeError("completed v9 mining score hash is invalid")
        return {
            "scores": scores,
            "resume": {
                "idempotent_reuse": True,
                "resumed_completed_chunks": len(expected_chunks),
                "newly_completed_chunks": 0,
                "total_chunks": len(expected_chunks),
                "resumed_completed_candidates": len(scores),
                "newly_completed_candidates": 0,
                "total_candidates": len(scores),
                "identity_sha256": identity_hash,
                "adapter_tree_sha256": source_hash,
                "scores_sha256": completed["scores_sha256"],
            },
        }

    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress["resume_events"].append(
        {
            "started_at": datetime.now(UTC).isoformat(),
            "resumed_completed_chunks": resumed_chunks,
            "resumed_completed_candidates": resumed_candidates,
            "missing_chunks": len(expected_chunks) - resumed_chunks,
        }
    )
    write_json_atomic(progress_path, progress)
    volume.commit()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(V5_SOURCE_ADAPTER),
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=None,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    FastLanguageModel.for_inference(model)
    newly_completed_chunks = 0
    newly_completed_candidates = 0
    positive_labels = ("SUPPORT", "ATTACK", "REPHRASE")
    for chunk_index, expected in enumerate(expected_chunks):
        if persisted[chunk_index] is not None:
            print(
                f"reused v9 mining chunk {chunk_index + 1}/{len(expected_chunks)}",
                flush=True,
            )
            continue
        chunk_scores: list[dict] = []
        for batch_start in range(0, len(expected), V9_MINING_BATCH_SIZE):
            batch = expected[batch_start : batch_start + V9_MINING_BATCH_SIZE]
            scored = _score_pairwise_labels_batch(
                model,
                tokenizer,
                [item["prompt"] for item in batch],
            )
            for candidate, (selected_label, label_scores) in zip(
                batch,
                scored,
                strict=True,
            ):
                winning_positive = max(
                    positive_labels,
                    key=lambda label: (
                        label_scores[label],
                        -positive_labels.index(label),
                    ),
                )
                chunk_scores.append(
                    {
                        "schema_version": "dialam_selected_model_mining_score_v9",
                        "mining_candidate_id": candidate["mining_candidate_id"],
                        "decision_key": candidate["decision_key"],
                        "selected_label": selected_label,
                        "winning_positive_label": winning_positive,
                        "label_scores": label_scores,
                        "model_hardness": (
                            label_scores[winning_positive] - label_scores["NONE"]
                        ),
                        "support_evidence": (
                            label_scores["SUPPORT"] - label_scores["NONE"]
                        ),
                        "adapter_tree_sha256": source_hash,
                    }
                )
        write_json_atomic(chunk_dir / f"{chunk_index:04d}.json", chunk_scores)
        persisted[chunk_index] = chunk_scores
        newly_completed_chunks += 1
        newly_completed_candidates += len(chunk_scores)
        progress.update(
            {
                "completed_chunks": resumed_chunks + newly_completed_chunks,
                "completed_candidates": (
                    resumed_candidates + newly_completed_candidates
                ),
                "latest_completed_chunk": chunk_index,
                "latest_completed_candidate_id": chunk_scores[-1][
                    "mining_candidate_id"
                ],
                "committed_at": datetime.now(UTC).isoformat(),
            }
        )
        write_json_atomic(progress_path, progress)
        volume.commit()
        print(
            f"scored and committed v9 mining chunk "
            f"{chunk_index + 1}/{len(expected_chunks)}",
            flush=True,
        )
        if stop_after_chunks and newly_completed_chunks >= stop_after_chunks:
            write_json_atomic(
                run_dir / "intentional_interrupt.json",
                {
                    "schema_version": "dialam_v9_mining_intentional_interrupt_v1",
                    "committed_chunk": chunk_index,
                    "committed_candidates": (
                        resumed_candidates + newly_completed_candidates
                    ),
                    "created_at": datetime.now(UTC).isoformat(),
                },
            )
            volume.commit()
            raise RuntimeError(
                "INTENTIONAL_V9_MINING_INTERRUPTION_AFTER_COMMITTED_CHUNK"
            )

    del model
    torch.cuda.empty_cache()
    scores = [item for chunk in persisted if chunk is not None for item in chunk]
    if len(scores) != len(rows):
        raise RuntimeError("v9 mining did not cover every candidate")
    scores_hash = object_sha256(scores)
    write_json_atomic(
        completed_path,
        {
            "schema_version": "dialam_modal_resumable_mining_complete_v1",
            "identity_sha256": identity_hash,
            "candidate_count": len(scores),
            "chunk_count": len(expected_chunks),
            "scores_sha256": scores_hash,
            "adapter_tree_sha256": source_hash,
            "adapter_files": source_files,
            "completed_at": datetime.now(UTC).isoformat(),
        },
    )
    volume.commit()
    return {
        "scores": scores,
        "resume": {
            "idempotent_reuse": False,
            "resumed_completed_chunks": resumed_chunks,
            "newly_completed_chunks": newly_completed_chunks,
            "total_chunks": len(expected_chunks),
            "resumed_completed_candidates": resumed_candidates,
            "newly_completed_candidates": newly_completed_candidates,
            "total_candidates": len(scores),
            "identity_sha256": identity_hash,
            "adapter_tree_sha256": source_hash,
            "scores_sha256": scores_hash,
        },
    }


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
        if dataset_version in PAIRWISE_DATASET_VERSIONS
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
    timeout=60 * 60 * 4,
    volumes={str(PERSISTENT_ROOT): volume},
)
def train_preference_checkpoint(
    size: int,
    dataset_version: str,
    resume_mode: ResumeMode = "auto",
    interrupt_after_steps: int = 0,
) -> dict:
    if dataset_version not in {"v7-smoke", "v7"}:
        raise ValueError("preference training is registered only for v7-smoke or v7")
    _checkpoint_label(size, dataset_version)
    if interrupt_after_steps < 0:
        raise ValueError("interrupt_after_steps must be nonnegative")

    from datetime import UTC, datetime

    volume.reload()
    checkpoint_label = _checkpoint_label(size, dataset_version)
    train_path = REMOTE_DATA_DIR / _train_filename(size, dataset_version)
    rows = _load_rows(train_path)
    if len(rows) != size:
        raise ValueError(f"expected {size} v7 preference rows, found {len(rows)}")
    for index in range(0, len(rows), 2):
        positive, negative = rows[index : index + 2]
        if [positive["pair_role"], negative["pair_role"]] != ["POSITIVE", "NONE"]:
            raise ValueError(f"v7 rows {index}:{index + 2} are not reciprocal roles")
        if positive["pair_group_id"] != negative["pair_group_id"]:
            raise ValueError(f"v7 rows {index}:{index + 2} cross pair groups")
        if (positive["chosen_label"], positive["rejected_label"]) != (
            positive["positive_label"],
            "NONE",
        ):
            raise ValueError("positive candidate preference is not relation > NONE")
        if (negative["chosen_label"], negative["rejected_label"]) != (
            "NONE",
            negative["positive_label"],
        ):
            raise ValueError("hard-negative preference is not NONE > relation")

    source_adapter_hash, source_adapter_files = tree_manifest(V5_SOURCE_ADAPTER)
    if source_adapter_hash != V5_SOURCE_ADAPTER_TREE_SHA256:
        raise RuntimeError(
            "remote v5 source adapter differs from the frozen selected checkpoint"
        )
    save_steps = int(V7_CONFIG["save_steps"][dataset_version])
    expected_identity = {
        "schema_version": "dialam_modal_resumable_run_identity_v1",
        "implementation_version": "dialam-v7-preference-resume-v1",
        "checkpoint_label": checkpoint_label,
        "dataset_version": dataset_version,
        "size": size,
        "train_sha256": file_sha256(train_path),
        "source_adapter_path": str(V5_SOURCE_ADAPTER),
        "source_adapter_tree_sha256": source_adapter_hash,
        "fixed_config": V7_CONFIG,
        "loss_config": V7_LOSS_CONFIG,
        "seed": SEED,
        "save_steps": save_steps,
    }
    run_dir = CHECKPOINT_ROOT / checkpoint_label
    decision = prepare_resumable_run(
        run_dir,
        expected_identity,
        resume_mode=resume_mode,
    )
    run_identity_sha256 = object_sha256(expected_identity)
    if decision.action == "complete":
        completed = json.loads(
            (run_dir / "training_manifest.json").read_text(encoding="utf-8")
        )
        completed["idempotent_reuse"] = True
        return completed

    import torch
    from datasets import Dataset
    from unsloth import FastLanguageModel
    from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed

    set_seed(SEED)

    progress_path = run_dir / PROGRESS_FILENAME
    progress = (
        json.loads(progress_path.read_text(encoding="utf-8"))
        if progress_path.is_file()
        else {
            "schema_version": "dialam_modal_training_progress_v1",
            "run_identity_sha256": run_identity_sha256,
            "resume_events": [],
        }
    )
    progress["resume_events"].append(
        {
            "started_at": datetime.now(UTC).isoformat(),
            "action": decision.action,
            "from_global_step": decision.global_step,
            "checkpoint": str(decision.checkpoint) if decision.checkpoint else None,
            "ignored_incomplete_checkpoints": list(
                decision.ignored_incomplete_checkpoints
            ),
        }
    )
    progress.update(
        {
            "status": "initializing",
            "latest_committed_global_step": decision.global_step,
        }
    )
    write_json_atomic(progress_path, progress)
    volume.commit()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(V5_SOURCE_ADAPTER),
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=None,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model.config.use_cache = False
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    if trainable_parameters <= 0:
        raise RuntimeError("continued v5 adapter loaded with no trainable parameters")

    def tokenize_preference(row: dict) -> dict[str, list[int]]:
        prompt_text = tokenizer.apply_chat_template(
            row["messages"][:1],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        chosen_ids = tokenizer(row["chosen_label"], add_special_tokens=False)[
            "input_ids"
        ]
        rejected_ids = tokenizer(row["rejected_label"], add_special_tokens=False)[
            "input_ids"
        ]
        if not prompt_ids or not chosen_ids or not rejected_ids:
            raise ValueError("v7 prompt or preference label tokenization is empty")
        if len(prompt_ids) + max(len(chosen_ids), len(rejected_ids)) > MAX_SEQ_LENGTH:
            raise ValueError("v7 preference sequence exceeds the registered max length")

        def sequence(label_ids: list[int]) -> tuple[list[int], list[int], list[int]]:
            input_ids = prompt_ids + label_ids
            return (
                input_ids,
                [1] * len(input_ids),
                [-100] * len(prompt_ids) + label_ids,
            )

        chosen_input, chosen_attention, chosen_labels = sequence(chosen_ids)
        rejected_input, rejected_attention, rejected_labels = sequence(rejected_ids)
        return {
            "chosen_input_ids": chosen_input,
            "chosen_attention_mask": chosen_attention,
            "chosen_labels": chosen_labels,
            "rejected_input_ids": rejected_input,
            "rejected_attention_mask": rejected_attention,
            "rejected_labels": rejected_labels,
        }

    dataset = Dataset.from_list(rows).map(
        tokenize_preference,
        remove_columns=list(rows[0]),
        desc=f"tokenize DialAM {dataset_version} n={size}",
    )
    chosen_token_counts = [
        sum(token != -100 for token in labels) for labels in dataset["chosen_labels"]
    ]
    rejected_token_counts = [
        sum(token != -100 for token in labels)
        for labels in dataset["rejected_labels"]
    ]
    if min(chosen_token_counts + rejected_token_counts) <= 0:
        raise ValueError("at least one v7 preference has no scored label token")

    class PreferenceCollator:
        def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
            count = len(features)
            sequences = [
                {
                    "input_ids": feature[f"{prefix}_input_ids"],
                    "attention_mask": feature[f"{prefix}_attention_mask"],
                }
                for prefix in ("chosen", "rejected")
                for feature in features
            ]
            labels = [
                feature[f"{prefix}_labels"]
                for prefix in ("chosen", "rejected")
                for feature in features
            ]
            padded = tokenizer.pad(sequences, padding=True, return_tensors="pt")
            padded_labels = torch.full_like(padded["input_ids"], -100)
            for row_index, row_labels in enumerate(labels):
                padded_labels[row_index, : len(row_labels)] = torch.tensor(
                    row_labels, dtype=torch.long
                )
            return {
                "chosen_input_ids": padded["input_ids"][:count],
                "chosen_attention_mask": padded["attention_mask"][:count],
                "chosen_labels": padded_labels[:count],
                "rejected_input_ids": padded["input_ids"][count:],
                "rejected_attention_mask": padded["attention_mask"][count:],
                "rejected_labels": padded_labels[count:],
            }

    class PreferenceTrainer(Trainer):
        def compute_loss(
            self,
            model: object,
            inputs: dict,
            return_outputs: bool = False,
            num_items_in_batch: object | None = None,
        ) -> object:
            del num_items_in_batch
            chosen_input_ids = inputs.pop("chosen_input_ids")
            chosen_attention_mask = inputs.pop("chosen_attention_mask")
            chosen_labels = inputs.pop("chosen_labels")
            rejected_input_ids = inputs.pop("rejected_input_ids")
            rejected_attention_mask = inputs.pop("rejected_attention_mask")
            rejected_labels = inputs.pop("rejected_labels")
            batch_size = chosen_input_ids.shape[0]
            outputs = model(
                input_ids=torch.cat([chosen_input_ids, rejected_input_ids], dim=0),
                attention_mask=torch.cat(
                    [chosen_attention_mask, rejected_attention_mask], dim=0
                ),
            )
            labels = torch.cat([chosen_labels, rejected_labels], dim=0)
            shift_logits = outputs.logits[..., :-1, :].contiguous().float()
            shift_labels = labels[..., 1:].contiguous()
            mask = shift_labels.ne(-100)
            safe_labels = shift_labels.masked_fill(~mask, 0)
            token_log_probs = torch.nn.functional.log_softmax(
                shift_logits, dim=-1
            ).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
            sequence_scores = (token_log_probs * mask).sum(dim=1) / mask.sum(
                dim=1
            ).clamp_min(1)
            chosen_scores = sequence_scores[:batch_size]
            rejected_scores = sequence_scores[batch_size:]
            preference_loss = torch.nn.functional.softplus(
                -(chosen_scores - rejected_scores)
            ).mean()
            chosen_nll = -chosen_scores.mean()
            loss = preference_loss + (
                float(V7_LOSS_CONFIG["chosen_label_nll_weight"]) * chosen_nll
            )
            return (loss, outputs) if return_outputs else loss

    class PersistentCheckpointCallback(TrainerCallback):
        def on_save(self, args: object, state: object, control: object, **kwargs: object) -> object:
            del kwargs
            checkpoint = Path(str(args.output_dir)) / f"checkpoint-{state.global_step}"
            errors = checkpoint_validation_errors(checkpoint)
            if errors:
                raise RuntimeError(
                    f"refusing to commit incomplete checkpoint {checkpoint}: {errors}"
                )
            current = json.loads(progress_path.read_text(encoding="utf-8"))
            current.update(
                {
                    "status": "checkpointed",
                    "latest_committed_global_step": state.global_step,
                    "latest_checkpoint": str(checkpoint),
                    "latest_checkpoint_files": sorted(
                        item.name for item in checkpoint.iterdir() if item.is_file()
                    ),
                    "committed_at": datetime.now(UTC).isoformat(),
                }
            )
            write_json_atomic(progress_path, current)
            volume.commit()
            print(
                f"committed resumable checkpoint at global step {state.global_step}",
                flush=True,
            )
            if interrupt_after_steps and state.global_step >= interrupt_after_steps:
                write_json_atomic(
                    run_dir / "intentional_interrupt.json",
                    {
                        "schema_version": "dialam_intentional_interrupt_v1",
                        "committed_global_step": state.global_step,
                        "checkpoint": str(checkpoint),
                        "created_at": datetime.now(UTC).isoformat(),
                    },
                )
                volume.commit()
                raise RuntimeError(
                    "INTENTIONAL_RESUME_SMOKE_INTERRUPTION_AFTER_COMMITTED_CHECKPOINT"
                )
            return control

    training_args = TrainingArguments(
        output_dir=str(run_dir / "trainer"),
        num_train_epochs=V7_CONFIG["epochs"],
        per_device_train_batch_size=V7_CONFIG["per_device_batch_size"],
        gradient_accumulation_steps=V7_CONFIG["gradient_accumulation_steps"],
        learning_rate=V7_CONFIG["learning_rate"],
        lr_scheduler_type=V7_CONFIG["lr_scheduler_type"],
        warmup_ratio=V7_CONFIG["warmup_ratio"],
        logging_steps=5,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=V7_CONFIG["save_total_limit"],
        save_safetensors=True,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        optim=V7_CONFIG["optimizer"],
        report_to="none",
        seed=SEED,
        data_seed=SEED,
        remove_unused_columns=False,
        ignore_data_skip=False,
        gradient_checkpointing=True,
    )
    trainer = PreferenceTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=PreferenceCollator(),
        callbacks=[PersistentCheckpointCallback()],
    )
    train_result = trainer.train(
        resume_from_checkpoint=(
            str(decision.checkpoint) if decision.action == "resume" else None
        )
    )
    global_step = int(trainer.state.global_step)
    log_history = list(trainer.state.log_history)
    adapter_dir = run_dir / "adapter"
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
    reload_probe, reload_scores = _score_pairwise_labels(
        reloaded_model,
        reloaded_tokenizer,
        rows[0]["messages"][0]["content"],
    )
    if not reload_probe.strip():
        raise RuntimeError("saved v7 adapter reloaded but produced an empty label")
    adapter_hash, adapter_files = tree_manifest(adapter_dir)
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress.update(
        {
            "status": "complete",
            "completed_global_step": global_step,
            "completed_at": datetime.now(UTC).isoformat(),
        }
    )
    write_json_atomic(progress_path, progress)
    manifest = {
        "schema_version": "dialam_modal_preference_qlora_run_v1",
        "completed_at": datetime.now(UTC).isoformat(),
        "size": size,
        "dataset_version": dataset_version,
        "train_sha256": file_sha256(train_path),
        "run_identity_sha256": run_identity_sha256,
        "fixed_config": V7_CONFIG,
        "loss_config": V7_LOSS_CONFIG,
        "source_adapter": {
            "path": str(V5_SOURCE_ADAPTER),
            "tree_sha256": source_adapter_hash,
            "files": source_adapter_files,
        },
        "resumability": {
            "resume_mode": resume_mode,
            "initial_action": decision.action,
            "resumed_from_step": decision.global_step,
            "resume_events": progress["resume_events"],
            "save_steps": save_steps,
            "save_total_limit": V7_CONFIG["save_total_limit"],
            "full_trainer_state": True,
            "volume_commit_on_save": True,
            "trainer_state_preserved_after_completion": True,
        },
        "global_step": global_step,
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "label_token_counts": {
            "chosen_minimum": min(chosen_token_counts),
            "chosen_maximum": max(chosen_token_counts),
            "rejected_minimum": min(rejected_token_counts),
            "rejected_maximum": max(rejected_token_counts),
        },
        "gpu": torch.cuda.get_device_name(0),
        "cuda": torch.version.cuda,
        "torch": str(torch.__version__),
        "metrics": train_result.metrics,
        "log_history": log_history,
        "adapter_path": str(adapter_dir),
        "adapter_tree_sha256": adapter_hash,
        "adapter_files": adapter_files,
        "reload_verified": True,
        "reload_probe": reload_probe,
        "reload_probe_label_scores": reload_scores,
        "idempotent_reuse": False,
    }
    write_json_atomic(run_dir / "training_manifest.json", manifest)
    volume.commit()
    return json.loads(json.dumps(manifest))


@app.function(
    image=image,
    gpu="L4",
    timeout=60 * 60 * 4,
    volumes={str(PERSISTENT_ROOT): volume},
)
def train_listwise_checkpoint(
    size: int,
    dataset_version: str,
    resume_mode: ResumeMode = "auto",
    interrupt_after_steps: int = 0,
) -> dict:
    if dataset_version not in {"v8-smoke", "v8", "v9-smoke", "v9"}:
        raise ValueError("restricted-label training is registered only for v8 or v9")
    _checkpoint_label(size, dataset_version)
    if interrupt_after_steps < 0:
        raise ValueError("interrupt_after_steps must be nonnegative")
    is_v9 = dataset_version in {"v9-smoke", "v9"}
    config = V9_CONFIG if is_v9 else V8_CONFIG
    loss_config = V9_LOSS_CONFIG if is_v9 else V8_LOSS_CONFIG

    from datetime import UTC, datetime

    volume.reload()
    checkpoint_label = _checkpoint_label(size, dataset_version)
    train_path = REMOTE_DATA_DIR / _train_filename(size, dataset_version)
    rows = _load_rows(train_path)
    if len(rows) != size:
        raise ValueError(f"expected {size} restricted-label rows, found {len(rows)}")
    if is_v9:
        from collections import Counter

        expected_labels = (
            {"NONE": 128, "ATTACK": 43, "REPHRASE": 43, "SUPPORT": 42}
            if dataset_version == "v9-smoke"
            else {"NONE": 4096, "ATTACK": 1366, "REPHRASE": 1365, "SUPPORT": 1365}
        )
        if Counter(row.get("label") for row in rows) != Counter(expected_labels):
            raise ValueError("v9 row label mix differs from its registered corpus")
        expected_roles = (
            {"POSITIVE_REHEARSAL": 128, "MINED_NONE": 128}
            if dataset_version == "v9-smoke"
            else {"POSITIVE_REHEARSAL": 4096, "MINED_NONE": 4096}
        )
        if Counter(row.get("row_role") for row in rows) != Counter(expected_roles):
            raise ValueError("v9 row-role mix differs from its registered corpus")
        if len({row.get("training_row_id") for row in rows}) != len(rows):
            raise ValueError("v9 training row IDs are not unique")
    else:
        for index in range(0, len(rows), 3):
            triple = rows[index : index + 3]
            if len(triple) != 3:
                raise ValueError("v8 listwise data ends with an incomplete triple")
            if [row["group_position"] for row in triple] != [0, 1, 2]:
                raise ValueError(f"v8 rows {index}:{index + 3} are not ordered triples")
            if len({row["group_id"] for row in triple}) != 1:
                raise ValueError(f"v8 rows {index}:{index + 3} cross groups")
            if len({row["source_example_id"] for row in triple}) != 1:
                raise ValueError(f"v8 rows {index}:{index + 3} cross source blocks")
            if len({row["candidate_target_id"] for row in triple}) != 3:
                raise ValueError(f"v8 rows {index}:{index + 3} repeat a candidate")
            if [row["label"] for row in triple[1:]] != ["NONE", "NONE"]:
                raise ValueError(f"v8 rows {index}:{index + 3} lack two NONE labels")

    source_adapter_hash, source_adapter_files = tree_manifest(V5_SOURCE_ADAPTER)
    if source_adapter_hash != V5_SOURCE_ADAPTER_TREE_SHA256:
        raise RuntimeError(
            "remote v5 source adapter differs from the frozen selected checkpoint"
        )
    save_steps = int(config["save_steps"][dataset_version])
    expected_identity = {
        "schema_version": "dialam_modal_resumable_run_identity_v1",
        "implementation_version": (
            "dialam-v9-model-error-corrective-resume-v1"
            if is_v9
            else "dialam-v8-listwise-resume-v1"
        ),
        "checkpoint_label": checkpoint_label,
        "dataset_version": dataset_version,
        "size": size,
        "train_sha256": file_sha256(train_path),
        "source_adapter_path": str(V5_SOURCE_ADAPTER),
        "source_adapter_tree_sha256": source_adapter_hash,
        "fixed_config": config,
        "loss_config": loss_config,
        "seed": SEED,
        "save_steps": save_steps,
    }
    run_dir = CHECKPOINT_ROOT / checkpoint_label
    decision = prepare_resumable_run(
        run_dir,
        expected_identity,
        resume_mode=resume_mode,
    )
    run_identity_sha256 = object_sha256(expected_identity)
    if decision.action == "complete":
        completed = json.loads(
            (run_dir / "training_manifest.json").read_text(encoding="utf-8")
        )
        completed["idempotent_reuse"] = True
        return completed

    import torch
    from datasets import Dataset
    from unsloth import FastLanguageModel
    from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed

    set_seed(SEED)

    progress_path = run_dir / PROGRESS_FILENAME
    progress = (
        json.loads(progress_path.read_text(encoding="utf-8"))
        if progress_path.is_file()
        else {
            "schema_version": "dialam_modal_training_progress_v1",
            "run_identity_sha256": run_identity_sha256,
            "resume_events": [],
        }
    )
    progress["resume_events"].append(
        {
            "started_at": datetime.now(UTC).isoformat(),
            "action": decision.action,
            "from_global_step": decision.global_step,
            "checkpoint": str(decision.checkpoint) if decision.checkpoint else None,
            "ignored_incomplete_checkpoints": list(
                decision.ignored_incomplete_checkpoints
            ),
        }
    )
    progress.update(
        {
            "status": "initializing",
            "latest_committed_global_step": decision.global_step,
        }
    )
    write_json_atomic(progress_path, progress)
    volume.commit()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(V5_SOURCE_ADAPTER),
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=None,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model.config.use_cache = False
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    if trainable_parameters <= 0:
        raise RuntimeError("continued v5 adapter loaded with no trainable parameters")

    label_ids = {
        label: tokenizer(label, add_special_tokens=False)["input_ids"]
        for label in PAIRWISE_LABELS
    }
    if any(not tokens for tokens in label_ids.values()):
        raise ValueError("restricted-label tokenization is empty")

    def tokenize_listwise(row: dict) -> dict[str, list[int] | int]:
        prompt_text = tokenizer.apply_chat_template(
            row["messages"][:1],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        if not prompt_ids:
            raise ValueError("restricted-label prompt tokenization is empty")
        if len(prompt_ids) + max(len(tokens) for tokens in label_ids.values()) > MAX_SEQ_LENGTH:
            raise ValueError("restricted-label sequence exceeds the registered max length")
        tokenized: dict[str, list[int] | int] = {
            "target_index": PAIRWISE_LABELS.index(row["label"]),
        }
        for label in PAIRWISE_LABELS:
            tokens = label_ids[label]
            prefix = label.lower()
            tokenized[f"{prefix}_input_ids"] = prompt_ids + tokens
            tokenized[f"{prefix}_attention_mask"] = [1] * (
                len(prompt_ids) + len(tokens)
            )
            tokenized[f"{prefix}_labels"] = [-100] * len(prompt_ids) + tokens
        return tokenized

    dataset = Dataset.from_list(rows).map(
        tokenize_listwise,
        remove_columns=list(rows[0]),
        desc=f"tokenize DialAM {dataset_version} n={size}",
    )

    class ListwiseCollator:
        def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
            sequences = []
            sequence_labels = []
            for feature in features:
                for label in PAIRWISE_LABELS:
                    prefix = label.lower()
                    sequences.append(
                        {
                            "input_ids": feature[f"{prefix}_input_ids"],
                            "attention_mask": feature[f"{prefix}_attention_mask"],
                        }
                    )
                    sequence_labels.append(feature[f"{prefix}_labels"])
            padded = tokenizer.pad(sequences, padding=True, return_tensors="pt")
            padded_labels = torch.full_like(padded["input_ids"], -100)
            for row_index, row_labels in enumerate(sequence_labels):
                padded_labels[row_index, : len(row_labels)] = torch.tensor(
                    row_labels,
                    dtype=torch.long,
                )
            return {
                "input_ids": padded["input_ids"],
                "attention_mask": padded["attention_mask"],
                "labels": padded_labels,
                "target_indices": torch.tensor(
                    [feature["target_index"] for feature in features],
                    dtype=torch.long,
                ),
            }

    class ListwiseTrainer(Trainer):
        def compute_loss(
            self,
            model: object,
            inputs: dict,
            return_outputs: bool = False,
            num_items_in_batch: object | None = None,
        ) -> object:
            del num_items_in_batch
            labels = inputs.pop("labels")
            target_indices = inputs.pop("target_indices")
            outputs = model(**inputs)
            shift_logits = outputs.logits[..., :-1, :].contiguous().float()
            shift_labels = labels[..., 1:].contiguous()
            mask = shift_labels.ne(-100)
            safe_labels = shift_labels.masked_fill(~mask, 0)
            token_log_probs = torch.nn.functional.log_softmax(
                shift_logits,
                dim=-1,
            ).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
            sequence_scores = (token_log_probs * mask).sum(dim=1) / mask.sum(
                dim=1
            ).clamp_min(1)
            label_scores = sequence_scores.view(-1, len(PAIRWISE_LABELS))
            loss = torch.nn.functional.cross_entropy(label_scores, target_indices)
            return (loss, outputs) if return_outputs else loss

    class PersistentCheckpointCallback(TrainerCallback):
        def on_save(
            self,
            args: object,
            state: object,
            control: object,
            **kwargs: object,
        ) -> object:
            del kwargs
            checkpoint = Path(str(args.output_dir)) / f"checkpoint-{state.global_step}"
            errors = checkpoint_validation_errors(checkpoint)
            if errors:
                raise RuntimeError(
                    f"refusing to commit incomplete checkpoint {checkpoint}: {errors}"
                )
            current = json.loads(progress_path.read_text(encoding="utf-8"))
            current.update(
                {
                    "status": "checkpointed",
                    "latest_committed_global_step": state.global_step,
                    "latest_checkpoint": str(checkpoint),
                    "latest_checkpoint_files": sorted(
                        item.name for item in checkpoint.iterdir() if item.is_file()
                    ),
                    "committed_at": datetime.now(UTC).isoformat(),
                }
            )
            write_json_atomic(progress_path, current)
            volume.commit()
            print(
                f"committed resumable checkpoint at global step {state.global_step}",
                flush=True,
            )
            if interrupt_after_steps and state.global_step >= interrupt_after_steps:
                write_json_atomic(
                    run_dir / "intentional_interrupt.json",
                    {
                        "schema_version": "dialam_intentional_interrupt_v1",
                        "committed_global_step": state.global_step,
                        "checkpoint": str(checkpoint),
                        "created_at": datetime.now(UTC).isoformat(),
                    },
                )
                volume.commit()
                raise RuntimeError(
                    "INTENTIONAL_RESUME_SMOKE_INTERRUPTION_AFTER_COMMITTED_CHECKPOINT"
                )
            return control

    training_args = TrainingArguments(
        output_dir=str(run_dir / "trainer"),
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["per_device_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        lr_scheduler_type=config["lr_scheduler_type"],
        warmup_ratio=config["warmup_ratio"],
        logging_steps=5,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=config["save_total_limit"],
        save_safetensors=True,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        optim=config["optimizer"],
        report_to="none",
        seed=SEED,
        data_seed=SEED,
        remove_unused_columns=False,
        ignore_data_skip=False,
        gradient_checkpointing=True,
    )
    trainer = ListwiseTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=ListwiseCollator(),
        callbacks=[PersistentCheckpointCallback()],
    )
    train_result = trainer.train(
        resume_from_checkpoint=(
            str(decision.checkpoint) if decision.action == "resume" else None
        )
    )
    global_step = int(trainer.state.global_step)
    log_history = list(trainer.state.log_history)
    adapter_dir = run_dir / "adapter"
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
    reload_probe, reload_scores = _score_pairwise_labels(
        reloaded_model,
        reloaded_tokenizer,
        rows[0]["messages"][0]["content"],
    )
    if not reload_probe.strip():
        raise RuntimeError("saved restricted-label adapter produced an empty label")
    adapter_hash, adapter_files = tree_manifest(adapter_dir)
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress.update(
        {
            "status": "complete",
            "completed_global_step": global_step,
            "completed_at": datetime.now(UTC).isoformat(),
        }
    )
    write_json_atomic(progress_path, progress)
    manifest = {
        "schema_version": (
            "dialam_modal_model_error_corrective_qlora_run_v1"
            if is_v9
            else "dialam_modal_listwise_qlora_run_v1"
        ),
        "completed_at": datetime.now(UTC).isoformat(),
        "size": size,
        "dataset_version": dataset_version,
        "train_sha256": file_sha256(train_path),
        "run_identity_sha256": run_identity_sha256,
        "fixed_config": config,
        "loss_config": loss_config,
        "source_adapter": {
            "path": str(V5_SOURCE_ADAPTER),
            "tree_sha256": source_adapter_hash,
            "files": source_adapter_files,
        },
        "resumability": {
            "resume_mode": resume_mode,
            "initial_action": decision.action,
            "resumed_from_step": decision.global_step,
            "resume_events": progress["resume_events"],
            "save_steps": save_steps,
            "save_total_limit": config["save_total_limit"],
            "full_trainer_state": True,
            "volume_commit_on_save": True,
            "trainer_state_preserved_after_completion": True,
        },
        "global_step": global_step,
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "label_token_counts": {
            label: len(tokens) for label, tokens in label_ids.items()
        },
        "gpu": torch.cuda.get_device_name(0),
        "cuda": torch.version.cuda,
        "torch": str(torch.__version__),
        "metrics": train_result.metrics,
        "log_history": log_history,
        "adapter_path": str(adapter_dir),
        "adapter_tree_sha256": adapter_hash,
        "adapter_files": adapter_files,
        "reload_verified": True,
        "reload_probe": reload_probe,
        "reload_probe_label_scores": reload_scores,
        "idempotent_reuse": False,
    }
    write_json_atomic(run_dir / "training_manifest.json", manifest)
    volume.commit()
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
        if dataset_version in PAIRWISE_DATASET_VERSIONS
        else EVAL_INPUT_FILENAMES
    )
    rows = _load_rows(REMOTE_DATA_DIR / input_filenames[eval_split])
    predictions = []
    for index, row in enumerate(rows, start=1):
        pairwise_decisions = None
        if dataset_version in PAIRWISE_DATASET_VERSIONS:
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


@app.function(
    image=image,
    gpu="L4",
    timeout=60 * 60,
    volumes={str(PERSISTENT_ROOT): volume},
)
def generate_resumable_v8_eval(
    size: int,
    dataset_version: str,
    eval_split: str,
) -> dict:
    """Persist each completed restricted-label scenario for exact resume."""

    import torch
    from unsloth import FastLanguageModel

    if dataset_version not in {"v8-smoke", "v8", "v9-smoke", "v9"}:
        raise ValueError("resumable evaluation is registered only for v8 or v9")
    if eval_split not in EVAL_INPUT_FILENAMES:
        raise ValueError(f"eval_split must be one of {tuple(EVAL_INPUT_FILENAMES)}")
    volume.reload()
    checkpoint_label = _checkpoint_label(size, dataset_version)
    adapter_dir = CHECKPOINT_ROOT / checkpoint_label / "adapter"
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"missing trained adapter: {adapter_dir}")
    adapter_hash, adapter_files = tree_manifest(adapter_dir)
    input_path = REMOTE_DATA_DIR / V5_EVAL_INPUT_FILENAMES[eval_split]
    rows = _load_rows(input_path)
    expected_identity = {
        "schema_version": "dialam_modal_resumable_eval_identity_v1",
        "implementation_version": (
            "dialam-v9-pairwise-eval-resume-v1"
            if dataset_version in {"v9-smoke", "v9"}
            else "dialam-v8-pairwise-eval-resume-v1"
        ),
        "checkpoint_label": checkpoint_label,
        "dataset_version": dataset_version,
        "eval_split": eval_split,
        "input_sha256": file_sha256(input_path),
        "example_ids": [row["example_id"] for row in rows],
        "adapter_tree_sha256": adapter_hash,
        "scoring": "mean allowed-label token log probability",
        "labels": list(PAIRWISE_LABELS),
    }
    identity_hash = object_sha256(expected_identity)
    run_dir = EVALUATION_ROOT / f"{checkpoint_label}_{eval_split}"
    identity_path = run_dir / "run_identity.json"
    progress_path = run_dir / "progress.json"
    completed_path = run_dir / "completed.json"
    row_dir = run_dir / "rows"
    if run_dir.exists():
        if not identity_path.is_file():
            raise RuntimeError("existing evaluation has no immutable run identity")
        observed = json.loads(identity_path.read_text(encoding="utf-8"))
        observed_hash = observed.pop("identity_sha256", None)
        if observed_hash != object_sha256(observed):
            raise RuntimeError("stored evaluation identity hash is invalid")
        if observed != expected_identity or observed_hash != identity_hash:
            raise RuntimeError(
                "evaluation identity mismatch; input, adapter, split, or scorer changed"
            )
    else:
        run_dir.mkdir(parents=True)
        row_dir.mkdir()
        write_json_atomic(
            identity_path,
            {**expected_identity, "identity_sha256": identity_hash},
        )
        write_json_atomic(
            progress_path,
            {
                "schema_version": "dialam_modal_resumable_eval_progress_v1",
                "identity_sha256": identity_hash,
                "resume_events": [],
                "completed_count": 0,
            },
        )
        volume.commit()

    row_dir.mkdir(exist_ok=True)

    def load_persisted() -> tuple[list[dict | None], int]:
        persisted: list[dict | None] = []
        count = 0
        for index, expected in enumerate(rows):
            path = row_dir / f"{index:04d}.json"
            if not path.is_file():
                persisted.append(None)
                continue
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("example_id") != expected["example_id"]:
                raise RuntimeError(f"persisted evaluation row {index} has the wrong ID")
            persisted.append(value)
            count += 1
        return persisted, count

    persisted, resumed_count = load_persisted()
    if completed_path.is_file():
        completed = json.loads(completed_path.read_text(encoding="utf-8"))
        predictions = [row for row in persisted if row is not None]
        if len(predictions) != len(rows):
            raise RuntimeError("completed evaluation is missing persisted scenario rows")
        if completed.get("predictions_sha256") != object_sha256(predictions):
            raise RuntimeError("completed evaluation prediction hash is invalid")
        return {
            "predictions": predictions,
            "resume": {
                "idempotent_reuse": True,
                "resumed_completed_scenarios": len(predictions),
                "newly_completed_scenarios": 0,
                "total_scenarios": len(predictions),
                "identity_sha256": identity_hash,
                "adapter_tree_sha256": adapter_hash,
                "predictions_sha256": completed["predictions_sha256"],
            },
        }

    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress["resume_events"].append(
        {
            "resumed_completed_scenarios": resumed_count,
            "missing_scenarios": len(rows) - resumed_count,
        }
    )
    write_json_atomic(progress_path, progress)
    volume.commit()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_dir),
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=None,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    FastLanguageModel.for_inference(model)
    newly_completed = 0
    for index, row in enumerate(rows):
        if persisted[index] is not None:
            print(
                f"reused tuned {index + 1}/{len(rows)} from persistent eval state",
                flush=True,
            )
            continue
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
        prediction = {
            "example_id": row["example_id"],
            "update_id": row["update_id"],
            "target": "tuned",
            "model": BASE_MODEL,
            "adapter_size": size,
            "dataset_version": dataset_version,
            "eval_split": eval_split,
            "raw_response": json.dumps(
                {"relations": relations},
                separators=(",", ":"),
            ),
            "pairwise_decisions": pairwise_decisions,
        }
        write_json_atomic(row_dir / f"{index:04d}.json", prediction)
        persisted[index] = prediction
        newly_completed += 1
        progress.update(
            {
                "completed_count": resumed_count + newly_completed,
                "latest_completed_index": index,
                "latest_completed_example_id": row["example_id"],
            }
        )
        write_json_atomic(progress_path, progress)
        volume.commit()
        print(f"generated tuned {index + 1}/{len(rows)} and committed", flush=True)

    del model
    torch.cuda.empty_cache()
    predictions = [row for row in persisted if row is not None]
    if len(predictions) != len(rows):
        raise RuntimeError("resumable evaluation did not cover every scenario")
    predictions_hash = object_sha256(predictions)
    write_json_atomic(
        completed_path,
        {
            "schema_version": "dialam_modal_resumable_eval_complete_v1",
            "identity_sha256": identity_hash,
            "scenario_count": len(predictions),
            "predictions_sha256": predictions_hash,
            "adapter_tree_sha256": adapter_hash,
            "adapter_files": adapter_files,
        },
    )
    volume.commit()
    return {
        "predictions": predictions,
        "resume": {
            "idempotent_reuse": False,
            "resumed_completed_scenarios": resumed_count,
            "newly_completed_scenarios": newly_completed,
            "total_scenarios": len(predictions),
            "identity_sha256": identity_hash,
            "adapter_tree_sha256": adapter_hash,
            "predictions_sha256": predictions_hash,
        },
    }


@app.local_entrypoint()
def main(
    action: str,
    size: int = 256,
    target: str = "base",
    dataset_version: str = "v1",
    eval_split: str = "frozen",
    output_path: str = "",
    resume_mode: str = "auto",
    interrupt_after_steps: int = 0,
    stop_after_chunks: int = 0,
) -> None:
    resume_result = None
    if action == "train":
        checkpoint_label = _checkpoint_label(size, dataset_version)
        if dataset_version in {"v7-smoke", "v7"}:
            if resume_mode not in {"never", "auto", "required"}:
                raise ValueError("resume_mode must be never, auto, or required")
            result = train_preference_checkpoint.remote(
                size,
                dataset_version,
                resume_mode,
                interrupt_after_steps,
            )
        elif dataset_version in {"v8-smoke", "v8", "v9-smoke", "v9"}:
            if resume_mode not in {"never", "auto", "required"}:
                raise ValueError("resume_mode must be never, auto, or required")
            result = train_listwise_checkpoint.remote(
                size,
                dataset_version,
                resume_mode,
                interrupt_after_steps,
            )
        else:
            if interrupt_after_steps:
                raise ValueError(
                    "intentional interruption is available only for v7, v8, or v9"
                )
            result = train_checkpoint.remote(size, dataset_version)
        default = (
            PROJECT_ROOT
            / "artifacts"
            / "dialam_qlora"
            / checkpoint_label
            / "remote_training_result.json"
        )
    elif action == "evaluate":
        if target == "tuned" and dataset_version in {
            "v8-smoke",
            "v8",
            "v9-smoke",
            "v9",
        }:
            resumable_result = generate_resumable_v8_eval.remote(
                size,
                dataset_version,
                eval_split,
            )
            result = resumable_result["predictions"]
            resume_result = resumable_result["resume"]
        else:
            result = generate_model_eval.remote(target, size, dataset_version, eval_split)
        label = "base" if target == "base" else _checkpoint_label(size, dataset_version)
        default = (
            PROJECT_ROOT
            / "results"
            / "dialam_model_eval"
            / (label if eval_split == "frozen" else f"{label}_{eval_split}")
            / "predictions.jsonl"
        )
    elif action == "mine-v9":
        mined = mine_v9_hard_negatives.remote(stop_after_chunks)
        result = mined["scores"]
        resume_result = mined["resume"]
        default = LOCAL_DATA_DIR / V9_MINING_SCORE_FILENAME
    else:
        raise ValueError("action must be train, evaluate, or mine-v9")

    output = Path(output_path).resolve() if output_path else default
    output.parent.mkdir(parents=True, exist_ok=True)
    if action == "train":
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        with output.open("w", encoding="utf-8") as handle:
            for row in result:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        if resume_result is not None:
            resume_output = output.with_name(f"{output.stem}.resume.json")
            resume_output.write_text(
                json.dumps(resume_result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    print(output)

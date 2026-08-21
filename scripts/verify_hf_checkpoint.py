#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load a fused checkpoint with Transformers and run a minimal generation"
    )
    parser.add_argument("model", type=Path)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float32,
        device_map="cpu",
    )
    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": "Return only JSON for this empty graph: {\"relations\":[]}",
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(
        output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()
    if not generated:
        raise RuntimeError("checkpoint loaded but produced no verification tokens")
    evidence = {
        "verified_at": datetime.now(UTC).isoformat(),
        "runtime": "transformers",
        "device": "cpu",
        "model_path": str(args.model),
        "model_class": type(model).__name__,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "generation_nonempty": True,
    }
    output_path = args.model / "flowjudge_hf_verification.json"
    output_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()

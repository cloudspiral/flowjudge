from __future__ import annotations

import os

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.environ["FLOWJUDGE_MODEL_ID"]

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto", device_map="auto")
model.eval()


def reconstruct_graph(transcript: str) -> str:
    prompt = (
        "Identify every direct response edge in this debate excerpt. Return only one bare JSON "
        "object with a relations array. An edge must point from a later opposing-side unit to the "
        "earlier unit it directly answers, attacks, mitigates, or turns. Exclude topical similarity, "
        "same-side extensions, repetition/rephrase, and independent counterarguments.\n\n"
        f"{transcript.strip()}"
    )
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
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()


EXAMPLE = """Resolution: Cities should adopt congestion pricing.
U1 [AFF]: Congestion pricing reduces traffic by charging drivers at peak times.
U2 [AFF]: The revenue can also improve public transit.
U3 [NEG]: Charging at peak times does not reduce necessary trips; it only makes commuting more expensive.
U4 [NEG]: The proposal would begin next January.
U5 [AFF]: That affordability objection is mitigated by exempting low-income commuters."""

demo = gr.Interface(
    fn=reconstruct_graph,
    inputs=gr.Textbox(label="Numbered debate transcript", lines=12, value=EXAMPLE),
    outputs=gr.Code(label="Predicted response graph", language="json"),
    title="FlowJudge Qwen3 0.6B",
    description="Reconstruct direct later-to-earlier cross-side response edges. This research checkpoint is intentionally narrow and imperfect.",
)

if __name__ == "__main__":
    demo.launch()

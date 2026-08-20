from __future__ import annotations

from pathlib import Path

from .data import PROJECT_ROOT, render_transcript
from .schemas import Scenario

PROMPT_NAMES = ("zero_shot", "few_shot", "structured_checklist")
PROMPTS_DIR = PROJECT_ROOT / "prompts"


def load_prompt_template(name: str) -> str:
    if name not in (*PROMPT_NAMES, "judge"):
        raise ValueError(f"unknown prompt template: {name}")
    return (PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8")


def build_prompt(name: str, scenario: Scenario) -> str:
    template = load_prompt_template(name)
    marker = "{{TRANSCRIPT}}"
    if template.count(marker) != 1:
        raise ValueError(f"{name}.txt must contain exactly one {marker} marker")
    return template.replace(marker, render_transcript(scenario))

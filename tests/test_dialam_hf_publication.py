from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = PROJECT_ROOT / "hf_dataset" / "dialam_patch"


def _keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | set().union(*(_keys(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(_keys(item) for item in value)) if value else set()
    return set()


def test_public_metadata_omits_uncleared_qt30_identifiers() -> None:
    forbidden_keys = {
        "dialogue_ids",
        "eval_dialogue_ids",
        "example_ids",
        "failure_cases",
        "few_shot_example_ids",
        "heldout_dialogue_ids",
        "heldout_parent_episode_ids",
        "raw_artifacts",
        "scenario_failure_cases",
        "train_dialogue_ids",
        "training_parent_episode_ids",
    }
    forbidden_values = re.compile(
        r"qt30:|nodeset\d+|qt\d{8}wtxt|cutie(?:s)?testrun\w+",
        re.IGNORECASE,
    )

    for path in sorted((PACKAGE / "metadata").glob("*.json")):
        text = path.read_text(encoding="utf-8")
        value = json.loads(text)
        assert not (_keys(value) & forbidden_keys), path
        assert not forbidden_values.search(text), path


def test_publication_manifest_hashes_every_declared_file() -> None:
    manifest = json.loads(
        (PACKAGE / "publish_manifest.json").read_text(encoding="utf-8")
    )
    paths = [entry["path"] for entry in manifest["files"]]
    assert len(paths) == len(set(paths))
    assert manifest["contains_raw_or_transformed_qt30_text"] is False
    assert manifest["contains_original_qt30_identifiers"] is False
    for entry in manifest["files"]:
        artifact = PACKAGE / entry["path"]
        assert artifact.is_file()
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == entry["sha256"]
    actual = {
        str(path.relative_to(PACKAGE))
        for path in PACKAGE.rglob("*")
        if path.is_file() and path.name != "publish_manifest.json"
    }
    assert actual == set(paths)

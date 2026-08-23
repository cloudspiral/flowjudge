from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = PROJECT_ROOT / "hf_dataset" / "dialam_patch"

def test_public_package_contains_only_transformed_rows_not_raw_source_maps() -> None:
    manifest = json.loads(
        (PACKAGE / "publish_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["contains_raw_qt30_archive_or_maps"] is False
    assert manifest["contains_transformed_qt30_text"] is True
    assert manifest["contains_original_qt30_identifiers"] is True
    assert manifest["permission_basis"]
    assert re.fullmatch(r"[0-9a-f]{40}", manifest["source_code_commit"])
    paths = {entry["path"] for entry in manifest["files"]}
    assert "data/train_v3_n4096.jsonl" in paths
    assert "data/frozen_own_eval_30.jsonl" in paths
    assert "evidence/prompt_ceiling_candidate_and_judge_records.jsonl" in paths
    assert "evidence/v3_n4096_judge_transcripts.jsonl" in paths
    assert "evidence/training_v1_n256.json" in paths
    assert "evidence/training_v5_n8192.json" in paths
    assert "data/train_v5_n8192.jsonl" in paths
    assert "metadata/dialam_v5_1_result.json" in paths
    assert "evidence/v5_1_n8192_frozen_candidate_transcripts.jsonl" in paths
    assert "evidence/v5_1_n8192_frozen_judge_transcripts.jsonl" in paths
    assert "reconstruction/eval.py" in paths
    assert not any(path.endswith(".zip") for path in paths)
    assert not any("/maps/" in f"/{path}" or "/raw/" in f"/{path}" for path in paths)


def test_public_transformed_jsonl_is_valid_and_row_counts_match() -> None:
    manifest = json.loads(
        (PACKAGE / "publish_manifest.json").read_text(encoding="utf-8")
    )
    entries = [entry for entry in manifest["files"] if entry["path"].endswith(".jsonl")]
    assert entries
    for entry in entries:
        path = PACKAGE / entry["path"]
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        assert len(rows) == entry["rows"]
        assert rows


def test_publication_manifest_hashes_every_declared_file() -> None:
    manifest = json.loads(
        (PACKAGE / "publish_manifest.json").read_text(encoding="utf-8")
    )
    paths = [entry["path"] for entry in manifest["files"]]
    assert len(paths) == len(set(paths))
    assert manifest["contains_raw_qt30_archive_or_maps"] is False
    assert manifest["contains_transformed_qt30_text"] is True
    assert manifest["contains_original_qt30_identifiers"] is True
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


def test_publication_tree_contains_no_api_credentials() -> None:
    credential_pattern = re.compile(
        r"(?:sk-[A-Za-z0-9_-]{20,}|authorization\s*[:=]\s*bearer\s+\S+|x-api-key\s*[:=])",
        re.IGNORECASE,
    )
    for path in PACKAGE.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".jsonl", ".md", ".txt"}:
            assert not credential_pattern.search(path.read_text(encoding="utf-8")), path

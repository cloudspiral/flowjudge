#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ID_PATTERN = re.compile(r"\bU[1-9][0-9]*\b")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the error-driven v2 attachment and multi-edge dataset"
    )
    parser.add_argument("--input", type=Path, default=Path("data/training/v1_n96.jsonl"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/training/v2_attachment_n120.jsonl")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/training/v2_attachment_manifest.json")
    )
    parser.add_argument("--variants", type=int, default=24)
    args = parser.parse_args()
    print(build_v2(args.input, args.output, args.manifest, variants=args.variants))


def build_v2(input_path: Path, output_path: Path, manifest_path: Path, *, variants: int) -> Path:
    rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line]
    if variants < 1 or variants > len(rows):
        raise ValueError("variants must be between 1 and the number of v1 rows")
    ranked = sorted(rows, key=lambda row: (-_attachment_hardness(row), row["example_id"]))
    selected = ranked[:variants]
    augmented = [
        _remap_example(row, start_id=1001 + index * 20)
        for index, row in enumerate(selected)
    ]
    combined = rows + augmented
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in combined),
        encoding="utf-8",
    )
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "version": "v2_attachment",
        "diagnosed_mvp_failure": "n=96 produced one edge per case and usually attached it to a nearby wrong unit",
        "data_change": "add semantics-preserving ID-remapped variants of the 24 hardest multi-edge and cross-side-attachment examples",
        "training_configuration_change": False,
        "base_examples": len(rows),
        "added_variants": len(augmented),
        "total_examples": len(combined),
        "selected_example_ids": [row["example_id"] for row in selected],
        "output": str(output_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output_path


def _attachment_hardness(row: dict[str, Any]) -> int:
    graph = json.loads(row["messages"][1]["content"])
    gold = {(edge["source"], edge["target"]) for edge in graph["relations"]}
    prompt = row["messages"][0]["content"]
    units = [
        (match.group(1), match.group(2))
        for match in re.finditer(r"^(U[1-9][0-9]*) \[(AFF|NEG)\]:", prompt, re.MULTILINE)
    ]
    cross_side_candidates = 0
    for source_index, (source_id, source_side) in enumerate(units):
        for target_id, target_side in units[:source_index]:
            if source_side != target_side and (source_id, target_id) not in gold:
                cross_side_candidates += 1
    return len(gold) * 100 + cross_side_candidates


def _remap_example(row: dict[str, Any], *, start_id: int) -> dict[str, Any]:
    copied = json.loads(json.dumps(row))
    prompt = copied["messages"][0]["content"]
    ordered_ids = []
    for value in ID_PATTERN.findall(prompt):
        if value not in ordered_ids:
            ordered_ids.append(value)
    mapping = {old: f"U{start_id + index}" for index, old in enumerate(ordered_ids)}

    def replace(match: re.Match[str]) -> str:
        return mapping.get(match.group(0), match.group(0))

    copied["messages"][0]["content"] = ID_PATTERN.sub(replace, prompt)
    graph = json.loads(copied["messages"][1]["content"])
    for edge in graph["relations"]:
        edge["source"] = mapping[edge["source"]]
        edge["target"] = mapping[edge["target"]]
    copied["messages"][1]["content"] = json.dumps(graph, separators=(",", ":"))
    copied["example_id"] = f"{copied['example_id']}__v2_id_remap"
    copied["v2_augmentation"] = {
        "type": "id_remap_attachment_hard_case",
        "source_example_id": row["example_id"],
        "id_mapping": mapping,
    }
    return copied


if __name__ == "__main__":
    main()

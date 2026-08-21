#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

TOPICS = (
    "regional bus",
    "school lunch",
    "public library",
    "community clinic",
    "recycling pickup",
    "rural broadband",
    "city park",
    "after-school tutoring",
    "flood barrier",
    "bike-share",
    "job-training",
    "water conservation",
    "housing inspection",
    "fire prevention",
    "public market",
    "senior transport",
    "tree planting",
    "road repair",
    "childcare voucher",
    "farm cooperative",
    "storm shelter",
    "arts grant",
    "energy retrofit",
    "ambulance dispatch",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build v2 with explicit multi-edge and same-side-negative curriculum cases"
    )
    parser.add_argument("--input", type=Path, default=Path("data/training/v1_n96.jsonl"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/training/v2_curriculum_n120.jsonl")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/training/v2_curriculum_manifest.json")
    )
    args = parser.parse_args()
    print(build_v2(args.input, args.output, args.manifest))


def build_v2(input_path: Path, output_path: Path, manifest_path: Path) -> Path:
    real_rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line]
    if len(real_rows) != 96:
        raise ValueError(f"expected 96 real v1 rows, found {len(real_rows)}")
    curriculum = [_curriculum_example(index, topic) for index, topic in enumerate(TOPICS)]
    combined = real_rows + curriculum
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in combined),
        encoding="utf-8",
    )
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "version": "v2_curriculum",
        "diagnosed_mvp_failure": "v1 n=96 emitted one edge per case and usually attached it to a nearby wrong unit",
        "rejected_v2_attempt": "ID-remapped hard-case reweighting reduced own-eval edge F1 from 16% to 8%",
        "data_change": "add 24 concise three-edge response chains with adjacent same-side and independent hard negatives",
        "training_configuration_change": False,
        "real_vivesdebate_examples": len(real_rows),
        "manual_curriculum_examples": len(curriculum),
        "real_data_share": len(real_rows) / len(combined),
        "total_examples": len(combined),
        "output": str(output_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output_path


def _curriculum_example(index: int, topic: str) -> dict:
    start = 2001 + index * 10
    ids = [f"U{start + offset}" for offset in range(6)]
    if index % 2 == 0:
        first, second = "AFF", "NEG"
    else:
        first, second = "NEG", "AFF"
    units = [
        (ids[0], first, f"The {topic} plan lowers costs by sharing fixed expenses."),
        (ids[1], first, f"The {topic} plan also publishes transparent monthly reports."),
        (
            ids[2],
            second,
            "Sharing fixed expenses does not lower total costs; it merely shifts them to taxpayers.",
        ),
        (ids[3], second, f"The {topic} plan would begin next July."),
        (
            ids[4],
            first,
            "That cost-shifting objection fails because user fees cover the plan without new taxes.",
        ),
        (
            ids[5],
            second,
            "User fees do not cover maintenance, so the claimed tax-free funding still leaves a shortfall.",
        ),
    ]
    transcript = "\n".join(f"{unit_id} [{side}]: {text}" for unit_id, side, text in units)
    prompt = (
        "Identify every direct response edge in this debate excerpt. Return only one bare JSON object "
        "with a relations array. An edge must point from a later opposing-side unit to the earlier unit "
        "it directly answers, attacks, mitigates, or turns. Exclude topical similarity, same-side "
        "extensions, repetition/rephrase, and independent counterarguments.\n\n"
        "Resolution: Local governments should adopt the proposed public-service plan.\n"
        f"{transcript}"
    )
    graph = {
        "relations": [
            {"source": ids[2], "target": ids[0], "type": "responds_to"},
            {"source": ids[4], "target": ids[2], "type": "responds_to"},
            {"source": ids[5], "target": ids[4], "type": "responds_to"},
        ]
    }
    return {
        "example_id": f"v2_curriculum_{index + 1:02d}",
        "source_debate": "manual_v2_curriculum",
        "source_unit_ids": [int(value[1:]) for value in ids],
        "teacher_model": "manual-error-driven-template",
        "filter_model": "deterministic-v2-validator",
        "quality": {
            "accept": True,
            "meaning_preserved": True,
            "edge_support_preserved": True,
            "no_new_response_cues": True,
            "fluent_and_atomic": True,
            "brief_reason": "Three explicit cross-side response links and two excluded adjacent hard negatives are unambiguous by construction.",
        },
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": json.dumps(graph, separators=(",", ":"))},
        ],
        "v2_augmentation": {
            "type": "multi_edge_attachment_curriculum",
            "topic": topic,
            "hard_negative_pairs": [
                {"source": ids[1], "target": ids[0], "reason": "same-side extension"},
                {"source": ids[3], "target": ids[1], "reason": "independent timing claim"},
            ],
        },
    }


if __name__ == "__main__":
    main()

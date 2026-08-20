from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from flowjudge.data import PROJECT_ROOT, load_benchmark
from flowjudge.schemas import (
    Category,
    GoldBlueprint,
    HardNegative,
    JuryOutcome,
    JuryScore,
    Scenario,
    SourceAnnotationIssue,
    SourceProvenance,
    SourceRelation,
    SourceRelationType,
    SourceStance,
    Unit,
)

SOURCE_DIR = PROJECT_ROOT / "data" / "source" / "vivesdebate"
SOURCE_MANIFEST_PATH = SOURCE_DIR / "manifest.json"
SCENARIOS_PATH = PROJECT_ROOT / "data" / "benchmark_scenarios.jsonl"
GOLD_PATH = PROJECT_ROOT / "data" / "benchmark_gold.jsonl"
BENCHMARK_MANIFEST_PATH = PROJECT_ROOT / "data" / "benchmark_manifest.json"

RESOLUTION = "Should surrogacy be legalised?"
RELATION_TYPES = {
    "RA": SourceRelationType.INFERENCE,
    "CA": SourceRelationType.CONFLICT,
    "MA": SourceRelationType.REPHRASE,
}
SIDE_BY_STANCE = {SourceStance.FAVOUR: "AFF", SourceStance.AGAINST: "NEG"}


def main() -> None:
    source_manifest = json.loads(SOURCE_MANIFEST_PATH.read_text(encoding="utf-8"))
    _verify_file(SOURCE_DIR / source_manifest["evaluation_file"]["file"], source_manifest["evaluation_file"]["md5"])
    jury_by_debate = _load_jury_outcomes(SOURCE_DIR / source_manifest["evaluation_file"]["file"])

    scenarios: list[Scenario] = []
    blueprints: list[GoldBlueprint] = []
    debate_stats: list[dict[str, Any]] = []
    for selection in source_manifest["selected_debates"]:
        source_path = SOURCE_DIR / selection["file"]
        _verify_file(source_path, selection["md5"])
        scenario, blueprint, stats = _convert_debate(
            source_path,
            selection["debate_id"],
            selection["md5"],
            jury_by_debate[selection["debate_id"]],
        )
        scenarios.append(scenario)
        blueprints.append(blueprint)
        debate_stats.append(stats)

    synthetic = _synthetic_cases()
    scenarios.extend(case[0] for case in synthetic)
    blueprints.extend(case[1] for case in synthetic)

    _write_jsonl(SCENARIOS_PATH, scenarios)
    _write_jsonl(GOLD_PATH, blueprints)
    cases = load_benchmark()
    benchmark_manifest = _build_benchmark_manifest(source_manifest, cases, debate_stats)
    BENCHMARK_MANIFEST_PATH.write_text(
        json.dumps(benchmark_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "scenarios": len(cases),
                "real_debates": sum(case.scenario.category == Category.VIVESDEBATE for case in cases),
                "synthetic_scenarios": sum(case.scenario.category != Category.VIVESDEBATE for case in cases),
                "converted_units": sum(len(case.scenario.units) for case in cases),
                "gold_response_edges": sum(len(case.gold.gold_relations) for case in cases),
            },
            indent=2,
        )
    )


def _convert_debate(
    source_path: Path,
    debate_id: str,
    source_md5: str,
    jury_outcome: JuryOutcome,
) -> tuple[Scenario, GoldBlueprint, dict[str, Any]]:
    with source_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    if not rows:
        raise ValueError(f"{source_path.name} is empty")

    ids = [int(row["ID (Chronological)"]) for row in rows]
    if ids != sorted(set(ids)):
        raise ValueError(f"{source_path.name} IDs are not unique and chronological")
    relation_slots = _relation_slots(fieldnames)
    units: list[Unit] = []
    for row in rows:
        source_id = int(row["ID (Chronological)"])
        stance_raw = row["TEAM STANCE"]
        if not stance_raw:
            raise ValueError(f"{source_path.name} U{source_id} has no source stance")
        stance = SourceStance(stance_raw.upper())
        units.append(
            Unit(
                id=f"U{source_id}",
                side=SIDE_BY_STANCE[stance],
                text=row["ADU_EN"],
                source_id=source_id,
                phase=row["TYPE (Part + Person)"] or None,
                argument_number=row["ARGUMENT NUMBER"] or None,
                source_stance=stance,
                text_ca=row["ADU_CAT"],
                text_es=row["ADU_ES"],
                text_en=row["ADU_EN"],
            )
        )

    unit_by_id = {unit.id: unit for unit in units}
    source_relations: list[SourceRelation] = []
    source_annotation_issues: list[SourceAnnotationIssue] = []
    relation_counts: Counter[str] = Counter()
    source_relation_slots: set[tuple[str, str]] = set()
    for row in rows:
        source = f'U{int(row["ID (Chronological)"])}'
        for related_column, type_column in relation_slots:
            related_raw = row.get(related_column, "") or ""
            label_raw = row.get(type_column, "") or ""
            if not related_raw and not label_raw:
                continue
            source_relation_slots.add((source, related_column))
            if not related_raw or not label_raw:
                source_annotation_issues.append(
                    SourceAnnotationIssue(
                        source=source,
                        source_column=related_column,
                        type_column=type_column,
                        raw_related_id=related_raw,
                        raw_relation_type=label_raw,
                        reason="incomplete_pair",
                    )
                )
                continue
            normalized_label = label_raw.upper()
            if normalized_label not in RELATION_TYPES:
                source_annotation_issues.append(
                    SourceAnnotationIssue(
                        source=source,
                        source_column=related_column,
                        type_column=type_column,
                        raw_related_id=related_raw,
                        raw_relation_type=label_raw,
                        reason="unknown_relation_type",
                    )
                )
                continue
            related_ids = _parse_related_ids(related_raw)
            for related_id in related_ids:
                target = f"U{related_id}"
                if target not in unit_by_id:
                    source_annotation_issues.append(
                        SourceAnnotationIssue(
                            source=source,
                            source_column=related_column,
                            type_column=type_column,
                            raw_related_id=related_raw,
                            raw_relation_type=label_raw,
                            reason="unknown_target",
                        )
                    )
                    continue
                relation_type = RELATION_TYPES[normalized_label]
                source_relations.append(
                    SourceRelation(
                        source=source,
                        target=target,
                        type=relation_type,
                        source_label=label_raw,
                        source_column=related_column,
                    )
                )
                relation_counts[relation_type.value] += 1

    scenario = Scenario(
        scenario_id=f"vives_{debate_id.lower()}",
        category=Category.VIVESDEBATE,
        title=f"VivesDebate {debate_id}",
        resolution=RESOLUTION,
        units=units,
        source=SourceProvenance(
            corpus="VivesDebate",
            debate_id=debate_id,
            source_file=source_path.name,
            source_md5=source_md5,
            source_doi="10.5281/zenodo.6531487",
            license="CC BY-NC-SA 4.0",
            selected_language="ADU_EN",
        ),
        source_relations=source_relations,
        source_annotation_issues=source_annotation_issues,
        jury_outcome=jury_outcome,
    )
    blueprint = _derive_gold(scenario)
    source_id_set = set(ids)
    missing_ids = sorted(set(range(min(ids), max(ids) + 1)) - source_id_set)
    cross_conflicts = sum(
        relation.type == SourceRelationType.CONFLICT
        and unit_by_id[relation.source].side != unit_by_id[relation.target].side
        for relation in source_relations
    )
    same_conflicts = sum(
        relation.type == SourceRelationType.CONFLICT
        and unit_by_id[relation.source].side == unit_by_id[relation.target].side
        for relation in source_relations
    )
    stats = {
        "debate_id": debate_id,
        "source_file": source_path.name,
        "source_md5": source_md5,
        "adu_count": len(units),
        "first_source_id": min(ids),
        "last_source_id": max(ids),
        "missing_source_ids": missing_ids,
        "missing_phase_annotations": sum(unit.phase is None for unit in units),
        "source_relation_count": len(source_relations),
        "source_relation_slot_count": len(source_relation_slots),
        "source_relations_by_type": dict(sorted(relation_counts.items())),
        "source_annotation_issue_count": len(source_annotation_issues),
        "source_annotation_issues_by_reason": dict(
            sorted(Counter(issue.reason for issue in source_annotation_issues).items())
        ),
        "cross_stance_conflicts": cross_conflicts,
        "same_stance_conflicts_preserved_not_mapped": same_conflicts,
        "flowjudge_response_edges": len(blueprint.gold_relations),
        "jury_winner": jury_outcome.winner,
        "jury_margin": jury_outcome.margin,
    }
    return scenario, blueprint, stats


def _derive_gold(scenario: Scenario) -> GoldBlueprint:
    unit_by_id = {unit.id: unit for unit in scenario.units}
    evidence: dict[tuple[str, str], list[SourceRelation]] = defaultdict(list)
    for relation in scenario.source_relations:
        if relation.type != SourceRelationType.CONFLICT:
            continue
        if unit_by_id[relation.source].side == unit_by_id[relation.target].side:
            continue
        source_number = int(relation.source[1:])
        target_number = int(relation.target[1:])
        later = relation.source if source_number > target_number else relation.target
        earlier = relation.target if source_number > target_number else relation.source
        evidence[(later, earlier)].append(relation)

    explained_edges = []
    for (later, earlier), relations in sorted(
        evidence.items(), key=lambda item: (int(item[0][0][1:]), int(item[0][1][1:]))
    ):
        original = ", ".join(
            f"{relation.source_label} {relation.source}->{relation.target}"
            for relation in relations
        )
        explained_edges.append(
            {
                "source": later,
                "target": earlier,
                "type": "responds_to",
                "explanation": (
                    f"Mapped from VivesDebate conflict annotation(s) {original}. The linked ADUs have opposing "
                    f"stances; the FlowJudge mapping treats CA as symmetric and orients the edge from later "
                    f"{later} to earlier {earlier}."
                ),
            }
        )
    return GoldBlueprint(
        scenario_id=scenario.scenario_id,
        category=scenario.category,
        design_intent=(
            "Complete VivesDebate debate converted without excerpting; gold responses are derived only from "
            "opposite-stance CA conflict annotations."
        ),
        gold_relations=explained_edges,
        hard_negatives=[],
    )


def _synthetic_cases() -> list[tuple[Scenario, GoldBlueprint]]:
    dropped = Scenario(
        scenario_id="synthetic_dropped_01",
        category=Category.SYNTHETIC_DROPPED,
        title="Dropped affordability argument",
        resolution="A city ought to abolish minimum parking requirements for new housing.",
        units=[
            Unit(id="U1", side="AFF", text="Parking minimums raise rents because every tenant pays for mandated construction, even without owning a car."),
            Unit(id="U2", side="AFF", text="Frequent transit means many downtown residents can reach work without private vehicles."),
            Unit(id="U3", side="NEG", text="Removing minimums can increase cruising when visitors search for scarce curb spaces."),
            Unit(id="U4", side="NEG", text="Late-night and weekend transit gaps undermine the claim that residents can reliably avoid cars."),
            Unit(id="U5", side="AFF", text="The proposal applies downtown, where all-night bus routes and weekend rail directly answer the service-gap objection."),
            Unit(id="U6", side="NEG", text="Delivery drivers also compete for limited curb access during business hours."),
        ],
    )
    dropped_gold = GoldBlueprint(
        scenario_id=dropped.scenario_id,
        category=dropped.category,
        design_intent="U1 is intentionally dropped while the exchange directly answers U2 and then U4.",
        gold_relations=[
            {
                "source": "U4",
                "target": "U2",
                "type": "responds_to",
                "explanation": "U4 directly attacks U2's transit-availability warrant.",
            },
            {
                "source": "U5",
                "target": "U4",
                "type": "responds_to",
                "explanation": "U5 directly answers U4 with a geographic and scheduling qualification.",
            },
        ],
        hard_negatives=[
            HardNegative(
                source="U3",
                target="U1",
                explanation="Cruising is an independent curb-space disadvantage, not an answer to U1's rent mechanism.",
            ),
            HardNegative(
                source="U6",
                target="U1",
                explanation="Delivery access is another independent curb-use claim; the affordability argument remains dropped.",
            ),
        ],
    )

    cross_application = Scenario(
        scenario_id="synthetic_cross_application_01",
        category=Category.SYNTHETIC_CROSS_APPLICATION,
        title="One rebuttal cross-applied to two claims",
        resolution="Employers ought to adopt a four-day workweek without reducing pay.",
        units=[
            Unit(id="U1", side="AFF", text="A shorter week reduces burnout because workers receive an additional recovery day."),
            Unit(id="U2", side="AFF", text="It also improves retention because employees value the additional recovery day."),
            Unit(id="U3", side="NEG", text="Both benefits assume a real day off, but fixed workloads are compressed into longer shifts and spill into the fifth day, so the promised recovery time disappears for U1 and U2 alike."),
            Unit(id="U4", side="NEG", text="Customer support may also become harder to schedule across five weekdays."),
            Unit(id="U5", side="AFF", text="Workload caps and a ban on fifth-day messaging preserve the recovery day, directly answering the compression objection."),
            Unit(id="U6", side="AFF", text="Rotating teams can maintain customer coverage on every weekday."),
        ],
    )
    cross_application_gold = GoldBlueprint(
        scenario_id=cross_application.scenario_id,
        category=cross_application.category,
        design_intent="U3 explicitly cross-applies one warrant to two earlier arguments.",
        gold_relations=[
            {
                "source": "U3",
                "target": "U1",
                "type": "responds_to",
                "explanation": "U3 directly attacks the recovery-time mechanism underlying U1.",
            },
            {
                "source": "U3",
                "target": "U2",
                "type": "responds_to",
                "explanation": "The same U3 warrant is explicitly applied to U2's retention claim.",
            },
            {
                "source": "U5",
                "target": "U3",
                "type": "responds_to",
                "explanation": "U5 directly answers the workload-compression objection in U3.",
            },
            {
                "source": "U6",
                "target": "U4",
                "type": "responds_to",
                "explanation": "U6 directly mitigates U4's weekday-coverage disadvantage.",
            },
        ],
        hard_negatives=[
            HardNegative(
                source="U4",
                target="U1",
                explanation="Customer-support scheduling is an independent disadvantage, not a burnout response.",
            )
        ],
    )
    return [(dropped, dropped_gold), (cross_application, cross_application_gold)]


def _load_jury_outcomes(path: Path) -> dict[str, JuryOutcome]:
    grouped: dict[str, list[JuryScore]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            stance = SourceStance(row["STANCE"].upper())
            grouped[row["DEBATE"]].append(
                JuryScore(
                    stance=stance,
                    score=float(row["SCORE"]),
                    thesis_solidity=float(row["THESIS SOLIDITY"]),
                    argumentation_quality=float(row["ARGUMENTATION QUALITY"]),
                    adaptability=float(row["ADAPTABILITY"]),
                )
            )
    outcomes: dict[str, JuryOutcome] = {}
    for debate_id, scores in grouped.items():
        by_stance = {score.stance.value: score.score for score in scores}
        winner = "TIE"
        if by_stance["FAVOUR"] > by_stance["AGAINST"]:
            winner = "FAVOUR"
        elif by_stance["AGAINST"] > by_stance["FAVOUR"]:
            winner = "AGAINST"
        outcomes[debate_id] = JuryOutcome(
            winner=winner,
            margin=round(abs(by_stance["FAVOUR"] - by_stance["AGAINST"]), 4),
            scores=scores,
        )
    return outcomes


def _relation_slots(fieldnames: list[str]) -> list[tuple[str, str]]:
    slots: list[tuple[int, str, str]] = []
    for field in fieldnames:
        match = re.fullmatch(r"RELATED ID(?:\.([0-9]+))?", field)
        if not match:
            continue
        index = int(match.group(1) or 0)
        type_field = "ARGUMENTAL RELATION  TYPE" + (f".{index}" if index else "")
        if type_field not in fieldnames:
            raise ValueError(f"missing relation type column for {field}")
        slots.append((index, field, type_field))
    return [(related, relation_type) for _, related, relation_type in sorted(slots)]


def _parse_related_ids(raw_value: str) -> list[int]:
    parsed: list[int] = []
    for token in raw_value.split(";"):
        token = token.strip()
        if not re.fullmatch(r"[0-9]+(?:\.0+)?", token):
            raise ValueError(f"unsupported related ID token: {token!r}")
        parsed.append(int(float(token)))
    return parsed


def _verify_file(path: Path, expected_md5: str) -> None:
    actual = hashlib.md5(path.read_bytes()).hexdigest()
    if actual != expected_md5:
        raise ValueError(f"checksum mismatch for {path.name}: expected {expected_md5}, got {actual}")


def _write_jsonl(path: Path, models: list[Any]) -> None:
    rows = [
        json.dumps(model.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))
        for model in models
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _build_benchmark_manifest(
    source_manifest: dict[str, Any],
    cases: list[Any],
    debate_stats: list[dict[str, Any]],
) -> dict[str, Any]:
    real_cases = [case for case in cases if case.scenario.category == Category.VIVESDEBATE]
    synthetic_cases = [case for case in cases if case.scenario.category != Category.VIVESDEBATE]
    source_units = sum(item["adu_count"] for item in debate_stats)
    converted_units = sum(len(case.scenario.units) for case in real_cases)
    source_relations = sum(item["source_relation_count"] for item in debate_stats)
    converted_relations = sum(len(case.scenario.source_relations) for case in real_cases)
    source_relation_slots = sum(item["source_relation_slot_count"] for item in debate_stats)
    converted_relation_slots = sum(
        len(
            {
                (relation.source, relation.source_column)
                for relation in case.scenario.source_relations
            }
            | {
                (issue.source, issue.source_column)
                for issue in case.scenario.source_annotation_issues
            }
        )
        for case in real_cases
    )
    all_edges_valid = all(
        int(edge.source[1:]) > int(edge.target[1:])
        and next(unit for unit in case.scenario.units if unit.id == edge.source).side
        != next(unit for unit in case.scenario.units if unit.id == edge.target).side
        for case in cases
        for edge in case.gold.gold_relations
    )
    return {
        "benchmark_version": 2,
        "source": {
            "corpus": "VivesDebate",
            "doi": source_manifest["doi"],
            "paper_doi": source_manifest["paper_doi"],
            "license": source_manifest["license"],
            "selected_language": source_manifest["selected_language"],
            "selection_rule": source_manifest["selection_rule"],
            "selected_debates": [item["debate_id"] for item in source_manifest["selected_debates"]],
            "excluded_before_tenth_selection": source_manifest["excluded_before_tenth_selection"],
        },
        "mapping": {
            "FAVOUR": "AFF",
            "AGAINST": "NEG",
            "ADU_EN": "unit.text",
            "chronological_ID": "unit.id as U<source ID>; gaps are preserved",
            "RA": "preserved as source_relations[type=inference]; not mapped to responds_to",
            "MA": "preserved as source_relations[type=rephrase]; not mapped to responds_to",
            "CA_same_stance": "preserved as source_relations[type=conflict]; not mapped to responds_to",
            "CA_opposite_stance": "mapped to responds_to and oriented later ADU -> earlier ADU",
            "jury_scores": "preserved with winner and score margin",
        },
        "validation": {
            "scenario_count": len(cases),
            "real_debate_count": len(real_cases),
            "synthetic_scenario_count": len(synthetic_cases),
            "source_adu_count": source_units,
            "converted_real_unit_count": converted_units,
            "all_source_adus_preserved": source_units == converted_units,
            "source_relation_count": source_relations,
            "converted_source_relation_count": converted_relations,
            "source_relation_slot_count": source_relation_slots,
            "converted_relation_slot_count": converted_relation_slots,
            "all_source_relation_slots_accounted_for": source_relation_slots == converted_relation_slots,
            "source_annotation_issue_count": sum(
                len(case.scenario.source_annotation_issues) for case in real_cases
            ),
            "all_selected_checksums_verified": True,
            "all_source_ids_chronological": True,
            "all_stances_present": True,
            "all_jury_outcomes_present": all(case.scenario.jury_outcome is not None for case in real_cases),
            "all_gold_edges_later_and_cross_stance": all_edges_valid,
            "gold_response_edge_count": sum(len(case.gold.gold_relations) for case in cases),
        },
        "debates": debate_stats,
    }


if __name__ == "__main__":
    main()

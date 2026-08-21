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
    BenchmarkSplit,
    Category,
    GoldBlueprint,
    HardNegative,
    HardNegativePhenomenon,
    JuryOutcome,
    JuryScore,
    Scenario,
    ScenarioPhenomenon,
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
CURATION_DIR = PROJECT_ROOT / "data" / "curation"
EXCERPT_BLUEPRINTS_PATH = CURATION_DIR / "vives_excerpt_blueprints.json"
REVIEWED_ENGLISH_PATH = CURATION_DIR / "vives_reviewed_english.json"

RESOLUTION = "Should surrogacy be legalised?"
RELATION_TYPES = {
    "RA": SourceRelationType.INFERENCE,
    "CA": SourceRelationType.CONFLICT,
    "MA": SourceRelationType.REPHRASE,
}
SIDE_BY_STANCE = {SourceStance.FAVOUR: "AFF", SourceStance.AGAINST: "NEG"}


def main() -> None:
    source_manifest = _load_json_without_duplicate_keys(SOURCE_MANIFEST_PATH)
    excerpt_blueprints = _load_json_without_duplicate_keys(EXCERPT_BLUEPRINTS_PATH)
    reviewed_english = _load_json_without_duplicate_keys(REVIEWED_ENGLISH_PATH)
    scenario_ids = [item["scenario_id"] for item in excerpt_blueprints]
    if len(scenario_ids) != len(set(scenario_ids)):
        raise ValueError("duplicate scenario_id in excerpt blueprints")
    blueprints_by_debate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in excerpt_blueprints:
        blueprints_by_debate[item["debate_id"]].append(item)
    expected_debates = {item["debate_id"] for item in source_manifest["selected_debates"]}
    if set(blueprints_by_debate) != expected_debates:
        raise ValueError("excerpt blueprints must cover exactly the selected debates")
    if any(len(items) != 3 for items in blueprints_by_debate.values()):
        raise ValueError("formal ablation requires exactly three excerpts per selected debate")
    if set(reviewed_english["debates"]) != expected_debates:
        raise ValueError("curated English must cover exactly the selected debates")
    for debate_id, items in blueprints_by_debate.items():
        selected_ids = {source_id for item in items for source_id in item["source_unit_ids"]}
        curated_ids = {int(source_id) for source_id in reviewed_english["debates"][debate_id]}
        if curated_ids != selected_ids:
            raise ValueError(f"{debate_id} curated English must exactly cover the union of excerpt IDs")
    _verify_file(SOURCE_DIR / source_manifest["evaluation_file"]["file"], source_manifest["evaluation_file"]["md5"])
    jury_by_debate = _load_jury_outcomes(SOURCE_DIR / source_manifest["evaluation_file"]["file"])

    scenarios: list[Scenario] = []
    blueprints: list[GoldBlueprint] = []
    debate_stats: list[dict[str, Any]] = []
    for selection in source_manifest["selected_debates"]:
        source_path = SOURCE_DIR / selection["file"]
        _verify_file(source_path, selection["md5"])
        for excerpt_blueprint in blueprints_by_debate[selection["debate_id"]]:
            scenario, blueprint, stats = _convert_debate(
                source_path,
                selection["debate_id"],
                selection["md5"],
                jury_by_debate[selection["debate_id"]],
                excerpt_blueprint,
                reviewed_english["debates"][selection["debate_id"]],
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
                "real_debates": len(
                    {
                        case.scenario.source.debate_id
                        for case in cases
                        if case.scenario.source is not None
                    }
                ),
                "real_scenarios": sum(
                    case.scenario.category == Category.VIVESDEBATE for case in cases
                ),
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
    excerpt_blueprint: dict[str, Any],
    reviewed_text: dict[str, str],
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
    row_by_id = {int(row["ID (Chronological)"]): row for row in rows}
    selected_ids = excerpt_blueprint["source_unit_ids"]
    if not 6 <= len(selected_ids) <= 12:
        raise ValueError(f"{debate_id} excerpt must contain 6-12 ADUs")
    if selected_ids != sorted(set(selected_ids)):
        raise ValueError(f"{debate_id} excerpt IDs must be unique and chronological")
    if not set(selected_ids).issubset(row_by_id):
        raise ValueError(f"{debate_id} excerpt references an unknown ADU")
    missing_curated_ids = set(selected_ids) - {int(source_id) for source_id in reviewed_text}
    if missing_curated_ids:
        raise ValueError(f"{debate_id} is missing curated English for {sorted(missing_curated_ids)}")

    relation_slots = _relation_slots(fieldnames)
    units: list[Unit] = []
    for source_id in selected_ids:
        row = row_by_id[source_id]
        source_id = int(row["ID (Chronological)"])
        stance_raw = row["TEAM STANCE"]
        if not stance_raw:
            raise ValueError(f"{source_path.name} U{source_id} has no source stance")
        stance = SourceStance(stance_raw.upper())
        units.append(
            Unit(
                id=f"U{source_id}",
                side=SIDE_BY_STANCE[stance],
                text=reviewed_text[str(source_id)],
                source_id=source_id,
                phase=row["TYPE (Part + Person)"] or None,
                argument_number=row["ARGUMENT NUMBER"] or None,
                source_stance=stance,
                text_ca=row["ADU_CAT"],
                text_es=row["ADU_ES"],
                text_en=row["ADU_EN"],
            )
        )

    all_unit_ids = {f"U{source_id}" for source_id in ids}
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
                if target not in all_unit_ids:
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
        scenario_id=excerpt_blueprint["scenario_id"],
        category=Category.VIVESDEBATE,
        title=f"VivesDebate {debate_id}: {excerpt_blueprint['title']}",
        resolution=RESOLUTION,
        split=BenchmarkSplit(excerpt_blueprint["split"]),
        phenomena=[ScenarioPhenomenon(value) for value in excerpt_blueprint["phenomena"]],
        units=units,
        source=SourceProvenance(
            corpus="VivesDebate",
            debate_id=debate_id,
            source_file=source_path.name,
            source_md5=source_md5,
            source_doi="10.5281/zenodo.6531487",
            license="CC BY-NC-SA 4.0",
            selected_language="CURATED_EN",
            model_text_method="manually_curated_from_ADU_ES_and_ADU_CAT",
            excerpt_source_ids=selected_ids,
            blueprint_file=str(EXCERPT_BLUEPRINTS_PATH.relative_to(PROJECT_ROOT)),
            translation_file=str(REVIEWED_ENGLISH_PATH.relative_to(PROJECT_ROOT)),
        ),
        source_relations=source_relations,
        source_annotation_issues=source_annotation_issues,
        jury_outcome=jury_outcome,
    )
    blueprint = _derive_gold(scenario, excerpt_blueprint)
    unit_by_id = {unit.id: unit for unit in units}
    excerpt_unit_ids = set(unit_by_id)
    source_side_by_id = {
        f"U{source_id}": SIDE_BY_STANCE[SourceStance(row_by_id[source_id]["TEAM STANCE"].upper())]
        for source_id in ids
    }
    source_id_set = set(ids)
    missing_ids = sorted(set(range(min(ids), max(ids) + 1)) - source_id_set)
    cross_conflicts = sum(
        relation.type == SourceRelationType.CONFLICT
        and source_side_by_id[relation.source] != source_side_by_id[relation.target]
        for relation in source_relations
    )
    same_conflicts = sum(
        relation.type == SourceRelationType.CONFLICT
        and source_side_by_id[relation.source] == source_side_by_id[relation.target]
        for relation in source_relations
    )
    stats = {
        "scenario_id": scenario.scenario_id,
        "split": scenario.split.value,
        "phenomena": [phenomenon.value for phenomenon in scenario.phenomena],
        "debate_id": debate_id,
        "source_file": source_path.name,
        "source_md5": source_md5,
        "adu_count": len(rows),
        "source_adu_count": len(rows),
        "excerpt_adu_count": len(units),
        "first_source_id": min(ids),
        "last_source_id": max(ids),
        "missing_source_ids": missing_ids,
        "missing_phase_annotations": sum(unit.phase is None for unit in units),
        "source_relation_count": len(source_relations),
        "excerpt_internal_source_relation_count": sum(
            relation.source in excerpt_unit_ids and relation.target in excerpt_unit_ids
            for relation in source_relations
        ),
        "excerpt_boundary_source_relation_count": sum(
            (relation.source in excerpt_unit_ids) != (relation.target in excerpt_unit_ids)
            for relation in source_relations
        ),
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


def _derive_gold(scenario: Scenario, excerpt_blueprint: dict[str, Any]) -> GoldBlueprint:
    unit_by_id = {unit.id: unit for unit in scenario.units}
    evidence: dict[tuple[str, str], list[SourceRelation]] = defaultdict(list)
    for relation in scenario.source_relations:
        if relation.type != SourceRelationType.CONFLICT:
            continue
        if relation.source not in unit_by_id or relation.target not in unit_by_id:
            continue
        if unit_by_id[relation.source].side == unit_by_id[relation.target].side:
            continue
        source_number = int(relation.source[1:])
        target_number = int(relation.target[1:])
        later = relation.source if source_number > target_number else relation.target
        earlier = relation.target if source_number > target_number else relation.source
        evidence[(later, earlier)].append(relation)

    configured_edges = {
        (f"U{item['source_id']}", f"U{item['target_id']}"): item["explanation"]
        for item in excerpt_blueprint["gold_edges"]
    }
    if set(evidence) != set(configured_edges):
        raise ValueError(
            f"{scenario.source.debate_id} curated gold does not exactly match internal opposing CA annotations: "
            f"configured={sorted(configured_edges)} derived={sorted(evidence)}"
        )

    explained_edges = []
    for (later, earlier), explanation in sorted(
        configured_edges.items(), key=lambda item: (int(item[0][0][1:]), int(item[0][1][1:]))
    ):
        relations = evidence[(later, earlier)]
        original = ", ".join(
            f"{relation.source_label} {relation.source}->{relation.target}"
            for relation in relations
        )
        explained_edges.append(
            {
                "source": later,
                "target": earlier,
                "type": "responds_to",
                "explanation": f"{explanation} Source basis: {original}.",
            }
        )
    hard_negatives = [
        HardNegative(
            source=f"U{item['source_id']}",
            target=f"U{item['target_id']}",
            phenomenon=HardNegativePhenomenon(item["phenomenon"]),
            explanation=item["explanation"],
        )
        for item in excerpt_blueprint["hard_negatives"]
    ]
    return GoldBlueprint(
        scenario_id=scenario.scenario_id,
        category=scenario.category,
        design_intent=(
            "A readable 6-12 ADU excerpt selected from the source annotations before English polishing. "
            "Gold responses are exactly the internal opposite-stance CA conflicts; original multilingual ADUs "
            "and the debate's complete relation graph remain preserved as provenance."
        ),
        gold_relations=explained_edges,
        hard_negatives=hard_negatives,
    )


def _synthetic_cases() -> list[tuple[Scenario, GoldBlueprint]]:
    dropped = Scenario(
        scenario_id="synthetic_dropped_01",
        category=Category.SYNTHETIC_DROPPED,
        title="Dropped affordability argument",
        resolution="A city ought to abolish minimum parking requirements for new housing.",
        split=BenchmarkSplit.DEVELOPMENT,
        phenomena=[ScenarioPhenomenon.DROPPED_ARGUMENT, ScenarioPhenomenon.TOPICAL_DISTRACTOR],
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
                phenomenon=HardNegativePhenomenon.TOPICAL_NONRESPONSE,
                explanation="Cruising is an independent curb-space disadvantage, not an answer to U1's rent mechanism.",
            ),
            HardNegative(
                source="U6",
                target="U1",
                phenomenon=HardNegativePhenomenon.INDEPENDENT_COUNTERARGUMENT,
                explanation="Delivery access is another independent curb-use claim; the affordability argument remains dropped.",
            ),
        ],
    )

    cross_application = Scenario(
        scenario_id="synthetic_cross_application_01",
        category=Category.SYNTHETIC_CROSS_APPLICATION,
        title="One rebuttal cross-applied to two claims",
        resolution="Employers ought to adopt a four-day workweek without reducing pay.",
        split=BenchmarkSplit.DEVELOPMENT,
        phenomena=[ScenarioPhenomenon.CROSS_APPLICATION, ScenarioPhenomenon.BRANCHING],
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
                phenomenon=HardNegativePhenomenon.INDEPENDENT_COUNTERARGUMENT,
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


def _load_json_without_duplicate_keys(path: Path) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)


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
    representative_stats = {
        item["debate_id"]: item
        for item in debate_stats
    }
    representative_cases = {
        case.scenario.source.debate_id: case
        for case in real_cases
    }
    source_units = sum(item["source_adu_count"] for item in representative_stats.values())
    model_facing_units = sum(len(case.scenario.units) for case in real_cases)
    source_relations = sum(item["source_relation_count"] for item in representative_stats.values())
    converted_relations = sum(
        len(case.scenario.source_relations) for case in representative_cases.values()
    )
    source_relation_slots = sum(
        item["source_relation_slot_count"] for item in representative_stats.values()
    )
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
        for case in representative_cases.values()
    )
    all_edges_valid = all(
        int(edge.source[1:]) > int(edge.target[1:])
        and next(unit for unit in case.scenario.units if unit.id == edge.source).side
        != next(unit for unit in case.scenario.units if unit.id == edge.target).side
        for case in cases
        for edge in case.gold.gold_relations
    )
    return {
        "benchmark_version": 4,
        "source": {
            "corpus": "VivesDebate",
            "doi": source_manifest["doi"],
            "paper_doi": source_manifest["paper_doi"],
            "license": source_manifest["license"],
            "raw_english_field": source_manifest["selected_language"],
            "model_facing_language": "CURATED_EN",
            "model_text_method": "manually_curated_from_ADU_ES_and_ADU_CAT",
            "selection_rule": source_manifest["selection_rule"],
            "excerpt_selection_rule": (
                "Three self-contained 6-12 ADU exchanges per selected debate, fixed from source CA annotations "
                "before curated English was written; the original pilot excerpt remains development data and eight "
                "previously unseen excerpts form the held-out test split."
            ),
            "selected_debates": [item["debate_id"] for item in source_manifest["selected_debates"]],
            "excluded_before_tenth_selection": source_manifest["excluded_before_tenth_selection"],
        },
        "mapping": {
            "FAVOUR": "AFF",
            "AGAINST": "NEG",
            "ADU_EN": "preserved verbatim as unit.text_en; not used directly as model input",
            "ADU_ES_and_ADU_CAT": "curated English rendering in unit.text with ADU boundaries unchanged",
            "chronological_ID": "unit.id as U<source ID>; gaps are preserved",
            "excerpt_selection": "three 6-12 ADU excerpts per debate; source IDs and chronological order are preserved",
            "split": "24 development scenarios and 8 held-out-test scenarios; every held-out scenario is real VivesDebate data",
            "phenomena": "scenario-level structural tags are fixed in the pre-text blueprint; hard-negative pairs carry a negative-type tag",
            "excerpt_topic": "a human-written topic label is included as shared context; it does not alter any ADU",
            "RA": "preserved as source_relations[type=inference]; not mapped to responds_to",
            "MA": "preserved as source_relations[type=rephrase]; not mapped to responds_to",
            "CA_same_stance": "preserved as source_relations[type=conflict]; not mapped to responds_to",
            "CA_opposite_stance": "mapped to responds_to and oriented later ADU -> earlier ADU",
            "jury_scores": "preserved with winner and score margin",
        },
        "validation": {
            "scenario_count": len(cases),
            "real_debate_count": len(representative_cases),
            "real_scenario_count": len(real_cases),
            "synthetic_scenario_count": len(synthetic_cases),
            "development_scenario_count": sum(
                case.scenario.split == BenchmarkSplit.DEVELOPMENT for case in cases
            ),
            "heldout_test_scenario_count": sum(
                case.scenario.split == BenchmarkSplit.HELDOUT_TEST for case in cases
            ),
            "scenarios_by_phenomenon": dict(
                sorted(
                    Counter(
                        phenomenon.value
                        for case in cases
                        for phenomenon in case.scenario.phenomena
                    ).items()
                )
            ),
            "hard_negatives_by_phenomenon": dict(
                sorted(
                    Counter(
                        pair.phenomenon.value
                        for case in cases
                        for pair in case.gold.hard_negatives
                    ).items()
                )
            ),
            "source_adu_count": source_units,
            "model_facing_real_unit_count": model_facing_units,
            "all_source_adus_preserved_in_raw_files": True,
            "all_real_excerpts_have_6_to_12_adus": all(
                6 <= len(case.scenario.units) <= 12 for case in real_cases
            ),
            "all_selected_adus_preserve_raw_multilingual_text": all(
                all((unit.text_ca, unit.text_es, unit.text_en)) for case in real_cases for unit in case.scenario.units
            ),
            "all_selected_adus_have_curated_english": all(
                bool(unit.text.strip()) for case in real_cases for unit in case.scenario.units
            ),
            "source_relation_count": source_relations,
            "converted_source_relation_count": converted_relations,
            "all_source_relations_preserved": (
                source_relations == converted_relations
                and all(
                    len(case.scenario.source_relations)
                    == next(
                        item["source_relation_count"]
                        for item in debate_stats
                        if item["scenario_id"] == case.scenario.scenario_id
                    )
                    for case in real_cases
                )
            ),
            "source_relation_slot_count": source_relation_slots,
            "converted_relation_slot_count": converted_relation_slots,
            "all_source_relation_slots_accounted_for": source_relation_slots == converted_relation_slots,
            "source_annotation_issue_count": sum(
                len(case.scenario.source_annotation_issues)
                for case in representative_cases.values()
            ),
            "all_selected_checksums_verified": True,
            "all_curation_json_keys_unique": True,
            "all_source_ids_chronological": True,
            "all_stances_present": True,
            "all_jury_outcomes_present": all(case.scenario.jury_outcome is not None for case in real_cases),
            "all_gold_edges_later_and_cross_stance": all_edges_valid,
            "all_real_excerpts_have_hard_negatives": all(case.gold.hard_negatives for case in real_cases),
            "gold_response_edge_count": sum(len(case.gold.gold_relations) for case in cases),
        },
        "debates": debate_stats,
    }


if __name__ == "__main__":
    main()

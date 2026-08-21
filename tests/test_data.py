import csv
import hashlib
import json
import re
from collections import Counter

from flowjudge.data import PROJECT_ROOT, load_benchmark
from flowjudge.schemas import (
    BenchmarkSplit,
    Category,
    HardNegativePhenomenon,
    ScenarioPhenomenon,
    Side,
    SourceRelationType,
)

RELATION_TYPE_BY_LABEL = {
    "RA": SourceRelationType.INFERENCE,
    "CA": SourceRelationType.CONFLICT,
    "MA": SourceRelationType.REPHRASE,
}


def test_benchmark_has_thirty_real_and_two_targeted_synthetic_scenarios() -> None:
    cases = load_benchmark()
    categories = Counter(case.scenario.category for case in cases)

    assert len(cases) == 32
    assert categories == {
        Category.VIVESDEBATE: 30,
        Category.SYNTHETIC_DROPPED: 1,
        Category.SYNTHETIC_CROSS_APPLICATION: 1,
    }
    real_cases = [case for case in cases if case.scenario.category == Category.VIVESDEBATE]
    assert sum(len(case.scenario.units) for case in real_cases) == 210
    assert all(6 <= len(case.scenario.units) <= 12 for case in real_cases)
    assert all(case.gold.hard_negatives for case in real_cases)
    assert Counter(case.scenario.source.debate_id for case in real_cases) == {
        f"Debate{number}": 3 for number in range(1, 11)
    }
    assert Counter(case.scenario.split for case in cases) == {
        BenchmarkSplit.DEVELOPMENT: 24,
        BenchmarkSplit.HELDOUT_TEST: 8,
    }
    assert all(
        case.scenario.category == Category.VIVESDEBATE
        for case in cases
        if case.scenario.split == BenchmarkSplit.HELDOUT_TEST
    )
    covered_phenomena = {
        phenomenon
        for case in cases
        for phenomenon in case.scenario.phenomena
    }
    assert covered_phenomena == set(ScenarioPhenomenon)
    topical_cases = [
        case
        for case in cases
        if any(
            pair.phenomenon == HardNegativePhenomenon.TOPICAL_NONRESPONSE
            for pair in case.gold.hard_negatives
        )
    ]
    assert len(topical_cases) == 8
    assert {case.scenario.split for case in topical_cases} == set(BenchmarkSplit)


def test_source_manifest_checksums_match_unchanged_csvs() -> None:
    source_dir = PROJECT_ROOT / "data" / "source" / "vivesdebate"
    manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    selected = [*manifest["selected_debates"], manifest["evaluation_file"]]

    for item in selected:
        digest = hashlib.md5((source_dir / item["file"]).read_bytes()).hexdigest()
        assert digest == item["md5"]


def test_every_selected_vives_adu_preserves_source_fields_and_uses_reviewed_english() -> None:
    source_dir = PROJECT_ROOT / "data" / "source" / "vivesdebate"
    blueprints = json.loads(
        (PROJECT_ROOT / "data" / "curation" / "vives_excerpt_blueprints.json").read_text(encoding="utf-8")
    )
    reviewed = json.loads(
        (PROJECT_ROOT / "data" / "curation" / "vives_reviewed_english.json").read_text(encoding="utf-8")
    )["debates"]
    blueprint_by_scenario = {item["scenario_id"]: item for item in blueprints}
    real_cases = [case for case in load_benchmark() if case.scenario.category == Category.VIVESDEBATE]

    for case in real_cases:
        scenario = case.scenario
        with (source_dir / scenario.source.source_file).open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows_by_id = {int(row["ID (Chronological)"]): row for row in rows}
        expected_ids = blueprint_by_scenario[scenario.scenario_id]["source_unit_ids"]
        assert scenario.source.excerpt_source_ids == expected_ids
        assert [unit.source_id for unit in scenario.units] == expected_ids
        assert scenario.source.selected_language == "CURATED_EN"
        assert scenario.source.model_text_method == "manually_curated_from_ADU_ES_and_ADU_CAT"
        for unit in scenario.units:
            row = rows_by_id[unit.source_id]
            source_id = int(row["ID (Chronological)"])
            stance = row["TEAM STANCE"].upper()
            assert unit.id == f"U{source_id}"
            assert unit.source_id == source_id
            assert unit.source_stance.value == stance
            assert unit.side == (Side.AFF if stance == "FAVOUR" else Side.NEG)
            assert unit.text == reviewed[scenario.source.debate_id][str(source_id)]
            assert unit.text_en == row["ADU_EN"]
            assert unit.text_ca == row["ADU_CAT"]
            assert unit.text_es == row["ADU_ES"]
            assert unit.phase == (row["TYPE (Part + Person)"] or None)
            assert unit.argument_number == (row["ARGUMENT NUMBER"] or None)


def test_real_gold_is_exactly_opposite_stance_conflict_oriented_later_to_earlier() -> None:
    real_cases = [case for case in load_benchmark() if case.scenario.category == Category.VIVESDEBATE]

    for case in real_cases:
        unit_by_id = {unit.id: unit for unit in case.scenario.units}
        expected = set()
        for relation in case.scenario.source_relations:
            if relation.type != SourceRelationType.CONFLICT:
                continue
            if relation.source not in unit_by_id or relation.target not in unit_by_id:
                continue
            if unit_by_id[relation.source].side == unit_by_id[relation.target].side:
                continue
            ordered = sorted((relation.source, relation.target), key=lambda value: int(value[1:]))
            expected.add((ordered[1], ordered[0], "responds_to"))
        actual = {(edge.source, edge.target, edge.type) for edge in case.gold.gold_relations}
        assert actual == expected


def test_every_valid_raw_relation_target_is_preserved_and_malformed_slots_are_explicit() -> None:
    source_dir = PROJECT_ROOT / "data" / "source" / "vivesdebate"
    real_cases = [case for case in load_benchmark() if case.scenario.category == Category.VIVESDEBATE]

    for case in real_cases:
        scenario = case.scenario
        with (source_dir / scenario.source.source_file).open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fieldnames = reader.fieldnames or []
        unit_ids = {f'U{int(row["ID (Chronological)"])}' for row in rows}
        relation_slots = []
        for field in fieldnames:
            match = re.fullmatch(r"RELATED ID(?:\.([0-9]+))?", field)
            if match:
                suffix = f".{match.group(1)}" if match.group(1) else ""
                relation_slots.append((field, f"ARGUMENTAL RELATION  TYPE{suffix}"))

        expected_relations = Counter()
        expected_issues = Counter()
        for row in rows:
            source = f'U{int(row["ID (Chronological)"])}'
            for related_column, type_column in relation_slots:
                raw_related = row.get(related_column, "") or ""
                raw_type = row.get(type_column, "") or ""
                if not raw_related and not raw_type:
                    continue
                issue_base = (source, related_column, type_column, raw_related, raw_type)
                if not raw_related or not raw_type:
                    expected_issues[(*issue_base, "incomplete_pair")] += 1
                    continue
                normalized = raw_type.upper()
                if normalized not in RELATION_TYPE_BY_LABEL:
                    expected_issues[(*issue_base, "unknown_relation_type")] += 1
                    continue
                for token in raw_related.split(";"):
                    assert re.fullmatch(r"\s*[0-9]+(?:\.0+)?\s*", token)
                    target_number = int(float(token.strip()))
                    target = f"U{target_number}"
                    if target not in unit_ids:
                        expected_issues[(*issue_base, "unknown_target")] += 1
                    else:
                        expected_relations[
                            (source, target, RELATION_TYPE_BY_LABEL[normalized], raw_type, related_column)
                        ] += 1

        actual_relations = Counter(
            (relation.source, relation.target, relation.type, relation.source_label, relation.source_column)
            for relation in scenario.source_relations
        )
        actual_issues = Counter(
            (
                issue.source,
                issue.source_column,
                issue.type_column,
                issue.raw_related_id,
                issue.raw_relation_type,
                issue.reason,
            )
            for issue in scenario.source_annotation_issues
        )
        assert actual_relations == expected_relations
        assert actual_issues == expected_issues


def test_manifest_accounts_for_source_defects_without_imputation() -> None:
    manifest = json.loads((PROJECT_ROOT / "data" / "benchmark_manifest.json").read_text(encoding="utf-8"))
    validation = manifest["validation"]

    assert validation["all_source_adus_preserved_in_raw_files"] is True
    assert validation["all_selected_adus_preserve_raw_multilingual_text"] is True
    assert validation["all_selected_adus_have_curated_english"] is True
    assert validation["all_curation_json_keys_unique"] is True
    assert validation["all_real_excerpts_have_6_to_12_adus"] is True
    assert validation["all_real_excerpts_have_hard_negatives"] is True
    assert validation["all_source_relations_preserved"] is True
    assert validation["all_source_relation_slots_accounted_for"] is True
    assert validation["all_gold_edges_later_and_cross_stance"] is True
    assert validation["all_jury_outcomes_present"] is True
    assert validation["source_annotation_issue_count"] == 1
    assert validation["source_adu_count"] == 2932
    assert validation["real_debate_count"] == 10
    assert validation["real_scenario_count"] == 30
    assert validation["development_scenario_count"] == 24
    assert validation["heldout_test_scenario_count"] == 8
    assert validation["hard_negatives_by_phenomenon"]["topical_nonresponse"] == 8
    assert validation["model_facing_real_unit_count"] == 210
    representative_by_debate = {}
    for case in load_benchmark():
        if case.scenario.category == Category.VIVESDEBATE:
            representative_by_debate[case.scenario.source.debate_id] = case
    assert sum(
        len(case.scenario.source_annotation_issues)
        for case in representative_by_debate.values()
    ) == 1


def test_jury_scores_and_outcomes_match_source_evaluation() -> None:
    source_dir = PROJECT_ROOT / "data" / "source" / "vivesdebate"
    with (source_dir / "VivesDebate_eval.csv").open(encoding="utf-8", newline="") as handle:
        evaluation_rows = list(csv.DictReader(handle))
    rows_by_debate = {
        debate_id: [row for row in evaluation_rows if row["DEBATE"] == debate_id]
        for debate_id in {row["DEBATE"] for row in evaluation_rows}
    }

    for case in load_benchmark():
        if case.scenario.category != Category.VIVESDEBATE:
            continue
        scenario = case.scenario
        assert scenario.source.source_doi == "10.5281/zenodo.6531487"
        source_rows = rows_by_debate[scenario.source.debate_id]
        converted = {score.stance.value: score for score in scenario.jury_outcome.scores}
        for row in source_rows:
            score = converted[row["STANCE"].upper()]
            assert score.score == float(row["SCORE"])
            assert score.thesis_solidity == float(row["THESIS SOLIDITY"])
            assert score.argumentation_quality == float(row["ARGUMENTATION QUALITY"])
            assert score.adaptability == float(row["ADAPTABILITY"])


def test_synthetic_cases_are_concise_and_annotate_hard_negatives() -> None:
    synthetic = [case for case in load_benchmark() if case.scenario.category != Category.VIVESDEBATE]

    assert len(synthetic) == 2
    assert all(6 <= len(case.scenario.units) <= 8 for case in synthetic)
    assert all(case.gold.hard_negatives for case in synthetic)

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from .schemas import BenchmarkCase, GoldBlueprint, HardNegativePhenomenon, Scenario

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENARIOS_PATH = PROJECT_ROOT / "data" / "benchmark_scenarios.jsonl"
DEFAULT_GOLD_PATH = PROJECT_ROOT / "data" / "benchmark_gold.jsonl"

ModelT = TypeVar("ModelT", bound=BaseModel)


def _load_jsonl(path: Path, model: type[ModelT]) -> list[ModelT]:
    rows: list[ModelT] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(model.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"invalid {path.name} line {line_number}: {exc}") from exc
    return rows


def load_benchmark(
    scenarios_path: Path = DEFAULT_SCENARIOS_PATH,
    gold_path: Path = DEFAULT_GOLD_PATH,
) -> list[BenchmarkCase]:
    scenarios = _load_jsonl(scenarios_path, Scenario)
    blueprints = _load_jsonl(gold_path, GoldBlueprint)
    scenario_by_id = {scenario.scenario_id: scenario for scenario in scenarios}
    blueprint_by_id = {blueprint.scenario_id: blueprint for blueprint in blueprints}

    if len(scenario_by_id) != len(scenarios):
        raise ValueError(f"duplicate scenario_id in {scenarios_path.name}")
    if len(blueprint_by_id) != len(blueprints):
        raise ValueError(f"duplicate scenario_id in {gold_path.name}")
    if scenario_by_id.keys() != blueprint_by_id.keys():
        missing_gold = sorted(scenario_by_id.keys() - blueprint_by_id.keys())
        missing_scenarios = sorted(blueprint_by_id.keys() - scenario_by_id.keys())
        raise ValueError(f"benchmark files do not align: missing_gold={missing_gold}, missing_scenarios={missing_scenarios}")

    cases: list[BenchmarkCase] = []
    for scenario in scenarios:
        blueprint = blueprint_by_id[scenario.scenario_id]
        if scenario.category != blueprint.category:
            raise ValueError(f"category mismatch for {scenario.scenario_id}")
        _validate_annotations(scenario, blueprint)
        cases.append(BenchmarkCase(scenario=scenario, gold=blueprint))
    return cases


def _validate_annotations(scenario: Scenario, blueprint: GoldBlueprint) -> None:
    unit_by_id = {unit.id: unit for unit in scenario.units}
    seen_pairs: set[tuple[str, str]] = set()
    for annotation in [*blueprint.gold_relations, *blueprint.hard_negatives]:
        pair = (annotation.source, annotation.target)
        if pair in seen_pairs:
            raise ValueError(f"duplicate annotated pair {pair} in {scenario.scenario_id}")
        seen_pairs.add(pair)
        if annotation.source not in unit_by_id or annotation.target not in unit_by_id:
            raise ValueError(f"unknown unit in annotated pair {pair} for {scenario.scenario_id}")
        if int(annotation.source[1:]) <= int(annotation.target[1:]):
            raise ValueError(f"response pair must point backward in {scenario.scenario_id}: {pair}")

    for edge in blueprint.gold_relations:
        if unit_by_id[edge.source].side == unit_by_id[edge.target].side:
            raise ValueError(f"gold response must cross sides in {scenario.scenario_id}: {edge.source}->{edge.target}")
    for pair in blueprint.hard_negatives:
        same_side = unit_by_id[pair.source].side == unit_by_id[pair.target].side
        if pair.phenomenon == HardNegativePhenomenon.SAME_SIDE_EXTENSION and not same_side:
            raise ValueError(
                f"same-side extension must stay on one side in {scenario.scenario_id}: "
                f"{pair.source}->{pair.target}"
            )
        if pair.phenomenon in {
            HardNegativePhenomenon.TOPICAL_NONRESPONSE,
            HardNegativePhenomenon.INDEPENDENT_COUNTERARGUMENT,
        } and same_side:
            raise ValueError(
                f"opposing-side hard negative cannot stay on one side in {scenario.scenario_id}: "
                f"{pair.source}->{pair.target}"
            )


def eligible_response_pairs(scenario: Scenario) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for source_index, source in enumerate(scenario.units):
        for target in scenario.units[:source_index]:
            if source.side != target.side:
                pairs.add((source.id, target.id))
    return pairs


def render_transcript(scenario: Scenario) -> str:
    lines = [f"Resolution: {scenario.resolution}", f"Excerpt topic: {scenario.title}"]
    lines.extend(f"{unit.id} [{unit.side.value}]: {unit.text}" for unit in scenario.units)
    return "\n".join(lines)

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from .data import PROJECT_ROOT, load_benchmark
from .schemas import Category

DEFAULT_REVIEW_PATH = PROJECT_ROOT / "docs" / "pilot_review.html"
BENCHMARK_MANIFEST_PATH = PROJECT_ROOT / "data" / "benchmark_manifest.json"


def generate_review_page(output_path: Path = DEFAULT_REVIEW_PATH) -> Path:
    cases = load_benchmark()
    manifest = json.loads(BENCHMARK_MANIFEST_PATH.read_text(encoding="utf-8"))
    real_cases = [case for case in cases if case.scenario.category == Category.VIVESDEBATE]
    synthetic_cases = [case for case in cases if case.scenario.category != Category.VIVESDEBATE]
    mapping_rows = "".join(
        f"<tr><th>{_e(source)}</th><td>{_e(target)}</td></tr>"
        for source, target in manifest["mapping"].items()
    )
    validation_rows = "".join(
        _validation_row(name, value) for name, value in manifest["validation"].items()
    )
    excerpt_rows = "".join(_excerpt_row(item) for item in manifest["debates"])
    cases_by_id = {case.scenario.scenario_id: case for case in real_cases}
    examples = "".join(
        _example(cases_by_id[scenario_id])
        for scenario_id in ("vives_debate1", "vives_debate8")
    )
    synthetic_rows = "".join(
        f"<tr><td>{_e(case.scenario.scenario_id)}</td><td>{_e(case.scenario.title)}</td>"
        f"<td>{_e(', '.join(item.value for item in case.scenario.phenomena))}</td>"
        f"<td>{len(case.scenario.units)}</td><td>{len(case.gold.gold_relations)}</td>"
        f"<td>{_e(case.gold.design_intent)}</td></tr>"
        for case in synthetic_cases
    )
    skipped = "".join(
        f"<li><strong>{_e(item['debate_id'])}:</strong> {_e(item['reason'])}</li>"
        for item in manifest["source"]["excluded_before_tenth_selection"]
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FlowJudge benchmark conversion review</title>
  <style>
    :root {{ color-scheme: light; --ink:#18212b; --muted:#5d6975; --line:#d9e0e6; --paper:#fff; --wash:#f3f6f8; --pass:#e5f6ec; --warn:#fff3d6; --aff:#e9f3ff; --neg:#fff0e8; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; color:var(--ink); background:var(--wash); font:15px/1.48 ui-sans-serif,system-ui,sans-serif; }}
    header {{ padding:2.5rem max(1.25rem,calc((100vw - 1120px)/2)); background:#18212b; color:white; }}
    header h1 {{ margin:0 0 .45rem; font-size:2rem; }} header p {{ margin:0; max-width:850px; color:#dbe3e9; }}
    main {{ max-width:1120px; margin:2rem auto; padding:0 1.25rem 3rem; }}
    section {{ margin:0 0 1.4rem; padding:1.25rem; border:1px solid var(--line); border-radius:12px; background:var(--paper); }}
    h2 {{ margin:0 0 .7rem; font-size:1.3rem; }} h3 {{ margin:1rem 0 .5rem; font-size:1rem; }}
    p {{ margin:.4rem 0 .8rem; }} .muted {{ color:var(--muted); }}
    .stats {{ display:grid; grid-template-columns:repeat(5,1fr); gap:.75rem; }}
    .stat {{ padding:.9rem; border-radius:9px; background:var(--wash); }} .stat strong {{ display:block; font-size:1.55rem; }}
    table {{ width:100%; border-collapse:collapse; font-size:.91rem; }} th,td {{ padding:.55rem; border:1px solid var(--line); text-align:left; vertical-align:top; }} th {{ background:var(--wash); }}
    .pass {{ background:var(--pass); }} .warn {{ background:var(--warn); }}
    .examples {{ display:grid; grid-template-columns:1fr; gap:1rem; }} .example {{ padding:1rem; border:1px solid var(--line); border-radius:9px; }}
    .edge {{ margin:.7rem 0; padding:.7rem; border-left:4px solid #7a8792; background:var(--wash); }}
    .adu {{ margin:.3rem 0; padding:.55rem; border-radius:7px; }} .adu.aff {{ background:var(--aff); }} .adu.neg {{ background:var(--neg); }}
    .adu details {{ margin-top:.35rem; color:var(--muted); }} .adu summary {{ cursor:pointer; }}
    .check {{ padding:.8rem; border-left:5px solid #2f855a; background:var(--pass); }}
    code {{ font-family:ui-monospace,SFMono-Regular,monospace; }} a {{ color:#155d91; }}
    @media(max-width:850px) {{ .stats {{ grid-template-columns:1fr 1fr; }} .wide {{ overflow-x:auto; }} }}
    @media(max-width:560px) {{ .stats {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <header>
    <h1>FlowJudge benchmark conversion review</h1>
    <p>A compact review of 30 readable VivesDebate excerpts plus two targeted synthetic cases. Original source IDs, annotations, multilingual ADUs, full relation graphs, and jury outcomes remain preserved.</p>
  </header>
  <main>
    <section>
      <h2>What you need to check</h2>
      <p class="check"><strong>No full manual audit is required.</strong> The mapping and source checks below are automatic. The two complete examples are included only as a readability and conversion spot-check.</p>
      <div class="stats">
        <div class="stat"><strong>{manifest['validation']['real_debate_count']}</strong>real debates</div>
        <div class="stat"><strong>{manifest['validation']['real_scenario_count']}</strong>real excerpts</div>
        <div class="stat"><strong>{manifest['validation']['heldout_test_scenario_count']}</strong>held-out cases</div>
        <div class="stat"><strong>{manifest['validation']['model_facing_real_unit_count']}</strong>readable excerpt ADUs</div>
        <div class="stat"><strong>{manifest['validation']['gold_response_edge_count']}</strong>FlowJudge gold edges</div>
      </div>
      <p class="muted">Source: <a href="https://doi.org/10.5281/zenodo.6531487">VivesDebate version 3 Zenodo release</a>, licensed CC BY-NC-SA 4.0. Model text is a curated English rendering from the clearer Spanish and Catalan fields. The original machine-translated English remains preserved and is visible under each example unit.</p>
    </section>
    <section>
      <h2>Explicit mapping rules</h2>
      <div class="wide"><table><tbody>{mapping_rows}</tbody></table></div>
      <p><strong>Important:</strong> “CA is symmetric” is a FlowJudge conversion convention used only to orient a conflict from the later ADU to the earlier ADU. The original relation direction and label remain preserved separately.</p>
    </section>
    <section>
      <h2>Automated validation</h2>
      <div class="wide"><table><thead><tr><th>Check or count</th><th>Result</th></tr></thead><tbody>{validation_rows}</tbody></table></div>
      <h3>Known source issue handling</h3>
      <p>{manifest['validation']['source_annotation_issue_count']} malformed source relation annotations are retained verbatim as structured issues and excluded from gold derivation. Nothing is imputed.</p>
      <ul>{skipped}</ul>
    </section>
    <section>
      <h2>Converted excerpt inventory</h2>
      <div class="wide"><table><thead><tr><th>Scenario</th><th>Debate</th><th>Split</th><th>Phenomena</th><th>Source / excerpt ADUs</th><th>Internal relations</th><th>Gold responses</th><th>Jury winner</th></tr></thead><tbody>{excerpt_rows}</tbody></table></div>
    </section>
    <section>
      <h2>Two representative complete excerpts</h2>
      <p class="muted">These are exactly the topic labels and units the tested models receive. Each unit remains one original VivesDebate ADU; only its English rendering was repaired. Expand “Source text” only if you want to compare it with VivesDebate's Spanish and raw English.</p>
      <div class="examples">{examples}</div>
    </section>
    <section>
      <h2>Two retained synthetic cases</h2>
      <div class="wide"><table><thead><tr><th>ID</th><th>Title</th><th>Phenomena</th><th>Units</th><th>Gold edges</th><th>Purpose</th></tr></thead><tbody>{synthetic_rows}</tbody></table></div>
    </section>
  </main>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
    return output_path


def _validation_row(name: str, value: Any) -> str:
    status_class = "pass" if value is True else "warn" if value is False else ""
    shown = "PASS" if value is True else "FAIL" if value is False else value
    return f'<tr><th>{_e(name.replace("_", " "))}</th><td class="{status_class}">{_e(shown)}</td></tr>'


def _excerpt_row(item: dict[str, Any]) -> str:
    return (
        f"<tr><td>{_e(item['scenario_id'])}</td><td>{_e(item['debate_id'])}</td>"
        f"<td>{_e(item['split'])}</td><td>{_e(', '.join(item['phenomena']))}</td>"
        f"<td>{item['source_adu_count']} / {item['excerpt_adu_count']}</td>"
        f"<td>{item['excerpt_internal_source_relation_count']}</td>"
        f"<td>{item['flowjudge_response_edges']}</td>"
        f"<td>{_e(item['jury_winner'])} (+{item['jury_margin']:.4f})</td></tr>"
    )


def _example(case: Any) -> str:
    scenario = case.scenario
    rendered_units = "".join(
        f'<div class="adu {unit.side.value.lower()}"><strong>{_e(unit.id)} [{_e(unit.side.value)}]</strong> '
        f'{_e(unit.text)}<details><summary>Source text</summary><div><strong>Spanish:</strong> {_e(unit.text_es)}</div>'
        f'<div><strong>Raw ADU_EN:</strong> {_e(unit.text_en)}</div></details></div>'
        for unit in scenario.units
    )
    rendered_edges = "".join(
        f'<div class="edge"><strong>{_e(edge.source)} → {_e(edge.target)}</strong>'
        f'<div>{_e(edge.explanation)}</div></div>'
        for edge in case.gold.gold_relations
    )
    rendered_negatives = "".join(
        f'<li><strong>{_e(item.source)} → {_e(item.target)}</strong> '
        f'({_e(item.phenomenon.value)}): {_e(item.explanation)}</li>'
        for item in case.gold.hard_negatives
    )
    jury = scenario.jury_outcome
    return (
        f'<article class="example"><h3>{_e(scenario.title)}</h3>'
        f'<p>{len(scenario.units)} model-facing ADUs · complete source relation graph preserved · '
        f'split: {_e(scenario.split.value)} · phenomena: '
        f'{_e(", ".join(item.value for item in scenario.phenomena))} · '
        f'jury: {_e(jury.winner)}</p><h3>Complete transcript</h3>{rendered_units}'
        f'<h3>Gold response arrows</h3>{rendered_edges}'
        f'<h3>Important non-response pairs</h3><ul>{rendered_negatives}</ul></article>'
    )


def _e(value: object) -> str:
    return html.escape(str(value), quote=True)

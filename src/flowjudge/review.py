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
    debate_rows = "".join(_debate_row(item) for item in manifest["debates"])
    examples = "".join(_example(case) for case in real_cases[:2])
    synthetic_rows = "".join(
        f"<tr><td>{_e(case.scenario.scenario_id)}</td><td>{_e(case.scenario.title)}</td>"
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
    .stats {{ display:grid; grid-template-columns:repeat(4,1fr); gap:.75rem; }}
    .stat {{ padding:.9rem; border-radius:9px; background:var(--wash); }} .stat strong {{ display:block; font-size:1.55rem; }}
    table {{ width:100%; border-collapse:collapse; font-size:.91rem; }} th,td {{ padding:.55rem; border:1px solid var(--line); text-align:left; vertical-align:top; }} th {{ background:var(--wash); }}
    .pass {{ background:var(--pass); }} .warn {{ background:var(--warn); }}
    .examples {{ display:grid; grid-template-columns:1fr 1fr; gap:1rem; }} .example {{ padding:1rem; border:1px solid var(--line); border-radius:9px; }}
    .edge {{ margin:.7rem 0; padding:.7rem; border-left:4px solid #7a8792; background:var(--wash); }}
    .adu {{ margin:.3rem 0; padding:.55rem; border-radius:7px; }} .adu.aff {{ background:var(--aff); }} .adu.neg {{ background:var(--neg); }}
    code {{ font-family:ui-monospace,SFMono-Regular,monospace; }} a {{ color:#155d91; }}
    @media(max-width:850px) {{ .stats,.examples {{ grid-template-columns:1fr 1fr; }} .wide {{ overflow-x:auto; }} }}
    @media(max-width:560px) {{ .stats,.examples {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <header>
    <h1>FlowJudge benchmark conversion review</h1>
    <p>A compact provenance and validation report for 10 complete VivesDebate conversions plus two targeted synthetic cases. This page is not a case-by-case manual labeling queue.</p>
  </header>
  <main>
    <section>
      <h2>What changed</h2>
      <div class="stats">
        <div class="stat"><strong>{manifest['validation']['real_debate_count']}</strong>real debates</div>
        <div class="stat"><strong>{manifest['validation']['source_adu_count']}</strong>preserved ADUs</div>
        <div class="stat"><strong>{manifest['validation']['converted_source_relation_count']}</strong>valid source relations</div>
        <div class="stat"><strong>{manifest['validation']['gold_response_edge_count']}</strong>FlowJudge gold edges</div>
      </div>
      <p class="muted">Source: <a href="https://doi.org/10.5281/zenodo.6531487">VivesDebate version 3 Zenodo release</a>, English machine-translation column <code>ADU_EN</code>, licensed CC BY-NC-SA 4.0. Jury results are preserved for every selected debate.</p>
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
      <h2>Converted debate inventory</h2>
      <div class="wide"><table><thead><tr><th>Debate</th><th>ADUs</th><th>RA / CA / MA</th><th>Issues</th><th>Gold responses</th><th>Jury winner</th></tr></thead><tbody>{debate_rows}</tbody></table></div>
    </section>
    <section>
      <h2>Two representative converted examples</h2>
      <p class="muted">Only a few mapped edges are shown here. The JSONL retains every ADU, all valid source relations, source issues, and complete jury metadata.</p>
      <div class="examples">{examples}</div>
    </section>
    <section>
      <h2>Two retained synthetic cases</h2>
      <div class="wide"><table><thead><tr><th>ID</th><th>Phenomenon</th><th>Units</th><th>Gold edges</th><th>Purpose</th></tr></thead><tbody>{synthetic_rows}</tbody></table></div>
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


def _debate_row(item: dict[str, Any]) -> str:
    relations = item["source_relations_by_type"]
    relation_summary = f"{relations.get('inference', 0)} / {relations.get('conflict', 0)} / {relations.get('rephrase', 0)}"
    return (
        f"<tr><td>{_e(item['debate_id'])}</td><td>{item['adu_count']}</td><td>{relation_summary}</td>"
        f"<td>{item['source_annotation_issue_count']}</td><td>{item['flowjudge_response_edges']}</td>"
        f"<td>{_e(item['jury_winner'])} (+{item['jury_margin']:.4f})</td></tr>"
    )


def _example(case: Any) -> str:
    scenario = case.scenario
    unit_by_id = {unit.id: unit for unit in scenario.units}
    edges = case.gold.gold_relations[:3]
    rendered_edges = "".join(
        f'<div class="edge"><strong>{_e(edge.source)} → {_e(edge.target)}</strong>'
        f'<div class="adu {unit_by_id[edge.target].side.value.lower()}"><strong>{_e(edge.target)} [{_e(unit_by_id[edge.target].side.value)}]</strong> {_e(unit_by_id[edge.target].text)}</div>'
        f'<div class="adu {unit_by_id[edge.source].side.value.lower()}"><strong>{_e(edge.source)} [{_e(unit_by_id[edge.source].side.value)}]</strong> {_e(unit_by_id[edge.source].text)}</div>'
        f'<div class="muted">{_e(edge.explanation)}</div></div>'
        for edge in edges
    )
    jury = scenario.jury_outcome
    return (
        f'<article class="example"><h3>{_e(scenario.title)}</h3>'
        f'<p>{len(scenario.units)} ADUs · {len(scenario.source_relations)} valid source relations · '
        f'{len(scenario.source_annotation_issues)} source issues · jury: {_e(jury.winner)}</p>{rendered_edges}</article>'
    )


def _e(value: object) -> str:
    return html.escape(str(value), quote=True)

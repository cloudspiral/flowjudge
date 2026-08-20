from __future__ import annotations

import html
from pathlib import Path

from .data import PROJECT_ROOT, load_pilot

DEFAULT_REVIEW_PATH = PROJECT_ROOT / "docs" / "pilot_review.html"


def generate_review_page(output_path: Path = DEFAULT_REVIEW_PATH) -> Path:
    cases = load_pilot()
    cards = "\n".join(_render_case(case) for case in cases)
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FlowJudge Pilot Gold Review</title>
  <style>
    :root {{ color-scheme: light; --ink: #17202a; --muted: #59636e; --line: #d8dee4; --paper: #fff; --wash: #f4f7f9; --aff: #e9f3ff; --neg: #fff0e8; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; color: var(--ink); background: var(--wash); font: 15px/1.45 ui-sans-serif, system-ui, sans-serif; }}
    header {{ padding: 2.5rem max(1.25rem, calc((100vw - 1180px) / 2)); background: #17202a; color: white; }}
    header h1 {{ margin: 0 0 .4rem; font-size: 2rem; }}
    header p {{ margin: 0; max-width: 780px; color: #dce4ea; }}
    main {{ max-width: 1180px; margin: 2rem auto; padding: 0 1.25rem 3rem; }}
    .case {{ margin: 0 0 1.5rem; padding: 1.25rem; border: 1px solid var(--line); border-radius: 12px; background: var(--paper); box-shadow: 0 2px 10px #16202a0a; }}
    .case h2 {{ margin: 0; font-size: 1.25rem; }}
    .meta {{ margin: .2rem 0 1rem; color: var(--muted); }}
    .grid {{ display: grid; grid-template-columns: minmax(0, 1.1fr) minmax(330px, .9fr); gap: 1.25rem; align-items: start; }}
    .resolution, .intent {{ padding: .75rem; border-radius: 8px; background: var(--wash); }}
    .unit {{ display: grid; grid-template-columns: 3rem 3.2rem 1fr; gap: .55rem; margin-top: .5rem; padding: .65rem; border-radius: 8px; }}
    .unit.aff {{ background: var(--aff); }} .unit.neg {{ background: var(--neg); }}
    .uid, .side {{ font-weight: 700; }}
    h3 {{ margin: 1rem 0 .45rem; font-size: 1rem; }}
    table {{ width: 100%; border-collapse: collapse; font-size: .92rem; }}
    th, td {{ padding: .55rem; border: 1px solid var(--line); text-align: left; vertical-align: top; }}
    th {{ background: var(--wash); }}
    .empty {{ color: var(--muted); font-style: italic; }}
    @media (max-width: 800px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <header>
    <h1>FlowJudge pilot gold review</h1>
    <p>12 manually reviewable scenarios, two per category. Gold edges are later-to-earlier, cross-side direct responses. Hard negatives identify tempting pairs that must remain absent.</p>
  </header>
  <main>{cards}</main>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
    return output_path


def _render_case(case: object) -> str:
    scenario = case.scenario
    gold = case.gold
    units = "".join(
        f'<div class="unit {unit.side.value.lower()}"><span class="uid">{_e(unit.id)}</span>'
        f'<span class="side">{_e(unit.side.value)}</span><span>{_e(unit.text)}</span></div>'
        for unit in scenario.units
    )
    if gold.gold_relations:
        edge_rows = "".join(
            f"<tr><td>{_e(edge.source)} → {_e(edge.target)}</td><td>{_e(edge.explanation)}</td></tr>"
            for edge in gold.gold_relations
        )
    else:
        edge_rows = '<tr><td colspan="2" class="empty">No gold response edges.</td></tr>'
    negative_rows = "".join(
        f"<tr><td>{_e(pair.source)} ↛ {_e(pair.target)}</td><td>{_e(pair.explanation)}</td></tr>"
        for pair in gold.hard_negatives
    )
    return f"""
    <section class="case" id="{_e(scenario.scenario_id)}">
      <h2>{_e(scenario.title)}</h2>
      <div class="meta">{_e(scenario.scenario_id)} · {_e(scenario.category.value)} · {len(scenario.units)} units</div>
      <div class="grid">
        <div>
          <div class="resolution"><strong>Resolution:</strong> {_e(scenario.resolution)}</div>
          {units}
        </div>
        <div>
          <div class="intent"><strong>Design intent:</strong> {_e(gold.design_intent)}</div>
          <h3>Gold graph</h3>
          <table><thead><tr><th>Edge</th><th>Why it is a direct response</th></tr></thead><tbody>{edge_rows}</tbody></table>
          <h3>Important hard negatives</h3>
          <table><thead><tr><th>Absent pair</th><th>Why it is not a response</th></tr></thead><tbody>{negative_rows}</tbody></table>
        </div>
      </div>
    </section>"""


def _e(value: object) -> str:
    return html.escape(str(value), quote=True)

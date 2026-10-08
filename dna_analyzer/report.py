"""Text, HTML and JSON reports."""

import html
import json
import textwrap
from datetime import date

from analyzer import FOUND
from snp_database import CATEGORY_ORDER, EVIDENCE_LEVELS

DISCLAIMER = (
    "For education only - not a medical test or diagnosis. Consumer DNA chips "
    "can misread variants, and most traits depend on many genes plus lifestyle. "
    "Confirm anything health-related with a clinical test and a doctor or "
    "genetic counsellor."
)


def _by_category(findings):
    groups = {}
    for f in findings:
        groups.setdefault(f.category, []).append(f)
    return [(c, groups[c]) for c in CATEGORY_ORDER if c in groups]


def _summary_lines(dna, findings, hidden):
    found = sum(1 for f in findings if f.status == FOUND)
    lines = [
        f"File format    : {dna.source_format}",
        f"Markers in file: {dna.total_markers:,}",
        f"Call rate      : {dna.call_rate:.1%}",
        f"Variants found : {found} of {len(findings)} analysed",
    ]
    if hidden:
        lines.append(
            f"Hidden         : {hidden} sensitive medical result(s) - "
            "re-run with --include-sensitive to see them"
        )
    return lines


def text_report(dna, findings, hidden, width=78):
    wrap = lambda s, indent: textwrap.fill(
        s, width=width, initial_indent=indent, subsequent_indent=indent
    )
    out = ["=" * width, "RAW DNA HEALTH GENE ANALYZER".center(width), "=" * width]
    out += _summary_lines(dna, findings, hidden)
    out += ["", wrap(DISCLAIMER, ""), ""]

    for category, items in _by_category(findings):
        out += ["-" * width, category.upper(), "-" * width]
        for f in items:
            out.append(f"{f.trait}  [{f.gene}, {f.rsid}]")
            if f.status == FOUND:
                out.append(f"    Genotype: {f.genotype}    Evidence: {f.evidence}")
                out.append(wrap(f.result, "    "))
                if f.note:
                    out.append(wrap("Note: " + f.note, "    "))
            else:
                out.append(f"    {f.status.capitalize()}" + (f" - {f.result}" if f.result else ""))
            out.append("")

    out.append("Evidence levels:")
    for level, meaning in EVIDENCE_LEVELS.items():
        out.append(wrap(f"{level}: {meaning}", "  "))
    if any(f.flipped_strand for f in findings):
        out.append("")
        out.append(wrap(
            "Some genotypes were reported on the opposite DNA strand and were "
            "flipped automatically.", ""))
    return "\n".join(out)


def json_report(dna, findings, hidden):
    return json.dumps(
        {
            "generated": date.today().isoformat(),
            "disclaimer": DISCLAIMER,
            "summary": {
                "format": dna.source_format,
                "markers": dna.total_markers,
                "call_rate": round(dna.call_rate, 4),
                "hidden_sensitive_results": hidden,
            },
            "findings": [f.to_dict() for f in findings],
        },
        indent=2,
    )


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DNA Gene Report</title>
<style>
  :root {{
    --bg: #f7f7f5; --card: #ffffff; --text: #1d1d1b; --muted: #66665f;
    --line: #e2e2dc; --strong: #1a7f4b; --moderate: #b26b00; --weak: #7a7a73;
    --warn-bg: #fff6e0; --warn-line: #f0d48a;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg: #161615; --card: #20201f; --text: #ececea; --muted: #a3a39c;
      --line: #34342f; --strong: #4cc38a; --moderate: #e9a23b; --weak: #9a9a92;
      --warn-bg: #2d2615; --warn-line: #5c4b1d;
    }}
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: var(--bg); color: var(--text);
         font: 16px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }}
  main {{ max-width: 860px; margin: 0 auto; padding: 32px 16px 64px; }}
  h1 {{ margin: 0 0 4px; font-size: 1.8rem; }}
  h2 {{ margin: 36px 0 12px; font-size: 1.15rem; text-transform: uppercase;
        letter-spacing: .06em; color: var(--muted); }}
  .sub {{ color: var(--muted); margin: 0 0 20px; }}
  .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 12px; margin-bottom: 20px; }}
  .stat {{ background: var(--card); border: 1px solid var(--line); border-radius: 10px;
           padding: 12px 14px; }}
  .stat b {{ display: block; font-size: 1.3rem; }}
  .stat span {{ color: var(--muted); font-size: .85rem; }}
  .warn {{ background: var(--warn-bg); border: 1px solid var(--warn-line);
           border-radius: 10px; padding: 12px 14px; font-size: .95rem; }}
  .card {{ background: var(--card); border: 1px solid var(--line); border-radius: 10px;
           padding: 14px 16px; margin-bottom: 10px; }}
  .head {{ display: flex; flex-wrap: wrap; justify-content: space-between; gap: 6px 12px; }}
  .trait {{ font-weight: 600; }}
  .meta {{ color: var(--muted); font-size: .85rem; }}
  .geno {{ font-family: ui-monospace, Menlo, Consolas, monospace; font-weight: 600; }}
  .badge {{ font-size: .75rem; font-weight: 600; padding: 2px 8px; border-radius: 99px;
            border: 1px solid currentColor; white-space: nowrap; align-self: flex-start; }}
  .strong {{ color: var(--strong); }} .moderate {{ color: var(--moderate); }}
  .weak {{ color: var(--weak); }}
  .card p {{ margin: 8px 0 0; }}
  .note {{ color: var(--muted); font-size: .9rem; }}
  .missing {{ opacity: .65; }}
</style>
</head>
<body>
<main>
  <h1>DNA Gene Report</h1>
  <p class="sub">Generated {generated} from a {fmt} raw data file. Analysed locally - nothing was uploaded.</p>
  <div class="stats">{stats}</div>
  <div class="warn">{disclaimer}</div>
  {sections}
</main>
</body>
</html>
"""


def html_report(dna, findings, hidden):
    e = html.escape
    found = sum(1 for f in findings if f.status == FOUND)
    stats = [
        (f"{dna.total_markers:,}", "markers in file"),
        (f"{dna.call_rate:.1%}", "call rate"),
        (f"{found} / {len(findings)}", "variants found"),
    ]
    if hidden:
        stats.append((str(hidden), "sensitive results hidden"))
    stats_html = "".join(f'<div class="stat"><b>{e(v)}</b><span>{e(k)}</span></div>' for v, k in stats)

    sections = []
    for category, items in _by_category(findings):
        cards = []
        for f in items:
            if f.status == FOUND:
                body = (
                    f'<p><span class="geno">{e(f.genotype)}</span> &middot; {e(f.result)}</p>'
                    + (f'<p class="note">{e(f.note)}</p>' if f.note else "")
                )
                cls = "card"
            else:
                body = f'<p>{e(f.status.capitalize())}{" - " + e(f.result) if f.result else ""}</p>'
                cls = "card missing"
            cards.append(
                f'<div class="{cls}"><div class="head">'
                f'<div><div class="trait">{e(f.trait)}</div>'
                f'<div class="meta">{e(f.gene)} &middot; {e(f.rsid)}</div></div>'
                f'<span class="badge {e(f.evidence)}">{e(f.evidence)} evidence</span>'
                f"</div>{body}</div>"
            )
        sections.append(f"<h2>{e(category)}</h2>" + "".join(cards))

    return HTML_TEMPLATE.format(
        generated=date.today().isoformat(),
        fmt=e(dna.source_format),
        stats=stats_html,
        disclaimer=e(DISCLAIMER),
        sections="\n".join(sections),
    )

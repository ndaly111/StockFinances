"""Render the microcap screener output as an HTML dashboard.

Takes the ranked candidate CSV produced by microcap_screener.py and writes
microcaps.html at the repo root, picked up by the next site refresh's
rsync to gh-pages.

Layout:
  - Header + last-updated stamp
  - Top N candidate cards (default 20) with business description, stats,
    and outbound links for further research
  - Sortable ranked table of ALL candidates below the cards

Reuses the existing retro.css theme via /static/css/retro.css for visual
consistency with the rest of the site.
"""

from __future__ import annotations

import argparse
import csv
import html
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("microcap_html")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IN = REPO_ROOT / "data" / "microcap_candidates.csv"
DEFAULT_OUT = REPO_ROOT / "microcaps.html"


def _pct(v: str, places: int = 1) -> str:
    """Format a decimal fraction string as a percent."""
    try:
        f = float(v)
        return f"{f*100:+.{places}f}%"
    except (TypeError, ValueError):
        return "—"


def _money(v: str) -> str:
    try:
        f = float(v)
        if f >= 1e9:
            return f"${f/1e9:.2f}B"
        if f >= 1e6:
            return f"${f/1e6:.0f}M"
        return f"${f:,.0f}"
    except (TypeError, ValueError):
        return "—"


def _safe(v) -> str:
    return html.escape(str(v or "")).replace("\n", " ")


def _research_links(ticker: str) -> str:
    t = html.escape(ticker)
    return (
        f'<a href="https://finance.yahoo.com/quote/{t}" target="_blank">Yahoo</a> · '
        f'<a href="https://finviz.com/quote.ashx?t={t}" target="_blank">Finviz</a> · '
        f'<a href="https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={t}" target="_blank">EDGAR</a>'
    )


def _card_html(row: dict) -> str:
    """One candidate card: header, stats grid, business summary, links."""
    gap_class = "positive" if (row.get("gap") and float(row["gap"]) > 0) else "negative"
    return f"""
<div class="card">
  <div class="card-head">
    <div>
      <h3>{_safe(row.get("ticker"))} — {_safe(row.get("name"))}</h3>
      <p class="subtle">{_safe(row.get("sector"))}{(' · ' + _safe(row.get("industry"))) if row.get("industry") else ''} · {_money(row.get("market_cap"))}</p>
    </div>
    <div class="links">{_research_links(row.get("ticker", ""))}</div>
  </div>
  <div class="stats">
    <div><span class="label">5-yr CAGR ({_safe(row.get("metric_used"))})</span><span class="val">{_pct(row.get("cagr_5yr"))}</span></div>
    <div><span class="label">Implied Growth</span><span class="val">{_pct(row.get("implied_growth"))}</span></div>
    <div><span class="label">Gap</span><span class="val {gap_class}">{_pct(row.get("gap"))}</span></div>
    <div><span class="label">Debt / Equity</span><span class="val">{_safe(row.get("debt_equity"))[:5]}</span></div>
    <div><span class="label">Consistency</span><span class="val">{_safe(row.get("years_positive"))}</span></div>
    <div><span class="label">YoY Sequence</span><span class="val seq">{_safe(row.get("yoy_sequence"))}</span></div>
  </div>
  <p class="biz">{_safe(row.get("business_summary"))}</p>
</div>
"""


def _table_html(rows: list[dict]) -> str:
    """Plain HTML table — all candidates, sortable via a small inline script."""
    if not rows:
        return '<p class="subtle">No candidates passed the filters.</p>'
    headers = [
        ("ticker", "Ticker"),
        ("name", "Name"),
        ("sector", "Sector"),
        ("market_cap", "Market Cap"),
        ("metric_used", "Metric"),
        ("cagr_5yr", "5-yr CAGR"),
        ("implied_growth", "Implied"),
        ("gap", "Gap"),
        ("debt_equity", "D/E"),
        ("years_positive", "Pos"),
        ("yoy_sequence", "YoY %"),
    ]
    head = "".join(
        f'<th data-key="{k}">{label}</th>' for k, label in headers
    )

    body_rows = []
    for r in rows:
        body_rows.append("<tr>" + "".join([
            f'<td><strong>{_safe(r.get("ticker"))}</strong></td>',
            f'<td>{_safe(r.get("name"))[:50]}</td>',
            f'<td>{_safe(r.get("sector"))}</td>',
            f'<td data-sort="{_safe(r.get("market_cap"))}">{_money(r.get("market_cap"))}</td>',
            f'<td>{_safe(r.get("metric_used"))}</td>',
            f'<td data-sort="{_safe(r.get("cagr_5yr"))}">{_pct(r.get("cagr_5yr"))}</td>',
            f'<td data-sort="{_safe(r.get("implied_growth"))}">{_pct(r.get("implied_growth"))}</td>',
            f'<td data-sort="{_safe(r.get("gap"))}">{_pct(r.get("gap"))}</td>',
            f'<td>{_safe(r.get("debt_equity"))[:5]}</td>',
            f'<td>{_safe(r.get("years_positive"))}</td>',
            f'<td class="seq">{_safe(r.get("yoy_sequence"))}</td>',
        ]) + "</tr>")

    return f"""
<table id="candidates">
  <thead><tr>{head}</tr></thead>
  <tbody>
    {''.join(body_rows)}
  </tbody>
</table>
"""


def build(candidates_csv: Path, out_path: Path, top_n: int = 20) -> None:
    if not candidates_csv.exists():
        log.error(f"Candidates CSV not found: {candidates_csv}")
        sys.exit(1)

    rows: list[dict] = []
    with open(candidates_csv, newline="", encoding="utf-8") as f:
        # Skip any comment lines we might have written.
        text = "\n".join(ln for ln in f.read().splitlines() if not ln.startswith("#"))
    if text.strip():
        reader = csv.DictReader(text.splitlines())
        rows = list(reader)

    # Already ranked by gap desc in the CSV; just take the top N for cards.
    top = rows[:top_n]
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    cards = "\n".join(_card_html(r) for r in top)
    table = _table_html(rows)

    page = f"""<!DOCTYPE html>
<html lang="en"><head>
  <meta charset="UTF-8">
  <title>Microcap Candidates — Nick's Stock Financials</title>
  <link rel="stylesheet" href="/static/css/retro.css">
  <style>
    .container {{ max-width: 1200px; margin: 0 auto; padding: 1rem 1.5rem 3rem; }}
    .lead {{ color:#000080; max-width: 70ch; }}
    .meta {{ color: #555; font-size: 0.9rem; margin: 0.5rem 0 1.5rem; }}

    .card {{
      border: 2px inset #C0C0C0;
      background: #fff;
      padding: 0.9rem 1rem;
      margin: 0.75rem 0;
      box-shadow: 1px 1px 0 #8080FF;
    }}
    .card-head {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 1rem;
      flex-wrap: wrap;
    }}
    .card h3 {{ margin: 0; color: #FF0000; text-shadow: 1px 1px #000080; }}
    .card .subtle {{ color: #555; font-size: 0.9rem; margin: 0.2rem 0 0; }}
    .card .links {{ font-size: 0.85rem; color: #444; white-space: nowrap; }}
    .card .links a {{ margin: 0 0.25rem; }}

    .stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 0.5rem 1rem;
      margin: 0.75rem 0;
      padding: 0.6rem 0.8rem;
      background: #f4f4ff;
      border: 1px solid #ccccff;
      border-radius: 4px;
    }}
    .stats > div {{ display: flex; flex-direction: column; }}
    .stats .label {{ font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; color: #555; }}
    .stats .val {{ font-weight: 700; color: #000080; font-variant-numeric: tabular-nums; }}
    .stats .val.positive {{ color: #0a7000; }}
    .stats .val.negative {{ color: #B00020; }}
    .stats .val.seq {{ font-family: 'JetBrains Mono', Consolas, monospace; font-size: 0.85rem; font-weight: 600; }}

    .biz {{ font-size: 0.92rem; color: #222; margin: 0.5rem 0 0; line-height: 1.4; }}

    table#candidates {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.85rem;
      margin-top: 1.5rem;
    }}
    table#candidates th {{
      background: #C0C0FF;
      padding: 6px;
      border: 1px solid #8080FF;
      text-align: left;
      cursor: pointer;
      user-select: none;
      position: sticky;
      top: 0;
    }}
    table#candidates td {{
      padding: 4px 6px;
      border: 1px solid #ddd;
      white-space: nowrap;
    }}
    table#candidates tr:nth-child(odd) td {{ background: #fafaff; }}
    table#candidates .seq {{ font-family: 'JetBrains Mono', Consolas, monospace; font-size: 0.78rem; }}
  </style>
</head><body><div class="container">

  <h1>Microcap Candidates</h1>
  <p class="lead">
    NYSE + NASDAQ common stocks with market cap between $50M and $1B,
    debt/equity below 0.5, ranked by the gap between 5-year historical
    growth (EPS, or revenue if not profitable five years ago) and the
    growth rate currently implied by the market price. A large positive
    gap means the market is pricing the company as if its track record
    won't continue — worth a closer look.
  </p>
  <p class="meta">Updated {timestamp} · {len(rows)} candidates passed filters · top {len(top)} shown as cards · <a href="/data/microcap_candidates.csv">CSV</a> · <a href="/data/microcap_candidates_skipped.csv">skipped (for tuning)</a></p>

  <h2>Top {len(top)} by Gap</h2>
  {cards}

  <h2>All Candidates</h2>
  {table}

</div>
<script>
  // Click any column header to sort. Numeric columns use data-sort
  // attributes when set; otherwise fall back to text compare.
  (function () {{
    var table = document.getElementById('candidates');
    if (!table) return;
    var headers = table.querySelectorAll('th');
    var dir = {{}};
    headers.forEach(function (h, idx) {{
      h.addEventListener('click', function () {{
        var rows = Array.from(table.tBodies[0].rows);
        var d = dir[idx] = !dir[idx];
        rows.sort(function (a, b) {{
          var av = a.cells[idx].getAttribute('data-sort');
          var bv = b.cells[idx].getAttribute('data-sort');
          var an = parseFloat(av), bn = parseFloat(bv);
          var bothNum = !isNaN(an) && !isNaN(bn);
          if (bothNum) return d ? an - bn : bn - an;
          av = (av != null ? av : a.cells[idx].textContent).toLowerCase();
          bv = (bv != null ? bv : b.cells[idx].textContent).toLowerCase();
          if (av < bv) return d ? -1 : 1;
          if (av > bv) return d ? 1 : -1;
          return 0;
        }});
        rows.forEach(function (r) {{ table.tBodies[0].appendChild(r); }});
      }});
    }});
  }})();
</script>
</body></html>"""

    out_path.write_text(page, encoding="utf-8")
    log.info(f"Wrote {len(top)} cards + {len(rows)}-row table -> {out_path}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--in", dest="src", type=Path, default=DEFAULT_IN)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--top", type=int, default=20)
    args = p.parse_args()
    build(args.src, args.out, top_n=args.top)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# generate_segment_tables.py
# Pulls segment data and writes HTML snippets to charts/{TICKER}_segments.html
# Also creates charts/segments_index.html to preview all.
#
# Usage:
#   pip install pandas requests beautifulsoup4 lxml
#   python generate_segment_tables.py

import time
from pathlib import Path
from datetime import datetime
import pandas as pd

from sec_segment_data_arelle import get_segment_data

# EDIT these to test
TICKERS = ["AAPL", "MSFT", "AMZN"]

OUTPUT_DIR = Path("charts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _fmt_billions(v) -> str:
    if pd.isna(v):
        return ""
    try:
        v = float(v)
    except Exception:
        return ""
    sign = "-" if v < 0 else ""
    v = abs(v)
    if v >= 1e9:
        return f"{sign}${v/1e9:.1f}B"
    elif v >= 1e6:
        return f"{sign}${v/1e6:.1f}M"
    else:
        return f"{sign}${v:,.0f}"


def _wrap_panel(inner: str) -> str:
    return f"""
<style>
.seg-block {{ margin: 16px 0; font-family: Arial, sans-serif; }}
.seg-block h3 {{ margin: 0 0 8px 0; font-size: 18px; }}
.seg-table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
.seg-table th {{ text-align: left; border-bottom: 2px solid #ddd; }}
.seg-table td {{ border-bottom: 1px solid #eee; }}
.seg-table th, .seg-table td {{ padding: 6px 8px; }}
.seg-foot {{ display:flex; gap:12px; margin-top:6px; font-size: 12px; color:#666; }}
</style>
{inner}
""".strip()


def render_table_html(ticker: str, df: pd.DataFrame) -> str:
    updated = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    title = f"{ticker} — Segment Revenue & Operating Income (Last 3 FY + TTM)"

    if df is None or df.empty:
        body = f"""
        <div class="seg-block">
          <h3>{title}</h3>
          <p>No segment-level data found from the latest 10-K/10-Q.</p>
          <p class="asof">Updated: {updated}</p>
        </div>
        """
        return _wrap_panel(body)

    # Sort: Year desc (TTM bottom), Revenue desc within year
    def year_key(y):
        return (9999 if y == "TTM" else int(y))
    df = df.copy()
    df["__yrkey"] = df["Year"].map(year_key)
    df.sort_values(["__yrkey", "Revenue"], ascending=[False, False], inplace=True)
    df.drop(columns="__yrkey", inplace=True)

    df["Revenue"] = df["Revenue"].map(_fmt_billions)
    df["OpIncome"] = df["OpIncome"].map(_fmt_billions)

    rows = []
    rows.append('<table class="seg-table" cellpadding="6" cellspacing="0">')
    rows.append("<thead><tr><th>Segment</th><th>Year</th><th>Revenue</th><th>Operating Income</th></tr></thead>")
    rows.append("<tbody>")
    for _, r in df.iterrows():
        rows.append(
            f"<tr><td>{r['Segment']}</td><td>{r['Year']}</td><td>{r['Revenue']}</td><td>{r['OpIncome']}</td></tr>"
        )
    rows.append("</tbody></table>")
    table_html = "\n".join(rows)

    body = f"""
    <div class="seg-block">
      <h3>{title}</h3>
      {table_html}
      <div class="seg-foot">
        <span class="asof">Updated: {updated}</span>
        <span class="src">Source: SEC Inline XBRL (10-K/10-Q)</span>
      </div>
    </div>
    """
    return _wrap_panel(body)


def save_html(path: Path, html: str):
    path.write_text(html, encoding="utf-8")
    print(f"✓ wrote {path}")


def main():
    all_snippets = []
    for i, t in enumerate(TICKERS, start=1):
        try:
            print(f"[{i}/{len(TICKERS)}] fetching {t}…")
            df = get_segment_data(t, dump_raw=True, raw_dir=OUTPUT_DIR / t)
            html = render_table_html(t, df)
            out_file = OUTPUT_DIR / f"{t}_segments.html"
            save_html(out_file, html)
            all_snippets.append((t, html))
            time.sleep(0.8)  # polite pacing
        except Exception as e:
            print(f"!! {t}: {e}")
            err_html = _wrap_panel(f"<h3>{t} — Segment Revenue & Operating Income</h3><p>Error: {e}</p>")
            save_html(OUTPUT_DIR / f"{t}_segments.html", err_html)

    pieces = [f"<h2>Segment Tables ({datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')})</h2>"]
    for t, html in all_snippets:
        pieces.append(f'<div id="{t}">{html}</div>')
    index_html = "\n\n".join(pieces)
    save_html(OUTPUT_DIR / "segments_index.html", index_html)


if __name__ == "__main__":
    main()

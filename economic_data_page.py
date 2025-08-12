#!/usr/bin/env python3
# economic_data_page.py – vertical layout + external CSS + 3Y button + CPI/UNRATE pp deltas
# -------------------------------------------------------------------
import sqlite3, pandas as pd, plotly.graph_objects as go, textwrap
from pathlib import Path
from datetime import datetime

DB_PATH      = "Stock Data.db"
HTML_OUT     = Path("economic_charts.html")
CSS_OUT      = Path("econ.css")

# ───────────── CSS content (written once) ─────────────
CSS_TXT = textwrap.dedent("""
  body{font-family:system-ui,Arial;margin:0 20px 40px;max-width:960px}
  h1  {margin:20px 0}
  nav {margin:0 0 20px}
  nav a{margin-right:14px;text-decoration:none;color:#0645AD}
  .sec{margin-top:80px}
  .stats{font-size:.9em;color:#555;margin:6px 0 0}
  .plotly-graph-div{width:100%;max-width:720px;height:350px}
""").strip()

def _ensure_css():
    if not CSS_OUT.exists():
        CSS_OUT.write_text(CSS_TXT, encoding="utf-8")
        print(f"[econ_page] wrote CSS → {CSS_OUT}")

# ───────────── helper functions ─────────────
def _get_series(conn, sid):
    # Order by normalized date so mixed formats don't break sorting
    return pd.read_sql(
        """SELECT substr(date,1,10) AS date, value
           FROM economic_data
           WHERE indicator=?
           ORDER BY substr(date,1,10)""",
        conn, params=(sid,)
    )

def _stats(df, sid=None):
    last   = df.iloc[-1]
    prev   = df.iloc[-2]  if len(df) > 1   else last
    yr_ago = df.iloc[-13] if len(df) > 13  else last

    latest_val = float(last['value'])
    mchg_val   = float(last['value']) - float(prev['value'])
    ychg_val   = float(last['value']) - float(yr_ago['value'])
    date_str   = str(last['date'])[:10]

    if sid in ("CPIAUCSL", "UNRATE"):
        return dict(
            latest=f"{latest_val:,.2f} %",
            date=date_str,
            mchg=f"{mchg_val:+.2f} pp",
            ychg=f"{ychg_val:+.2f} pp"
        )
    if sid == "DGS10":
        return dict(
            latest=f"{latest_val:,.2f} %",
            date=date_str,
            mchg=f"{mchg_val:+.2f}",
            ychg=f"{ychg_val:+.2f}"
        )
    if sid == "GDPC1":
        return dict(
            latest=f"{latest_val:,.2f}",
            date=date_str,
            mchg=f"{mchg_val:+.2f}",
            ychg=f"{ychg_val:+.2f}"
        )
    return dict(
        latest=f"{latest_val:,.2f}",
        date=date_str,
        mchg=f"{mchg_val:+.2f}",
        ychg=f"{ychg_val:+.2f}"
    )

def _plot_div(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.date, y=df.value, mode="lines"))
    fig.update_layout(
        margin=dict(l=0,r=0,t=20,b=0),
        height=350,
        xaxis=dict(
            rangeselector=dict(buttons=[
                dict(count=6,  label="6 M", step="month", stepmode="backward"),
                dict(count=1,  label="1 Y", step="year",  stepmode="backward"),
                dict(count=3,  label="3 Y", step="year",  stepmode="backward"),
                dict(count=5,  label="5 Y", step="year",  stepmode="backward"),
                dict(count=10, label="10 Y",step="year",  stepmode="backward"),
                dict(step="all", label="All")
            ]),
            rangeslider=dict(visible=False),
            type="date"
        )
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn")

# ───────────── main renderer ─────────────
def render_single_page(timestamp: str, indicators: dict):
    _ensure_css()

    toc, sections = [], []
    with sqlite3.connect(DB_PATH) as conn:
        for sid, meta in indicators.items():
            df = _get_series(conn, sid)
            if df.empty:
                continue
            s   = _stats(df, sid)
            div = _plot_div(df)

            toc.append(f'<a href="#{sid}">{meta["name"]}</a>')
            sections.append(f"""
              <div class="sec" id="{sid}">
                <h2>{meta['name']}</h2>
                {div}
                <p class="stats">
                  Latest : {s['latest']} ({s['date']}) |
                  1-mo Δ : {s['mchg']} |
                  YoY Δ : {s['ychg']}
                </p>
              </div>
            """)

    html = f"""<!doctype html><html><head>
<meta charset="utf-8">
<title>U.S. Economic Indicator Charts</title>
<link rel="stylesheet" href="{CSS_OUT}">
</head><body>
<h1>U.S. Economic Indicators – History & Charts</h1>
<p>Updated : {timestamp}</p>
<nav><strong>Jump to :</strong> {' '.join(toc)}</nav>
{''.join(sections)}
</body></html>"""

    HTML_OUT.write_text(html, encoding="utf-8")
    print(f"[econ_page] wrote → {HTML_OUT}")

# CLI
if __name__ == "__main__":
    from generate_economic_data import INDICATORS
    render_single_page(datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"), INDICATORS)

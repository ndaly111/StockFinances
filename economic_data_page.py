#!/usr/bin/env python3
# economic_data_page.py – vertical layout (no carousel)
# -----------------------------------------------------
import sqlite3, textwrap, pandas as pd, plotly.graph_objects as go
from pathlib import Path
from datetime import datetime

DB_PATH     = "Stock Data.db"
OUT_FILE    = Path("economic_charts.html")

# ─────────────── HTML & CSS ───────────────
CSS = """
  body{font-family:system-ui,Arial;margin:0 20px 40px;max-width:960px}
  h1{margin-top:20px}
  nav{margin-bottom:20px}
  nav a{margin-right:14px;text-decoration:none;color:#006}
  .sec{margin-top:80px}
  .stats{font-size:0.9em;color:#555;margin:6px 0 0}
  .plotly-graph-div{width:100%;max-width:720px;height:350px}
"""
HTML_HEAD = f"""<!doctype html><html><head>
<meta charset="utf-8"><title>U.S. Economic Indicator Charts</title>
<style>{CSS}</style></head><body>"""

HTML_FOOT = "</body></html>"
# ──────────────────────────────────────────

def _series_df(conn, sid: str) -> pd.DataFrame:
    return pd.read_sql(
        "SELECT date,value FROM economic_data WHERE indicator=? ORDER BY date",
        conn, params=(sid,)
    )

def _stats(df: pd.DataFrame) -> dict:
    last   = df.iloc[-1]
    prev   = df.iloc[-2]  if len(df) > 1   else last
    yr_ago = df.iloc[-13] if len(df) > 13  else last
    return dict(
        latest=f"{last['value']:,.2f}",
        date  =last['date'],
        mchg  =f"{last['value']-prev['value']:+.2f}",
        ychg  =f"{last['value']-yr_ago['value']:+.2f}"
    )

def _plotly_chart(df: pd.DataFrame) -> str:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["value"], mode="lines"))
    fig.update_layout(
        margin=dict(l=0, r=0, t=20, b=0),
        height=350,
        xaxis=dict(
            rangeselector=dict(
                buttons=[
                    dict(count=6,  label="6 M", step="month", stepmode="backward"),
                    dict(count=1,  label="1 Y", step="year",  stepmode="backward"),
                    dict(count=5,  label="5 Y", step="year",  stepmode="backward"),
                    dict(count=10, label="10 Y", step="year",  stepmode="backward"),
                    dict(step="all", label="All")
                ]
            ),
            rangeslider=dict(visible=False),
            type="date"
        )
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn")

# ─────────────── renderer ────────────────
def render_single_page(timestamp: str, indicators: dict):
    with sqlite3.connect(DB_PATH) as conn:
        toc_links, sections = [], []

        for sid, meta in indicators.items():
            df = _series_df(conn, sid)
            if df.empty:
                continue

            stats  = _stats(df)
            chart  = _plotly_chart(df)

            toc_links.append(f'<a href="#{sid}">{meta["name"]}</a>')
            sections.append(f"""
              <div class="sec" id="{sid}">
                <h2>{meta["name"]}</h2>
                {chart}
                <p class="stats">
                  Latest: {stats["latest"]} ({stats["date"]}) |
                  1-mo Δ: {stats["mchg"]} |
                  YoY Δ: {stats["ychg"]}
                </p>
              </div>
            """)

    html = (
        HTML_HEAD +
        "<h1>U.S. Economic Indicators – History & Charts</h1>" +
        f"<p>Updated: {timestamp}</p>" +
        "<nav><strong>Jump to:</strong> " + " ".join(toc_links) + "</nav>" +
        "".join(sections) +
        HTML_FOOT
    )
    OUT_FILE.write_text(html, encoding="utf-8")
    print(f"[econ_page] wrote → {OUT_FILE}")

# CLI helper
if __name__ == "__main__":
    from generate_economic_data import INDICATORS
    render_single_page(datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"), INDICATORS)

#!/usr/bin/env python3
# economic_data_page.py – builds economic_charts.html with interactive Plotly charts
# -------------------------------------------------------------------------------
import sqlite3, textwrap, pandas as pd, plotly.graph_objects as go
from pathlib import Path
from datetime import datetime

DB_PATH      = "Stock Data.db"
CHARTS_FILE  = Path("economic_charts.html")

HTML_HEAD = textwrap.dedent("""
<!doctype html><html><head>
<meta charset="utf-8">
<title>U.S. Economic Indicator Charts</title>
<style>
  body{font-family:system-ui,Arial;margin:20px;max-width:980px}
  h1{margin-top:0}
  nav ul{list-style:none;padding-left:0}
  nav li{margin:4px 0}
  .sec{margin-top:60px}
  .stats{font-size:0.9em;color:#555;margin-top:4px}
</style>
</head><body>
""")

HTML_FOOT = "</body></html>"

# ─────────────────────────────────────────────────────────────
def _series_df(conn, sid: str) -> pd.DataFrame:
    return pd.read_sql("""SELECT date,value
                          FROM   economic_data
                          WHERE  indicator=?
                          ORDER  BY date""",
                       conn, params=(sid,))

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

def _plotly_chart(df: pd.DataFrame, title: str) -> str:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["value"],
        mode="lines",
        name=title
    ))
    fig.update_layout(
        title=dict(text="",pad=0),
        margin=dict(l=0,r=0,t=20,b=0),
        height=320,
        xaxis=dict(
            rangeselector=dict(
                buttons=[
                    dict(count=6,  label="6 M", step="month", stepmode="backward"),
                    dict(count=1,  label="1 Y", step="year",  stepmode="backward"),
                    dict(count=5,  label="5 Y", step="year",  stepmode="backward"),
                    dict(count=10, label="10 Y",step="year",  stepmode="backward"),
                    dict(step="all", label="All")
                ]
            ),
            rangeslider=dict(visible=False),
            type="date"
        )
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn")

# Primary entry-point called from generate_economic_data.py
# ----------------------------------------------------------------
def render_single_page(timestamp: str, indicators: dict):
    with sqlite3.connect(DB_PATH) as conn:
        toc_parts   = []
        section_html = []

        for sid, meta in indicators.items():
            df = _series_df(conn, sid)
            if df.empty:
                continue

            stats = _stats(df)
            chart = _plotly_chart(df, meta["name"])

            toc_parts.append(f'<li><a href="#{sid}">{meta["name"]}</a></li>')
            section_html.append(f"""
              <div class="sec" id="{sid}">
                <h2>{meta["name"]}</h2>
                {chart}
                <p class="stats">
                  Latest: {stats["latest"]} ({stats["date"]}) &nbsp;|&nbsp;
                  1-mo Δ: {stats["mchg"]} &nbsp;|&nbsp;
                  YoY Δ: {stats["ychg"]}
                </p>
              </div>
            """)

    full_html = (
        HTML_HEAD +
        f"<h1>U.S. Economic Indicators – History & Charts</h1>"
        f"<p>Updated: {timestamp}</p>"
        "<nav><strong>Jump to:</strong><ul>" + "".join(toc_parts) + "</ul></nav>" +
        "".join(section_html) +
        HTML_FOOT
    )

    CHARTS_FILE.write_text(full_html, encoding="utf-8")
    print(f"[econ_page] wrote → {CHARTS_FILE}")

# Stand-alone CLI use ─ optional
if __name__ == "__main__":
    from generate_economic_data import INDICATORS  # reuse the dict
    render_single_page(datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"), INDICATORS)

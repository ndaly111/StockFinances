#!/usr/bin/env python3
# economic_data_page.py – builds economic_charts.html from DB
# -----------------------------------------------------------
import sqlite3, pandas as pd, textwrap, os
from pathlib import Path

DB_PATH      = "Stock Data.db"
CHARTS_DIR   = "charts"
OUT_FILE     = Path("economic_charts.html")

HTML_HEAD = textwrap.dedent("""\
    <!doctype html><html><head>
    <meta charset="utf-8">
    <title>U.S. Economic Indicator Charts</title>
    <style>
      body{font-family:system-ui,Arial;margin:20px}
      h1{margin-top:0}
      nav ul{list-style:none;padding-left:0}
      nav li{margin:4px 0}
      img{max-width:100%;height:auto}
      .sec{margin-top:50px}
      .stats{font-size:0.9em;color:#444}
    </style></head><body>
""")

HTML_FOOT = "</body></html>"

def _series_df(conn,sid):
    return pd.read_sql("""SELECT date,value FROM economic_data
                          WHERE indicator=? ORDER BY date""",conn,params=(sid,))

def _stats(df):
    last   = df.iloc[-1]
    prev   = df.iloc[-2] if len(df)>1 else last
    yrago  = df.iloc[-13] if len(df)>13 else last
    return dict(
        latest = f"{last['value']:,.2f}",
        date   = last['date'],
        mchg   = f"{last['value']-prev['value']:+.2f}",
        ychg   = f"{last['value']-yrago['value']:+.2f}"
    )

def render_single_page(timestamp:str, indicators:dict):
    with sqlite3.connect(DB_PATH) as conn:
        toc=[]
        sections=[]
        for sid,meta in indicators.items():
            df=_series_df(conn,sid)
            if df.empty: continue
            stats=_stats(df)
            toc.append(f'<li><a href="#{sid}">{meta["name"]}</a></li>')
            sections.append(f"""
              <div class="sec" id="{sid}">
                <h2>{meta["name"]}</h2>
                <img src="{CHARTS_DIR}/{sid}_history.png" alt="{meta["name"]} chart">
                <p class="stats">
                  Latest: {stats["latest"]} ({stats["date"]}) |
                  1-mo Δ: {stats["mchg"]} |
                  YoY Δ: {stats["ychg"]}
                </p>
              </div>
            """)
    html = (HTML_HEAD +
            f"<h1>U.S. Economic Indicators – History & Charts</h1>"
            f"<p>Updated: {timestamp}</p>"
            "<nav><strong>Jump to:</strong><ul>" + "".join(toc) + "</ul></nav>" +
            "".join(sections) +
            HTML_FOOT)
    OUT_FILE.write_text(html,encoding="utf-8")
    print(f"[econ_page] wrote → {OUT_FILE}")

# CLI helper
if __name__=="__main__":
    from datetime import datetime
    render_single_page(datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),{})  # indicators passed when imported

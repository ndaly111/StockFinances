#!/usr/bin/env python3
# economic_data_page.py – carousel version with scroll-snap
# ---------------------------------------------------------
import sqlite3, textwrap, pandas as pd, plotly.graph_objects as go
from pathlib import Path
from datetime import datetime

DB_PATH     = "Stock Data.db"
OUT_FILE    = Path("economic_charts.html")

CSS = """
  body{font-family:system-ui,Arial;margin:0}
  h1{margin:20px}
  nav{margin:0 20px 20px}
  nav a{margin-right:14px;text-decoration:none;color:#006}
  .carousel{display:flex;overflow-x:auto;scroll-snap-type:x mandatory;
            -webkit-overflow-scrolling:touch}
  .slide{scroll-snap-align:start;flex:0 0 100%;box-sizing:border-box;padding:0 20px}
  .stats{font-size:0.9em;color:#555;margin:6px 0 60px}
"""
HTML_HEAD = f"""<!doctype html><html><head>
<meta charset="utf-8"><title>U.S. Economic Indicator Charts</title>
<style>{CSS}</style>
</head><body>"""

HTML_FOOT = """
<script>
  // smooth scroll for nav links
  document.querySelectorAll('nav a').forEach(a=>{
    a.addEventListener('click',e=>{
      e.preventDefault();
      document.querySelector(a.getAttribute('href'))
              .scrollIntoView({behavior:'smooth'});
    });
  });
</script>
</body></html>"""

def _get_series(conn,sid):
    return pd.read_sql("SELECT date,value FROM economic_data WHERE indicator=? ORDER BY date",
                       conn,params=(sid,))

def _stats(df):
    last,prev,yr=df.iloc[-1],df.iloc[-2],df.iloc[-13] if len(df)>13 else df.iloc[-1]
    latest=f"{last['value']:,.2f}"
    return latest,str(last['date']),f"{last['value']-prev['value']:+.2f}",f"{last['value']-yr['value']:+.2f}"

def _plot(df):
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=df.date,y=df.value,mode="lines"))
    fig.update_layout(
        margin=dict(l=0,r=0,t=10,b=0),height=350,
        xaxis=dict(
            rangeselector=dict(buttons=[
              dict(count=6,label="6 M",step="month",stepmode="backward"),
              dict(count=1,label="1 Y",step="year",stepmode="backward"),
              dict(count=5,label="5 Y",step="year",stepmode="backward"),
              dict(count=10,label="10 Y",step="year",stepmode="backward"),
              dict(step="all",label="All")
            ]),
            rangeslider=dict(visible=False),type="date"))
    return fig.to_html(full_html=False,include_plotlyjs="cdn")

def render_single_page(ts,indicators):
    with sqlite3.connect(DB_PATH) as conn:
        nav_links=[]
        slides=[]
        for sid,meta in indicators.items():
            df=_get_series(conn,sid)
            if df.empty: continue
            latest,date,mchg,ychg=_stats(df)
            chart=_plot(df)
            nav_links.append(f'<a href="#{sid}">{meta["name"]}</a>')
            slides.append(f"""
              <section class="slide" id="{sid}">
                 <h2>{meta['name']}</h2>
                 {chart}
                 <p class="stats">Latest: {latest} ({date}) | 1-mo Δ: {mchg} | YoY Δ: {ychg}</p>
              </section>""")

    html=(HTML_HEAD+
          f"<h1>U.S. Economic Indicators – History & Charts</h1>"
          f"<nav>{' '.join(nav_links)}</nav>"
          f'<div class="carousel">{"".join(slides)}</div>'+
          HTML_FOOT)
    OUT_FILE.write_text(html,encoding="utf-8")
    print(f"[econ_page] wrote → {OUT_FILE}")

# standalone run
if __name__=="__main__":
    from generate_economic_data import INDICATORS
    render_single_page(datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"), INDICATORS)

#!/usr/bin/env python3
# html_generator2.py – adds “Implied-Growth Pctile” to the front dashboard
# ───────────────────────────────────────────────────────────
from jinja2 import Environment, FileSystemLoader, Template
import os, sqlite3, numpy as np, pandas as pd, yfinance as yf

db_path = "Stock Data.db"
env = Environment(loader=FileSystemLoader("templates"))

# ─── helpers (unchanged) ──────────────────────────────────
def ensure_directory_exists(p):  os.makedirs(p, exist_ok=True) if p else None
def create_template(path, content):
    ensure_directory_exists(os.path.dirname(path))
    if not os.path.exists(path) or open(path,encoding="utf-8").read()!=content:
        open(path,"w",encoding="utf-8").write(content)

def get_company_short_name(tk, cur):
    cur.execute("SELECT short_name FROM Tickers_Info WHERE ticker=?",(tk,))
    row=cur.fetchone()
    if row and row[0]: return row[0]
    name=(yf.Ticker(tk).info or {}).get("shortName","").strip() or tk
    cur.execute("UPDATE Tickers_Info SET short_name=? WHERE ticker=?",(name,tk))
    cur.connection.commit(); return name

def get_file_or_placeholder(p, ph="No data available"):
    try: return open(p,encoding="utf-8").read()
    except FileNotFoundError: return ph

# ─── templates – only JS block edited for new colour logic ─
def ensure_templates_exist():
    home_template = """<!DOCTYPE html>
<html lang="en"><head>
  <meta charset="UTF-8">
  <title>Nick's Stock Financials</title>
  <link rel="stylesheet" href="style.css">
  <link rel="stylesheet" href="https://cdn.datatables.net/1.10.21/css/jquery.dataTables.min.css">
  <style>td.positive{color:green;}td.negative{color:red;}.center-table{margin:0 auto;width:80%%}</style>
  <script src="https://code.jquery.com/jquery-3.5.1.js"></script>
  <script src="https://cdn.datatables.net/1.10.21/js/jquery.dataTables.min.js"></script>
  <script>
    $(function(){
      $('#sortable-table').DataTable({
        pageLength:100,
        createdRow:function(row){
          $('td',row).each(function(){
            var txt=$(this).text();
            if(txt.includes('%')){
              var n=parseFloat(txt.replace('%',''));
              if(isNaN(n)) return;
              var col=$(this).index();
              if(col===6){               // Implied-Growth Pctile column
                $(this).addClass(n<50?'negative':'positive');
              }else{                     // existing % columns
                $(this).addClass(n<0?'negative':'positive');
              }
            }
          });
        }
      });
    });
  </script>
</head><body>
  <header><h1>Financial Overview</h1></header>

  <nav class="navigation">
    {% for t in tickers %}
      <a href="pages/{{t}}_page.html" class="home-button">{{t}}</a>{% if not loop.last %} | {% endif %}
    {% endfor %}
  </nav>

  <div id="spy-qqq-growth" class="center-table">
    <h2>SPY vs QQQ Overview</h2>
    {{ spy_qqq_growth | safe }}
  </div>

  <div class="center-table">
    <h2>Past Earnings (Last 7 Days)</h2>
    {{ earnings_past | safe }}
    <h2>Upcoming Earnings</h2>
    {{ earnings_upcoming | safe }}
  </div>

  <div>{{ dashboard_table | safe }}</div>

  <footer><p>Nick's Financial Data Dashboard</p></footer>
</body></html>"""
    create_template("templates/home_template.html", home_template)
    # ticker / spy-qqq templates unchanged …

# ───────────────────────────────────────────────────────────
# Dashboard table + summary (now includes Pctile)
# ───────────────────────────────────────────────────────────
def generate_dashboard_table(dashboard_data):
    base_cols = ["Ticker","Share Price",
                 "Nick's TTM Value","Nick's Forward Value",
                 "Finviz TTM Value","Finviz Forward Value"]
    df = pd.DataFrame(dashboard_data, columns=base_cols)

    # ---- pull latest TTM percentile ----
    with sqlite3.connect(db_path) as conn:
        pct = pd.read_sql_query(
            """SELECT Ticker, Percentile
                 FROM Index_Growth_Pctile
                WHERE Growth_Type='TTM'
                  AND Date=(SELECT MAX(Date) FROM Index_Growth_Pctile)""", conn)
    df = df.merge(pct, how="left", on="Ticker")

    # ---- format numbers & add helper cols ----
    for col in base_cols[2:]:
        num = pd.to_numeric(df[col].astype(str).str.rstrip("%"), errors="coerce")
        df[col+"_num"] = num
        df[col] = num.apply(lambda x: f"{x:.1f}%" if pd.notnull(x) else "–")

    df["Implied-Growth Pctile_num"] = df["Percentile"]
    df["Implied-Growth Pctile"] = df["Percentile"].apply(
        lambda x: f"{x:.0f}%" if pd.notnull(x) else "–")
    df.drop(columns="Percentile", inplace=True)

    # ---- hyperlink tickers *after* merge ----
    def link(t):
        if t=="SPY": return '<a href="spy_growth.html">SPY</a>'
        if t=="QQQ": return '<a href="qqq_growth.html">QQQ</a>'
        return f'<a href="pages/{t}_page.html">{t}</a>'
    df["Ticker"] = df["Ticker"].apply(link)

    # sort by Nick's TTM %
    df.sort_values("Nick's TTM Value_num", ascending=False, inplace=True)

    # ---- build summary rows (unchanged logic) ----
    ttm   = df["Nick's TTM Value_num"].dropna()
    fwd   = df["Nick's Forward Value_num"].dropna()
    fttm  = df["Finviz TTM Value_num"].dropna()
    ffwd  = df["Finviz Forward Value_num"].dropna()
    pc = lambda s: f"{s:.1f}%" if pd.notnull(s) else "–"
    summary = [["Average",pc(ttm.mean()),pc(fwd.mean()),pc(fttm.mean()),pc(ffwd.mean())],
               ["Median", pc(ttm.median()),pc(fwd.median()),pc(fttm.median()),pc(ffwd.median())]]
    avg_html = pd.DataFrame(summary, columns=["Metric"]+base_cols[2:]).to_html(
        index=False, classes="table table-striped", escape=False)

    # ---- final column order for dashboard ----
    dash_cols = base_cols + ["Implied-Growth Pctile"]
    dash_html = df[dash_cols].to_html(
        index=False, classes="table table-striped", table_id="sortable-table", escape=False)

    ensure_directory_exists("charts")
    open("charts/dashboard.html","w",encoding="utf-8").write(avg_html+dash_html)

    return avg_html+dash_html, {
        "Nicks_TTM_Value_Average":ttm.mean(),"Nicks_TTM_Value_Median":ttm.median(),
        "Nicks_Forward_Value_Average":fwd.mean(),"Nicks_Forward_Value_Median":fwd.median(),
        "Finviz_TTM_Value_Average":fttm.mean() if not fttm.empty else None,
        "Finviz_TTM_Value_Median":fttm.median() if not fttm.empty else None,
        "Finviz_Forward_Value_Average":ffwd.mean() if not ffwd.empty else None,
        "Finviz_Forward_Value_Median":ffwd.median() if not ffwd.empty else None
    }

# ─── rest of file (create_home_page, per-ticker pages, etc.) unchanged ──
# …

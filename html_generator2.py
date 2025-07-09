#!/usr/bin/env python3
# html_generator2.py  –  responsive edition
from jinja2 import Environment, FileSystemLoader, Template
import os, sqlite3, numpy as np, pandas as pd, yfinance as yf

db_path = "Stock Data.db"
env = Environment(loader=FileSystemLoader("templates"))

# ─── helpers (unchanged) ─────────────────────────────────
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

def get_file_content_or_placeholder(p, ph="No data available"):
    try:  return open(p,encoding="utf-8").read()
    except FileNotFoundError: return ph

# ─── templates (viewport + container added) ──────────────
def ensure_templates_exist():
    home_template = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Nick's Stock Financials</title>
  <link rel="stylesheet" href="style.css">
  <link rel="stylesheet" href="https://cdn.datatables.net/1.10.21/css/jquery.dataTables.min.css">
  <style>
    td.positive{color:green;} td.negative{color:red;}
    .center-table{margin:0 auto;width:100%%;}
  </style>
  <script src="https://code.jquery.com/jquery-3.5.1.js"></script>
  <script src="https://cdn.datatables.net/1.10.21/js/jquery.dataTables.min.js"></script>
  <script>
    $(function(){
      $('#sortable-table').DataTable({
        pageLength:100,responsive:true,scrollX:true,
        createdRow:function(r){
          $('td',r).each(function(){
            var t=$(this).text();
            if(t.includes('%')){
              var n=parseFloat(t.replace('%',''));
              if(!isNaN(n))$(this).addClass(n<0?'negative':'positive');
            }
          });
        }
      });
    });
  </script>
</head>
<body>
<div class="container">
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
</div>
</body></html>"""

    ticker_template = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{{ ticker_data.company_name }} – Financial Overview</title>
  <link rel="stylesheet" href="../style.css">
</head>
<body><div class="container">
  <header>
    <a href="../index.html" class="home-button">Home</a>
    <h1>{{ ticker_data.company_name }} – Financial Overview</h1>
    <h2>Ticker: {{ ticker_data.ticker }}</h2>
  </header>

  <section>{{ ticker_data.ticker_info | safe }}</section>

  <div>
    <img src="../{{ ticker_data.revenue_net_income_chart_path }}" alt="Rev vs NI">
    <img src="../{{ ticker_data.eps_chart_path }}" alt="EPS">
    {{ ticker_data.financial_table | safe }}
  </div>

  <h1>{{ ticker_data.ticker }} – Forecast Data</h1>
  <div class="carousel-container">
    <div class="carousel-item"><img src="../{{ ticker_data.forecast_rev_net_chart_path }}"></div>
    <div class="carousel-item"><img src="../{{ ticker_data.forecast_eps_chart_path }}"></div>
  </div>

  <h1>{{ ticker_data.ticker }} – Y/Y % Change</h1>
  <div class="carousel-container">
    <img class="carousel-item" src="../{{ ticker_data.revenue_yoy_change_chart_path }}">
    <img class="carousel-item" src="../{{ ticker_data.eps_yoy_change_chart_path }}">
    <div class="carousel-item">{{ ticker_data.yoy_growth_table_html | safe }}</div>
  </div>

  <div class="balance-sheet-container">
    <div class="balance-sheet-table">{{ ticker_data.balance_sheet_table_html | safe }}</div>
    <div class="balance-sheet-chart"><img src="../{{ ticker_data.balance_sheet_chart_path }}"></div>
  </div>

  <h1>{{ ticker_data.ticker }} – Expense Overview</h1>
  <div class="carousel-container">
    <img class="carousel-item" src="../{{ ticker_data.expense_chart_path }}">
    <img class="carousel-item" src="../{{ ticker_data.expense_percent_chart_path }}">
    <div class="carousel-item">{{ ticker_data.expense_abs_html | safe }}</div>
    <div class="carousel-item">{{ ticker_data.expense_yoy_html | safe }}</div>
  </div>

  {% if ticker_data.implied_growth_chart_path %}
  <h1>{{ ticker_data.ticker }} – Implied Growth Summary</h1>
  <img src="../{{ ticker_data.implied_growth_chart_path }}">
  <div class="implied-growth-table">{{ ticker_data.implied_growth_table_html | safe }}</div>
  {% endif %}

  <footer><a href="../index.html" class="home-button">Back to Home</a></footer>
</div></body></html>"""

    spy_tpl = """<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SPY – Implied Growth & P/E</title>
<link rel="stylesheet" href="style.css"></head>
<body><div class="container">
  <header><a href="index.html">← Home</a></header>

  <h1>SPY – Implied Growth Summary</h1>
  <img src="charts/spy_growth_chart.png" alt="SPY Growth Chart">
  {{ spy_growth_summary | safe }}

  <h1>SPY – P/E Ratio Summary</h1>
  <img src="charts/spy_pe_chart.png" alt="SPY P/E Chart">
  {{ spy_pe_summary | safe }}
</div></body></html>"""

    qqq_tpl = spy_tpl.replace("SPY", "QQQ").replace("spy_", "qqq_")

    create_template("templates/home_template.html",   home_template)
    create_template("templates/ticker_template.html", ticker_template)
    create_template("templates/spy_growth_template.html", spy_tpl)
    create_template("templates/qqq_growth_template.html", qqq_tpl)

# ── rest of file (render functions, dashboard table, etc.) is unchanged ──
# keep all the code you already have below this point verbatim …

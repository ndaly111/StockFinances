from jinja2 import Environment, FileSystemLoader
import os
import pandas as pd
import sqlite3
import yfinance as yf
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DB_PATH = 'Stock Data.db'
CHARTS_DIR = 'charts'
PAGES_DIR = 'pages'
TEMPLATES_DIR = 'templates'
env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def create_template(template_path, content):
    ensure_directory_exists(os.path.dirname(template_path))
    if os.path.exists(template_path):
        with open(template_path, 'r', encoding='utf-8') as f:
            if f.read() == content:
                return
    with open(template_path, 'w', encoding='utf-8') as f:
        f.write(content)

def get_company_short_name(ticker, cursor):
    cursor.execute("SELECT short_name FROM Tickers_Info WHERE ticker = ?", (ticker,))
    row = cursor.fetchone()
    if row and row[0]:
        return row[0]
    stock = yf.Ticker(ticker)
    short_name = stock.info.get('shortName', '').strip()
    if short_name:
        cursor.execute(
            "UPDATE Tickers_Info SET short_name = ? WHERE ticker = ?",
            (short_name, ticker)
        )
        cursor.connection.commit()
        return short_name
    return ticker

def get_file_content_or_placeholder(path, placeholder="No data available"):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return placeholder

def ensure_templates_exist():
    ticker_template_content = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{{ ticker_data.company_name }} – Financial Overview</title>
  <link rel="stylesheet" href="../style.css">
</head>
<body>
  <header>
    <a href="../index.html" class="home-button">Home</a>
    <h1>{{ ticker_data.company_name }} – Financial Overview</h1>
    <h2>Ticker: {{ ticker_data.ticker }}</h2>
  </header>

  <section>{{ ticker_data.ticker_info | safe }}</section>

  <div>
    <img src="../{{ ticker_data.revenue_net_income_chart_path }}" alt="Rev vs NI">
    <img src="../{{ ticker_data.eps_chart_path }}"             alt="EPS">
    {{ ticker_data.financial_table | safe }}
  </div>

  <h1>{{ ticker_data.ticker }} – Forecast Data</h1>
  <div class="carousel-container">
    <div class="carousel-item">
      <img src="../{{ ticker_data.forecast_rev_net_chart_path }}" alt="Rev/NI Forecast">
    </div>
    <div class="carousel-item">
      <img src="../{{ ticker_data.forecast_eps_chart_path }}"     alt="EPS Forecast">
    </div>
  </div>

  <h1>{{ ticker_data.ticker }} – Y/Y % Change</h1>
  <div class="carousel-container">
    <img class="carousel-item" src="../{{ ticker_data.revenue_yoy_change_chart_path }}" alt="Rev YoY">
    <img class="carousel-item" src="../{{ ticker_data.eps_yoy_change_chart_path }}"     alt="EPS YoY">
    <div class="carousel-item">{{ ticker_data.yoy_growth_table_html | safe }}</div>
  </div>

  <div class="balance-sheet-container">
    <div class="balance-sheet-table">{{ ticker_data.balance_sheet_table_html | safe }}</div>
    <div class="balance-sheet-chart">
      <img src="../{{ ticker_data.balance_sheet_chart_path }}" alt="BS Chart">
    </div>
  </div>

  <h1>{{ ticker_data.ticker }} – Expense Overview</h1>
  <div class="carousel-container">
    <img class="carousel-item" src="../{{ ticker_data.expense_chart_path }}"         alt="Rev vs Exp">
    <img class="carousel-item" src="../{{ ticker_data.expense_percent_chart_path }}" alt="Exp % of Rev">
    <div class="carousel-item">{{ ticker_data.expense_table_html | safe }}</div>
    <div class="carousel-item">{{ ticker_data.expense_yoy_table_html | safe }}</div>
  </div>

  {% if ticker_data.unmapped_expense_html %}
  <h1>{{ ticker_data.ticker }} – Unmapped Items</h1>
  <div>{{ ticker_data.unmapped_expense_html | safe }}</div>
  {% endif %}

  {% if ticker_data.valuation_chart %}
  <h1>{{ ticker_data.ticker }} – Valuation Chart</h1>
  <img src="../{{ ticker_data.valuation_chart }}" alt="Valuation">
  <div class="valuation-tables">
    {{ ticker_data.valuation_info_table | safe }}
    {{ ticker_data.valuation_data_table | safe }}
  </div>
  {% endif %}

  {% if ticker_data.implied_growth_chart_path %}
  <h1>{{ ticker_data.ticker }} – Implied Growth Summary</h1>
  <img src="../{{ ticker_data.implied_growth_chart_path }}" alt="Implied Growth Summary">
  {% endif %}

  <footer>
    <a href="../index.html" class="home-button">Back to Home</a>
  </footer>
</body>
</html>"""
    create_template(f"{TEMPLATES_DIR}/ticker_template.html", ticker_template_content)

# ---------------------------------------------------------------------------
# Page Builders
# ---------------------------------------------------------------------------
def create_home_page(tickers, output_dir, dashboard_table, avg_values, spy_qqq_growth="", earnings_past="", earnings_upcoming=""):
    tpl = env.get_template('home_template.html')
    out_path = os.path.join(output_dir, 'index.html')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(tpl.render(
            tickers=tickers,
            dashboard_table=dashboard_table,
            dashboard_data=avg_values,
            spy_qqq_growth=spy_qqq_growth,
            earnings_past=earnings_past,
            earnings_upcoming=earnings_upcoming
        ))

def prepare_and_generate_ticker_pages(tickers, output_dir, charts_dir):
    ensure_directory_exists(os.path.join(output_dir, PAGES_DIR))
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        for t in tickers:
            name = get_company_short_name(t, cur)
            d = {
                'ticker': t,
                'company_name': name,
                'ticker_info': get_file_content_or_placeholder(f"{charts_dir}/{t}_ticker_info.html"),
                'revenue_net_income_chart_path': f"{charts_dir}/{t}_revenue_net_income_chart.png",
                'eps_chart_path': f"{charts_dir}/{t}_eps_chart.png",
                'financial_table': get_file_content_or_placeholder(f"{charts_dir}/{t}_rev_net_table.html"),
                'forecast_rev_net_chart_path': f"{charts_dir}/{t}_Revenue_Net_Income_Forecast.png",
                'forecast_eps_chart_path': f"{charts_dir}/{t}_EPS_Forecast.png",
                'yoy_growth_table_html': get_file_content_or_placeholder(f"{charts_dir}/{t}_yoy_growth_tbl.html"),
                'expense_chart_path': f"{charts_dir}/{t}_rev_expense_chart.png",
                'expense_percent_chart_path': f"{charts_dir}/{t}_expense_percent_chart.png",
                'expense_table_html': get_file_content_or_placeholder(f"{charts_dir}/{t}_yearly_financials.html"),
                'expense_yoy_table_html': get_file_content_or_placeholder(f"{charts_dir}/{t}_yoy_expense_change.html"),
                'balance_sheet_chart_path': f"{charts_dir}/{t}_balance_sheet_chart.png",
                'balance_sheet_table_html': get_file_content_or_placeholder(f"{charts_dir}/{t}_balance_sheet_table.html"),
                'revenue_yoy_change_chart_path': f"{charts_dir}/{t}_revenue_yoy_change.png",
                'eps_yoy_change_chart_path': f"{charts_dir}/{t}_eps_yoy_change.png",
                'valuation_chart': f"{charts_dir}/{t}_valuation_chart.png",
                'valuation_info_table': get_file_content_or_placeholder(f"{charts_dir}/{t}_valuation_info.html"),
                'valuation_data_table': get_file_content_or_placeholder(f"{charts_dir}/{t}_valuation_table.html"),
                'unmapped_expense_html': get_file_content_or_placeholder(f"{charts_dir}/{t}_unmapped_fields.html", "No unmapped expenses."),
                'eps_dividend_chart_path': f"{charts_dir}/{t}_eps_dividend_forecast.png",
                'implied_growth_chart_path': f"{charts_dir}/{t}_implied_growth_chart.png"
            }
            tpl = env.get_template('ticker_template.html')
            out = os.path.join(output_dir, PAGES_DIR, f"{t}_page.html")
            with open(out, 'w', encoding='utf-8') as f:
                f.write(tpl.render(ticker_data=d))

# ---------------------------------------------------------------------------
# Public Entrypoint
# ---------------------------------------------------------------------------
def html_generator2(tickers, financial_data, full_dashboard_html, avg_values, spy_qqq_growth_html=""):
    ensure_templates_exist()

    earnings_past = get_file_content_or_placeholder(f"{CHARTS_DIR}/earnings_past.html")
    earnings_upcoming = get_file_content_or_placeholder(f"{CHARTS_DIR}/earnings_upcoming.html")

    create_home_page(
        tickers=tickers,
        output_dir='.',
        dashboard_table=full_dashboard_html,
        avg_values=avg_values,
        spy_qqq_growth=spy_qqq_growth_html,
        earnings_past=earnings_past,
        earnings_upcoming=earnings_upcoming
    )
    prepare_and_generate_ticker_pages(tickers, '.', CHARTS_DIR)

# ---------------------------------------------------------------------------
# Mini-Main
# ---------------------------------------------------------------------------
def generate_all():
    tickers = ["AAPL", "MSFT"]  # placeholder example
    financial_data = {}         # optional for now
    full_dashboard_html = "<p>Sample Dashboard</p>"  # or load from charts/dashboard.html
    avg_values = {
        "Nicks_TTM_Value_Average": 18.2,
        "Nicks_Forward_Value_Average": 15.7,
        "Finviz_TTM_Value_Average": 21.1,
        "Finviz_Forward_Value_Average": 17.3
    }
    spy_qqq_growth_html = "<div>SPY/QQQ</div>"

    html_generator2(tickers, financial_data, full_dashboard_html, avg_values, spy_qqq_growth_html)

if __name__ == "__main__":
    generate_all()

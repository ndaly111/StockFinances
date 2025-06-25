from jinja2 import Environment, FileSystemLoader
import os
import pandas as pd
import sqlite3
import yfinance as yf
import numpy as np

DB_PATH = 'Stock Data.db'
CHARTS_DIR = 'charts'
PAGES_DIR = 'pages'
TEMPLATES_DIR = 'templates'
env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))

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
        cursor.execute("UPDATE Tickers_Info SET short_name = ? WHERE ticker = ?", (short_name, ticker))
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
    ticker_template = """<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>{{ ticker_data.company_name }}</title></head>
<body>
  <h1>{{ ticker_data.company_name }} – Financial Overview</h1>
  <h2>{{ ticker_data.ticker }}</h2>
  <section>{{ ticker_data.ticker_info | safe }}</section>
  <img src="../{{ ticker_data.revenue_net_income_chart_path }}" alt="Rev vs NI">
  <img src="../{{ ticker_data.eps_chart_path }}" alt="EPS">
  {{ ticker_data.financial_table | safe }}
  <img src="../{{ ticker_data.forecast_rev_net_chart_path }}" alt="Rev Forecast">
  <img src="../{{ ticker_data.forecast_eps_chart_path }}" alt="EPS Forecast">
  <img src="../{{ ticker_data.revenue_yoy_change_chart_path }}" alt="Rev YoY">
  <img src="../{{ ticker_data.eps_yoy_change_chart_path }}" alt="EPS YoY">
  {{ ticker_data.yoy_growth_table_html | safe }}
  <img src="../{{ ticker_data.expense_chart_path }}" alt="Expense Chart">
  <img src="../{{ ticker_data.expense_percent_chart_path }}" alt="Expense %">
  {{ ticker_data.expense_table_html | safe }}
  {{ ticker_data.expense_yoy_table_html | safe }}
  <img src="../{{ ticker_data.balance_sheet_chart_path }}" alt="BS Chart">
  {{ ticker_data.balance_sheet_table_html | safe }}
  {% if ticker_data.valuation_chart %}
    <img src="../{{ ticker_data.valuation_chart }}" alt="Valuation">
    {{ ticker_data.valuation_info_table | safe }}
    {{ ticker_data.valuation_data_table | safe }}
  {% endif %}
  {% if ticker_data.unmapped_expense_html %}
    <h2>Unmapped Fields</h2>
    {{ ticker_data.unmapped_expense_html | safe }}
  {% endif %}
  {% if ticker_data.implied_growth_chart_path %}
    <img src="../{{ ticker_data.implied_growth_chart_path }}" alt="Implied Growth">
  {% endif %}
</body>
</html>"""
    create_template(f"{TEMPLATES_DIR}/ticker_template.html", ticker_template)

def create_home_page(tickers, dashboard_table, avg_values, spy_qqq_growth, earnings_past, earnings_upcoming):
    tpl = env.get_template('home_template.html')
    with open('index.html', 'w', encoding='utf-8') as f:
        f.write(tpl.render(
            tickers=tickers,
            dashboard_table=dashboard_table,
            dashboard_data=avg_values,
            spy_qqq_growth=spy_qqq_growth,
            earnings_past=earnings_past,
            earnings_upcoming=earnings_upcoming
        ))

def prepare_and_generate_ticker_pages(tickers):
    ensure_directory_exists(PAGES_DIR)
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        for t in tickers:
            name = get_company_short_name(t, cur)
            d = {
                'ticker': t,
                'company_name': name,
                'ticker_info': get_file_content_or_placeholder(f"{CHARTS_DIR}/{t}_ticker_info.html"),
                'revenue_net_income_chart_path': f"{CHARTS_DIR}/{t}_revenue_net_income_chart.png",
                'eps_chart_path': f"{CHARTS_DIR}/{t}_eps_chart.png",
                'financial_table': get_file_content_or_placeholder(f"{CHARTS_DIR}/{t}_rev_net_table.html"),
                'forecast_rev_net_chart_path': f"{CHARTS_DIR}/{t}_Revenue_Net_Income_Forecast.png",
                'forecast_eps_chart_path': f"{CHARTS_DIR}/{t}_EPS_Forecast.png",
                'yoy_growth_table_html': get_file_content_or_placeholder(f"{CHARTS_DIR}/{t}_yoy_growth_tbl.html"),
                'expense_chart_path': f"{CHARTS_DIR}/{t}_rev_expense_chart.png",
                'expense_percent_chart_path': f"{CHARTS_DIR}/{t}_expense_percent_chart.png",
                'expense_table_html': get_file_content_or_placeholder(f"{CHARTS_DIR}/{t}_yearly_financials.html"),
                'expense_yoy_table_html': get_file_content_or_placeholder(f"{CHARTS_DIR}/{t}_yoy_expense_change.html"),
                'balance_sheet_chart_path': f"{CHARTS_DIR}/{t}_balance_sheet_chart.png",
                'balance_sheet_table_html': get_file_content_or_placeholder(f"{CHARTS_DIR}/{t}_balance_sheet_table.html"),
                'revenue_yoy_change_chart_path': f"{CHARTS_DIR}/{t}_revenue_yoy_change.png",
                'eps_yoy_change_chart_path': f"{CHARTS_DIR}/{t}_eps_yoy_change.png",
                'valuation_chart': f"{CHARTS_DIR}/{t}_valuation_chart.png",
                'valuation_info_table': get_file_content_or_placeholder(f"{CHARTS_DIR}/{t}_valuation_info.html"),
                'valuation_data_table': get_file_content_or_placeholder(f"{CHARTS_DIR}/{t}_valuation_table.html"),
                'unmapped_expense_html': get_file_content_or_placeholder(f"{CHARTS_DIR}/{t}_unmapped_fields.html", "No unmapped expenses."),
                'eps_dividend_chart_path': f"{CHARTS_DIR}/{t}_eps_dividend_forecast.png",
                'implied_growth_chart_path': f"{CHARTS_DIR}/{t}_implied_growth_chart.png"
            }
            tpl = env.get_template('ticker_template.html')
            with open(os.path.join(PAGES_DIR, f"{t}_page.html"), 'w', encoding='utf-8') as f:
                f.write(tpl.render(ticker_data=d))

# ✅ This is the correct mini-main entrypoint
def html_generator2():
    ensure_templates_exist()

    tickers = ["AAPL", "MSFT"]  # Replace with your real tickers
    full_dashboard_html = "<p>Sample Dashboard</p>"  # Replace or load actual table
    avg_values = {
        "Nicks_TTM_Value_Average": 18.2,
        "Nicks_Forward_Value_Average": 15.7,
        "Finviz_TTM_Value_Average": 21.1,
        "Finviz_Forward_Value_Average": 17.3
    }
    spy_qqq_growth_html = "<div>SPY/QQQ</div>"
    earnings_past = get_file_content_or_placeholder(f"{CHARTS_DIR}/earnings_past.html")
    earnings_upcoming = get_file_content_or_placeholder(f"{CHARTS_DIR}/earnings_upcoming.html")

    create_home_page(tickers, full_dashboard_html, avg_values, spy_qqq_growth_html, earnings_past, earnings_upcoming)
    prepare_and_generate_ticker_pages(tickers)

if __name__ == "__main__":
    html_generator2()

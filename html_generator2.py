from jinja2 import Environment, FileSystemLoader
import os
import pandas as pd
import sqlite3
import yfinance as yf
import numpy as np

# Path to the database file
db_path = 'Stock Data.db'
env = Environment(loader=FileSystemLoader('templates'))

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
        return open(path, 'r', encoding='utf-8').read()
    except FileNotFoundError:
        return placeholder

def create_home_page(tickers, output_dir,
                     dashboard_table, avg_values,
                     spy_qqq_growth="", earnings_past="", earnings_upcoming=""):
    tpl = env.get_template('home_template.html')
    out = os.path.join(output_dir, 'index.html')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(tpl.render(
            tickers=tickers,
            dashboard_table=dashboard_table,
            dashboard_data=avg_values,
            spy_qqq_growth=spy_qqq_growth,
            earnings_past=earnings_past,
            earnings_upcoming=earnings_upcoming
        ))

def prepare_and_generate_ticker_pages(tickers, output_dir, charts_dir):
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        for t in tickers:
            name = get_company_short_name(t, cur)
            d = {
                'ticker': t,
                'company_name': name,
                'ticker_info':      get_file_content_or_placeholder(f"{charts_dir}/{t}_ticker_info.html"),
                'revenue_net_income_chart_path': f"{charts_dir}/{t}_revenue_net_income_chart.png",
                'eps_chart_path':                f"{charts_dir}/{t}_eps_chart.png",
                'financial_table':   get_file_content_or_placeholder(f"{charts_dir}/{t}_rev_net_table.html"),
                'forecast_rev_net_chart_path': f"{charts_dir}/{t}_Revenue_Net_Income_Forecast.png",
                'forecast_eps_chart_path':     f"{charts_dir}/{t}_EPS_Forecast.png",
                'yoy_growth_table_html':       get_file_content_or_placeholder(f"{charts_dir}/{t}_yoy_growth_tbl.html"),
                'expense_chart_path':          f"{charts_dir}/{t}_rev_expense_chart.png",
                'expense_percent_chart_path':  f"{charts_dir}/{t}_expense_percent_chart.png",
                'expense_table_html':          get_file_content_or_placeholder(f"{charts_dir}/{t}_yearly_financials.html"),
                'expense_yoy_table_html':      get_file_content_or_placeholder(f"{charts_dir}/{t}_yoy_expense_change.html"),
                'balance_sheet_chart_path':    f"{charts_dir}/{t}_balance_sheet_chart.png",
                'balance_sheet_table_html':    get_file_content_or_placeholder(f"{charts_dir}/{t}_balance_sheet_table.html"),
                'revenue_yoy_change_chart_path': f"{charts_dir}/{t}_revenue_yoy_change.png",
                'eps_yoy_change_chart_path':      f"{charts_dir}/{t}_eps_yoy_change.png",
                'valuation_chart':              f"{charts_dir}/{t}_valuation_chart.png",
                'valuation_info_table':         get_file_content_or_placeholder(f"{charts_dir}/{t}_valuation_info.html"),
                'valuation_data_table':         get_file_content_or_placeholder(f"{charts_dir}/{t}_valuation_table.html"),
                'unmapped_expense_html':        get_file_content_or_placeholder(f"{charts_dir}/{t}_unmapped_fields.html", "No unmapped expenses."),
                'eps_dividend_chart_path':      f"{charts_dir}/{t}_eps_dividend_forecast.png",

                # ─── Implied Growth Additions ───
                'implied_growth_chart_path':    f"{charts_dir}/{t}_implied_growth_plot.png",
                'implied_growth_table_html':    get_file_content_or_placeholder(f"{charts_dir}/{t}_implied_growth_summary.html", "No implied growth data available.")
            }

            tpl = env.get_template('ticker_template.html')
            out = os.path.join(output_dir, 'pages', f"{t}_page.html")
            ensure_directory_exists(os.path.dirname(out))
            with open(out, 'w', encoding='utf-8') as f:
                f.write(tpl.render(ticker_data=d))

def html_generator2(tickers, financial_data, full_dashboard_html, avg_values, spy_qqq_growth_html=""):
    past     = get_file_content_or_placeholder("charts/earnings_past.html")
    upcoming = get_file_content_or_placeholder("charts/earnings_upcoming.html")

    create_home_page(
        tickers            = tickers,
        output_dir         = '.',
        dashboard_table    = full_dashboard_html,
        avg_values         = avg_values,
        spy_qqq_growth     = spy_qqq_growth_html,
        earnings_past      = past,
        earnings_upcoming  = upcoming
    )

    prepare_and_generate_ticker_pages(tickers, '.', 'charts/')

# Mini main entrypoint
def run_html_generator2():
    print("This is a legacy entrypoint placeholder. Please call html_generator2() from your main script.")

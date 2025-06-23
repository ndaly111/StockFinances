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

def ensure_templates_exist():
    # (Your existing ticker_template.html content goes here)
    ticker_template_content = """<!DOCTYPE html>
<html lang="en">
<head>…</head>
<body>…</body>
</html>"""
    create_template(
        os.path.join('templates','ticker_template.html'),
        ticker_template_content
    )

def get_file_content_or_placeholder(file_path, placeholder="No data available"):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return placeholder

def create_home_page(tickers, output_dir,
                     dashboard_table, avg_values,
                     spy_qqq_growth="", earnings_past="", earnings_upcoming=""):
    template = env.get_template('home_template.html')
    out_path = os.path.join(output_dir, 'index.html')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(template.render(
            tickers = tickers,
            dashboard_table = dashboard_table,
            dashboard_data = avg_values,
            spy_qqq_growth = spy_qqq_growth,
            earnings_past = earnings_past,
            earnings_upcoming = earnings_upcoming
        ))

def prepare_and_generate_ticker_pages(tickers, output_dir, charts_output_dir):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        for ticker in tickers:
            company_name = get_company_short_name(ticker, cursor)
            ticker_data = {
                'ticker': ticker,
                'company_name': company_name,
                'ticker_info': get_file_content_or_placeholder(
                    f"{charts_output_dir}/{ticker}_ticker_info.html"
                ),
                'revenue_net_income_chart_path': 
                    f"{charts_output_dir}/{ticker}_revenue_net_income_chart.png",
                'eps_chart_path': 
                    f"{charts_output_dir}/{ticker}_eps_chart.png",
                'financial_table': get_file_content_or_placeholder(
                    f"{charts_output_dir}/{ticker}_rev_net_table.html"
                ),
                'forecast_rev_net_chart_path':
                    f"{charts_output_dir}/{ticker}_Revenue_Net_Income_Forecast.png",
                'forecast_eps_chart_path':
                    f"{charts_output_dir}/{ticker}_EPS_Forecast.png",
                'yoy_growth_table_html': get_file_content_or_placeholder(
                    f"{charts_output_dir}/{ticker}_yoy_growth_tbl.html"
                ),
                'expense_chart_path': 
                    f"{charts_output_dir}/{ticker}_rev_expense_chart.png",
                'expense_percent_chart_path':
                    f"{charts_output_dir}/{ticker}_expense_percent_chart.png",
                'expense_table_html': get_file_content_or_placeholder(
                    f"{charts_output_dir}/{ticker}_yearly_financials.html"
                ),
                'expense_yoy_table_html': get_file_content_or_placeholder(
                    f"{charts_output_dir}/{ticker}_yoy_expense_change.html"
                ),
                'balance_sheet_chart_path':
                    f"{charts_output_dir}/{ticker}_balance_sheet_chart.png",
                'balance_sheet_table_html': get_file_content_or_placeholder(
                    f"{charts_output_dir}/{ticker}_balance_sheet_table.html"
                ),
                'revenue_yoy_change_chart_path':
                    f"{charts_output_dir}/{ticker}_revenue_yoy_change.png",
                'eps_yoy_change_chart_path':
                    f"{charts_output_dir}/{ticker}_eps_yoy_change.png",
                'valuation_chart': 
                    f"{charts_output_dir}/{ticker}_valuation_chart.png",
                'valuation_info_table': get_file_content_or_placeholder(
                    f"{charts_output_dir}/{ticker}_valuation_info.html"
                ),
                'valuation_data_table': get_file_content_or_placeholder(
                    f"{charts_output_dir}/{ticker}_valuation_table.html"
                ),
                'unmapped_expense_html': get_file_content_or_placeholder(
                    f"{charts_output_dir}/{ticker}_unmapped_fields.html",
                    "No unmapped expenses."
                ),
                'eps_dividend_chart_path': 
                    f"{charts_output_dir}/{ticker}_eps_dividend_forecast.png"
            }
            create_ticker_page(ticker, ticker_data, output_dir)

def create_ticker_page(ticker, ticker_data, output_dir):
    template = env.get_template('ticker_template.html')
    page_path = os.path.join(output_dir, 'pages', f'{ticker}_page.html')
    ensure_directory_exists(os.path.dirname(page_path))
    with open(page_path, 'w', encoding='utf-8') as f:
        f.write(template.render(ticker_data=ticker_data))

# ─────────────────────────────────────────────────────────────────────────────
# Here is your **old** generate_dashboard_table() logic, verbatim:
# ─────────────────────────────────────────────────────────────────────────────
def generate_dashboard_table(dashboard_data):
    # — Pad each row to 10 elements so DataFrame(...) never errors
    padded = []
    for row in dashboard_data:
        if len(row) < 10:
            row = list(row) + ["-"] * (10 - len(row))
        padded.append(row)

    df = pd.DataFrame(padded, columns=[
        "Ticker", "Share Price", "Nicks TTM Valuation", "Nicks TTM Value",
        "Nicks Forward Valuation", "Nicks Forward Value",
        "Finviz TTM Valuation",    "Finviz TTM Value",
        "Finviz Forward Valuation","Finviz Forward Value"
    ])
    df.drop(columns=[
        "Nicks TTM Valuation", "Nicks Forward Valuation",
        "Finviz TTM Valuation","Finviz Forward Valuation"
    ], inplace=True)
    df["Ticker"] = df["Ticker"].apply(
        lambda t: f'<a href="pages/{t}_page.html">{t}</a>'
    )

    # — Create hidden numeric twins for the “% Value” columns
    for col in ["Nicks TTM Value", "Nicks Forward Value",
                "Finviz TTM Value", "Finviz Forward Value"]:
        df[col + "_num"] = (
            df[col].astype(str)
                   .str.rstrip("%")
                   .replace("-", np.nan)
                   .astype(float)
        )

    # Sort rows by Nicks TTM Value_num desc
    df.sort_values("Nicks TTM Value_num", ascending=False, inplace=True)

    # Helper to strip “%” and parse floats
    def pct(x):
        try:
            return float(x.strip('%'))
        except:
            return None

    # Build your averages/medians exactly as before
    avgv = {
        'Nicks_TTM_Value_Average':
            df["Nicks TTM Value"].apply(pct).mean(),
        'Nicks_Forward_Value_Average':
            df["Nicks Forward Value"].apply(pct).mean(),
        'Finviz_TTM_Value_Average':
            df["Finviz TTM Value"].apply(pct).mean(),
        'Nicks_TTM_Value_Median':
            df["Nicks TTM Value"].apply(pct).median(),
        'Nicks_Forward_Value_Median':
            df["Nicks Forward Value"].apply(pct).median(),
        'Finviz_TTM_Value_Median':
            df["Finviz TTM Value"].apply(pct).median(),
        'Finviz_Forward_Value_Median':
            df["Finviz Forward Value"].apply(pct).median()
    }

    rows = [
        ["Average",
         f"{avgv['Nicks_TTM_Value_Average']:.1f}%",
         f"{avgv['Nicks_Forward_Value_Average']:.1f}%",
         f"{avgv['Finviz_TTM_Value_Average']:.1f}%",
         f"{avgv['Finviz_Forward_Value_Median']:.1f}%"],
        ["Median",
         f"{avgv['Nicks_TTM_Value_Median']:.1f}%",
         f"{avgv['Nicks_Forward_Value_Median']:.1f}%",
         f"{avgv['Finviz_TTM_Value_Median']:.1f}%",
         f"{avgv['Finviz_Forward_Value_Median']:.1f}%"]
    ]
    avg_table = pd.DataFrame(rows, columns=[
        "Metric","Nicks TTM Value","Nicks Forward Value",
        "Finviz TTM Value","Finviz Forward Value"
    ]).to_html(index=False, escape=False, classes='table table-striped')

    dash_table = df.to_html(
        index=False, escape=False,
        classes='table table-striped', table_id="sortable-table"
    )

    # Write combined HTML
    with open('charts/dashboard.html','w',encoding='utf-8') as f:
        f.write(avg_table + dash_table)

    return avg_table + dash_table, avgv

# ─────────────────────────────────────────────────────────────────────────────

def html_generator2(tickers, financial_data,
                    full_dashboard_html, avg_values,
                    spy_qqq_growth_html=""):
    ensure_templates_exist()

    past = get_file_content_or_placeholder("charts/earnings_past.html")
    upcoming = get_file_content_or_placeholder("charts/earnings_upcoming.html")

    create_home_page(
        tickers = tickers,
        output_dir = '.',
        dashboard_table = full_dashboard_html,
        avg_values = avg_values,
        spy_qqq_growth = spy_qqq_growth_html,
        earnings_past = past,
        earnings_upcoming = upcoming
    )

    prepare_and_generate_ticker_pages(tickers, '.', 'charts/')

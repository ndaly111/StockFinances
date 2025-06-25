from jinja2 import Environment, FileSystemLoader
import os
import pandas as pd
import sqlite3
import yfinance as yf
import numpy as np

# ───────────────────────────────────────────────────────────
# Basic setup (unchanged legacy)
# ───────────────────────────────────────────────────────────
db_path = 'Stock Data.db'
env = Environment(loader=FileSystemLoader('templates'))

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def create_template(template_path: str, content: str):
    ensure_directory_exists(os.path.dirname(template_path))
    if os.path.exists(template_path):
        with open(template_path, 'r', encoding='utf-8') as f:
            if f.read() == content:
                return
    with open(template_path, 'w', encoding='utf-8') as f:
        f.write(content)

# ───────────────────────────────────────────────────────────
# DB helpers (legacy)
# ───────────────────────────────────────────────────────────
def get_company_short_name(ticker, cursor):
    cursor.execute("SELECT short_name FROM Tickers_Info WHERE ticker = ?", (ticker,))
    row = cursor.fetchone()
    if row and row[0]:
        return row[0]
    short_name = yf.Ticker(ticker).info.get('shortName', '').strip()
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

# ───────────────────────────────────────────────────────────
# Home-page builder (legacy)
# ───────────────────────────────────────────────────────────
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

# ───────────────────────────────────────────────────────────
# Per-ticker page builder  (added implied-growth files)
# ───────────────────────────────────────────────────────────
def prepare_and_generate_ticker_pages(tickers, output_dir, charts_dir):
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        for t in tickers:
            name = get_company_short_name(t, cur)
            d = {
                'ticker': t,
                'company_name': name,

                # ─── existing files ───
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

                # ─── new implied-growth outputs ───
                'implied_growth_chart_path':    f"{charts_dir}/{t}_implied_growth_plot.png",
                'implied_growth_table_html':    get_file_content_or_placeholder(
                                                     f"{charts_dir}/{t}_implied_growth_summary.html",
                                                     "No implied growth data available."
                                                 )
            }

            tpl = env.get_template('ticker_template.html')
            out = os.path.join(output_dir, 'pages', f"{t}_page.html")
            ensure_directory_exists(os.path.dirname(out))
            with open(out, 'w', encoding='utf-8') as f:
                f.write(tpl.render(ticker_data=d))

# ───────────────────────────────────────────────────────────
# **generate_dashboard_table**   ← restored exactly as legacy
# ───────────────────────────────────────────────────────────
def generate_dashboard_table(dashboard_data):
    df = pd.DataFrame(dashboard_data, columns=[
        "Ticker", "Share Price",
        "Nick's TTM Value", "Nick's Forward Value",
        "Finviz TTM Value", "Finviz Forward Value"
    ])

    # hyperlink tickers to their pages
    df["Ticker"] = df["Ticker"].apply(lambda t: f'<a href="pages/{t}_page.html">{t}</a>')

    # create numeric versions for sorting / averaging
    for col in ["Nick's TTM Value", "Nick's Forward Value",
                "Finviz TTM Value", "Finviz Forward Value"]:
        df[col + "_num"] = (
            df[col].astype(str)
                   .str.rstrip('%')
                   .replace('-', np.nan)
                   .astype(float)
        )

    # sort by Nick’s TTM %
    df.sort_values("Nick's TTM Value_num", ascending=False, inplace=True)

    # averages / medians
    ttm  = df["Nick's TTM Value_num"].dropna()
    fwd  = df["Nick's Forward Value_num"].dropna()
    fttm = df["Finviz TTM Value_num"].dropna()
    ffwd = df["Finviz Forward Value_num"].dropna()

    def fmt(x): return f"{x:.1f}%" if pd.notnull(x) else "–"

    summary = [
        ["Average", fmt(ttm.mean()), fmt(fwd.mean()), fmt(fttm.mean()), fmt(ffwd.mean())],
        ["Median",  fmt(ttm.median()), fmt(fwd.median()), fmt(fttm.median()), fmt(ffwd.median())]
    ]
    avg_table = pd.DataFrame(summary, columns=[
        "Metric", "Nick's TTM Value", "Nick's Forward Value",
        "Finviz TTM Value", "Finviz Forward Value"
    ]).to_html(index=False, escape=False, classes='table table-striped')

    display = df[[
        "Ticker", "Share Price",
        "Nick's TTM Value", "Nick's Forward Value",
        "Finviz TTM Value", "Finviz Forward Value"
    ]]
    dash_table = display.to_html(
        index=False, escape=False,
        classes='table table-striped', table_id="sortable-table"
    )

    # write combined dashboard file
    with open('charts/dashboard.html', 'w', encoding='utf-8') as f:
        f.write(avg_table + dash_table)

    return avg_table + dash_table, {
        'Nicks_TTM_Avg':  ttm.mean(),  'Nicks_TTM_Med':  ttm.median(),
        'Nicks_FWD_Avg':  fwd.mean(),  'Nicks_FWD_Med':  fwd.median(),
        'Finviz_TTM_Avg': fttm.mean() if not fttm.empty else None,
        'Finviz_TTM_Med': fttm.median() if not fttm.empty else None,
        'Finviz_FWD_Avg': ffwd.mean() if not ffwd.empty else None,
        'Finviz_FWD_Med': ffwd.median() if not ffwd.empty else None
    }

# ───────────────────────────────────────────────────────────
# Top-level wrapper (legacy mini-main)
# ───────────────────────────────────────────────────────────
def html_generator2(tickers, financial_data,
                    full_dashboard_html, avg_values,
                    spy_qqq_growth_html=""):

    past     = get_file_content_or_placeholder("charts/earnings_past.html")
    upcoming = get_file_content_or_placeholder("charts/earnings_upcoming.html")

    create_home_page(
        tickers=tickers,
        output_dir='.',
        dashboard_table=full_dashboard_html,
        avg_values=avg_values,
        spy_qqq_growth=spy_qqq_growth_html,
        earnings_past=past,
        earnings_upcoming=upcoming
    )

    prepare_and_generate_ticker_pages(tickers, '.', 'charts/')

# legacy placeholder – main_remote.py calls html_generator2 directly
def mini_main_placeholder():
    print("html_generator2 is designed to be called from main_remote.py")

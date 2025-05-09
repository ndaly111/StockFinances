# start of main.py
import os
import sqlite3
import ticker_manager
from datetime import datetime
from annual_and_ttm_update import annual_and_ttm_update
from html_generator import create_html_for_tickers
from balance_sheet_data_fetcher import (
    fetch_balance_sheet_data,
    check_missing_balance_sheet_data,
    is_balance_sheet_data_outdated,
    fetch_balance_sheet_data_from_yahoo,
    store_fetched_balance_sheet_data
)
from balancesheet_chart import (
    fetch_balance_sheet_data as _fs,  # avoid name clash
    plot_chart,
    format_value,
    create_and_save_table
)
import pandas as pd
from Forward_data import scrape_forward_data
from forecasted_earnings_chart import generate_forecast_charts_and_tables
from bs4 import BeautifulSoup
from ticker_info import prepare_data_for_display, generate_html_table
import requests
from html_generator2 import html_generator2, generate_dashboard_table
from valuation_update import valuation_update, process_update_growth_csv

# ←— NEW: import your earnings generator
from generate_earnings_tables import generate_earnings_tables

# Constants
TICKERS_FILE_PATH     = 'tickers.csv'
DB_PATH               = 'Stock Data.db'
file_path             = "update_growth.csv"
charts_output_dir     = 'charts/'
is_remote             = True
table_name            = 'ForwardFinancialData'

def manage_tickers(path, is_remote=False):
    current_tickers = ticker_manager.read_tickers(path)
    current_tickers = ticker_manager.modify_tickers(current_tickers, is_remote)
    sorted_tickers = sorted(current_tickers)
    ticker_manager.write_tickers(sorted_tickers, path)
    return sorted_tickers

def establish_database_connection(db_path):
    full = os.path.abspath(db_path)
    if not os.path.exists(full):
        print(f"Database not found: {full}")
        return None
    return sqlite3.connect(full)

def log_average_valuations(avg_values, path):
    if path != 'tickers.csv':
        return
    today_str = datetime.now().strftime('%Y-%m-%d')
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS AverageValuations (
                date TEXT PRIMARY KEY,
                avg_ttm REAL, avg_forward REAL, avg_finviz REAL
            );
        ''')
        c.execute('SELECT 1 FROM AverageValuations WHERE date=?', (today_str,))
        if not c.fetchone():
            c.execute('''
                INSERT INTO AverageValuations (date, avg_ttm, avg_forward, avg_finviz)
                VALUES (?, ?, ?, ?)
            ''', (
                today_str,
                avg_values['Nicks_TTM_Value_Average'],
                avg_values['Nicks_Forward_Value_Average'],
                avg_values['Finviz_TTM_Value_Average']
            ))
            conn.commit()

def balancesheet_chart(ticker, out_dir):
    data = fetch_balance_sheet_data(ticker)
    if data is None: return
    plot_chart(data, out_dir, ticker)
    data['Debt_to_Equity_Ratio'] = data['Total_Debt'] / data['Total_Equity']
    create_and_save_table(data, out_dir, ticker)

def fetch_and_update_balance_sheet_data(ticker, cursor):
    current = fetch_balance_sheet_data(ticker)
    missing = check_missing_balance_sheet_data(ticker, cursor)
    outdated = is_balance_sheet_data_outdated(current)
    if missing or outdated:
        fresh = fetch_balance_sheet_data_from_yahoo(ticker)
        if fresh:
            store_fetched_balance_sheet_data(cursor, fresh)

def fetch_10_year_treasury_yield():
    url = "https://fred.stlouisfed.org/series/GS10"
    try:
        r = requests.get(url); r.raise_for_status()
        soup = BeautifulSoup(r.content, 'html.parser')
        span = soup.find("span", class_="series-meta-observation-value")
        return span.text.strip() if span else "N/A"
    except:
        return "N/A"

def main():
    # 0. Prep
    treasury_yield = fetch_10_year_treasury_yield()
    tickers = manage_tickers(TICKERS_FILE_PATH, is_remote=True)
    conn = establish_database_connection(DB_PATH)
    if conn is None:
        return

    dashboard_data = []
    try:
        cur = conn.cursor()
        process_update_growth_csv(file_path, DB_PATH)

        # 1. Per‐ticker pipelines
        for ticker in tickers:
            annual_and_ttm_update(ticker, DB_PATH)
            fetch_and_update_balance_sheet_data(ticker, cur)
            balancesheet_chart(ticker, charts_output_dir)
            scrape_forward_data(ticker, DB_PATH, table_name)
            generate_forecast_charts_and_tables(ticker, DB_PATH, charts_output_dir)
            prep_data, marketcap = prepare_data_for_display(ticker, treasury_yield)
            generate_html_table(prep_data, ticker)
            valuation_update(ticker, cur, treasury_yield, marketcap, dashboard_data)

        # 2. Dashboard table + averages
        full_dashboard_html, avg_values = generate_dashboard_table(dashboard_data)
        log_average_valuations(avg_values, TICKERS_FILE_PATH)

        # 3. SPY/QQQ growth
        from index_growth_table import index_growth
        spy_qqq_html = index_growth(treasury_yield)

        # ←— NEW: generate earnings HTML fragments
        generate_earnings_tables()

        # 4. Final page assembly
        html_generator2(
            tickers,
            {},  # financial_data not used by html2
            full_dashboard_html,
            avg_values,
            spy_qqq_html
        )

    finally:
        conn.close()

if __name__ == "__main__":
    main()
# start of main_remote.py
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
    fetch_balance_sheet_data as bs_fetch,  # renamed to avoid shadowing
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
from expense_reports import generate_expense_reports
from html_generator2 import html_generator2, generate_dashboard_table
from valuation_update import valuation_update, process_update_growth_csv
from index_growth_table import index_growth
# option A – rename the import
from eps_dividend_generator import eps_dividend_generator

# ←— NEW: import the function
from generate_earnings_tables import generate_earnings_tables

# Constants
TICKERS_FILE_PATH = 'tickers.csv'
db_path           = 'Stock Data.db'
DB_PATH           = 'Stock Data.db'
file_path         = "update_growth.csv"
charts_output_dir = 'charts/'
is_remote         = True
table_name        = 'ForwardFinancialData'

def manage_tickers(TICKERS_FILE_PATH, is_remote=False):
    current_tickers = ticker_manager.read_tickers(TICKERS_FILE_PATH)
    current_tickers = ticker_manager.modify_tickers(current_tickers, is_remote)
    sorted_tickers  = sorted(current_tickers)
    ticker_manager.write_tickers(sorted_tickers, TICKERS_FILE_PATH)
    return sorted_tickers

def establish_database_connection(db_path):
    db_full_path = os.path.abspath(db_path)
    if not os.path.exists(db_full_path):
        print(f"Database file not found: {db_full_path}")
        return None
    return sqlite3.connect(db_full_path)

def log_average_valuations(avg_values, TICKERS_FILE_PATH):
    if TICKERS_FILE_PATH != 'tickers.csv':
        return
    current_date = datetime.now().strftime('%Y-%m-%d')
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS AverageValuations (
                date DATE PRIMARY KEY,
                avg_ttm_valuation REAL,
                avg_forward_valuation REAL,
                avg_finviz_valuation REAL
            );
        ''')
        cursor.execute('SELECT 1 FROM AverageValuations WHERE date = ?;', (current_date,))
        if not cursor.fetchone():
            cursor.execute('''
                INSERT INTO AverageValuations (date, avg_ttm_valuation, avg_forward_valuation, avg_finviz_valuation)
                VALUES (?, ?, ?, ?);
            ''', (
                current_date,
                avg_values['Nicks_TTM_Value_Average'],
                avg_values['Nicks_Forward_Value_Average'],
                avg_values['Finviz_TTM_Value_Average']
            ))
            conn.commit()

def balancesheet_chart(ticker, charts_output_dir):
    data = bs_fetch(ticker)
    if data is not None:
        plot_chart(data, charts_output_dir, ticker)

        # --- BEGIN safe Debt/Equity calculation ---
        debt   = data.get('Total_Debt')
        equity = data.get('Total_Equity')
        if debt is None or equity is None or pd.isna(debt) or pd.isna(equity) or equity == 0:
            print(f"Skipping Debt to Equity for {ticker}: Debt={debt}, Equity={equity}")
            data['Debt_to_Equity_Ratio'] = None
        else:
            data['Debt_to_Equity_Ratio'] = debt / equity
        # --- END patch ---

        create_and_save_table(data, charts_output_dir, ticker)

def fetch_and_update_balance_sheet_data(ticker, cursor):
    current_data = fetch_balance_sheet_data(ticker)
    if check_missing_balance_sheet_data(ticker, cursor) or is_balance_sheet_data_outdated(current_data):
        fresh_data = fetch_balance_sheet_data_from_yahoo(ticker)
        if fresh_data:
            store_fetched_balance_sheet_data(cursor, fresh_data)

import yfinance as yf
def fetch_10_year_treasury_yield():
    try:
        bond = yf.Ticker("^TNX")
        return bond.info.get('regularMarketPrice')  # TNX is in tenths of a percent
    except Exception as e:
        print("YF fallback error:", e)
        return None

def main():
    financial_data = {}
    dashboard_data = []
    treasury_yield = fetch_10_year_treasury_yield()

    sorted_tickers = manage_tickers(TICKERS_FILE_PATH, is_remote=True)
    conn = establish_database_connection(DB_PATH)
    if conn is None:
        return

    try:
        cursor = conn.cursor()
        process_update_growth_csv(file_path, db_path)

        for ticker in sorted_tickers:
            annual_and_ttm_update(ticker, db_path)
            fetch_and_update_balance_sheet_data(ticker, cursor)
            balancesheet_chart(ticker, charts_output_dir)
            scrape_forward_data(ticker, db_path, table_name)
            generate_forecast_charts_and_tables(ticker, db_path, charts_output_dir)
            prepared_data, marketcap = prepare_data_for_display(ticker, treasury_yield)
            generate_html_table(prepared_data, ticker)
            valuation_update(ticker, cursor, treasury_yield, marketcap, dashboard_data)

            # ── COMMIT BEFORE EXPENSE REPORTS ─────────────────────────────
            conn.commit()

            # ←— NEW: generate expense reports
            generate_expense_reports(ticker)

        eps_dividend_generator()

        full_dashboard_html, avg_values = generate_dashboard_table(dashboard_data)
        log_average_valuations(avg_values, TICKERS_FILE_PATH)

        spy_qqq_growth_html = index_growth(treasury_yield)

        # ←— NEW: generate earnings tables
        generate_earnings_tables()

        html_generator2(
            sorted_tickers,
            financial_data,
            full_dashboard_html,
            avg_values,
            spy_qqq_growth_html
        )

    finally:
        conn.close()

if __name__ == "__main__":
    main()

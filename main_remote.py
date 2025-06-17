# start of main_remote.py
import os
import sqlite3
import ticker_manager
from datetime import datetime

from annual_and_ttm_update import annual_and_ttm_update

# DB‐backed balance‐sheet fetcher (needs cursor)
import balance_sheet_data_fetcher as bs_db

# Chart routines (only need ticker)
from balancesheet_chart import fetch_balance_sheet_data as fetch_bs_chart, plot_chart, format_value, create_and_save_table

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

# ←— NEW: import the EPS-Dividend mini-main
from eps_dividend_generator import eps_dividend_generator

# ←— NEW: import earnings tables generator
from generate_earnings_tables import generate_earnings_tables

# Constants
TICKERS_FILE_PATH = 'tickers.csv'
DB_PATH = 'Stock Data.db'
file_path = "update_growth.csv"
charts_output_dir = 'charts/'
is_remote = True
table_name = 'ForwardFinancialData'

def manage_tickers(TICKERS_FILE_PATH, is_remote=False):
    current_tickers = ticker_manager.read_tickers(TICKERS_FILE_PATH)
    current_tickers = ticker_manager.modify_tickers(current_tickers, is_remote)
    sorted_tickers = sorted(current_tickers)
    ticker_manager.write_tickers(sorted_tickers, TICKERS_FILE_PATH)
    return sorted_tickers

def establish_database_connection(db_path):
    full = os.path.abspath(db_path)
    if not os.path.exists(full):
        print(f"Database file not found: {full}")
        return None
    return sqlite3.connect(full)

def log_average_valuations(avg_values, TICKERS_FILE_PATH):
    if TICKERS_FILE_PATH != 'tickers.csv':
        return
    today = datetime.now().strftime('%Y-%m-%d')
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute('''
            CREATE TABLE IF NOT EXISTS AverageValuations (
                date DATE PRIMARY KEY,
                avg_ttm_valuation REAL,
                avg_forward_valuation REAL,
                avg_finviz_valuation REAL
            );
        ''')
        cur.execute('SELECT 1 FROM AverageValuations WHERE date = ?;', (today,))
        if not cur.fetchone():
            cur.execute('''
                INSERT INTO AverageValuations
                  (date, avg_ttm_valuation, avg_forward_valuation, avg_finviz_valuation)
                VALUES (?, ?, ?, ?);
            ''', (
                today,
                avg_values['Nicks_TTM_Value_Average'],
                avg_values['Nicks_Forward_Value_Average'],
                avg_values['Finviz_TTM_Value_Average']
            ))
            conn.commit()

def balancesheet_chart(ticker, charts_output_dir):
    data = fetch_bs_chart(ticker)
    if data is None:
        return
    plot_chart(data, charts_output_dir, ticker)

    # --- Debt/Equity patch ---
    debt   = data.get('Total_Debt')
    equity = data.get('Total_Equity')
    if debt is None or equity is None or pd.isna(debt) or pd.isna(equity) or equity == 0:
        print(f"Skipping D/E ratio for {ticker}: Debt={debt}, Equity={equity}")
    else:
        data['Debt_to_Equity_Ratio'] = debt / equity

    create_and_save_table(data, charts_output_dir, ticker)

def fetch_and_update_balance_sheet_data(ticker, cursor):
    # use the DB‐backed fetcher (needs cursor)
    current_data = bs_db.fetch_balance_sheet_data(ticker, cursor)
    missing = bs_db.check_missing_balance_sheet_data(ticker, cursor)
    outdated = bs_db.is_balance_sheet_data_outdated(current_data)
    if missing or outdated:
        fresh = bs_db.fetch_balance_sheet_data_from_yahoo(ticker)
        if fresh:
            bs_db.store_fetched_balance_sheet_data(cursor, fresh)

import yfinance as yf
def fetch_10_year_treasury_yield():
    try:
        bond = yf.Ticker("^TNX")
        return bond.info.get('regularMarketPrice')
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
        cur = conn.cursor()
        process_update_growth_csv(file_path, DB_PATH)

        for ticker in sorted_tickers:
            annual_and_ttm_update(ticker, DB_PATH)
            fetch_and_update_balance_sheet_data(ticker, cur)
            balancesheet_chart(ticker, charts_output_dir)
            scrape_forward_data(ticker, DB_PATH, table_name)
            generate_forecast_charts_and_tables(ticker, DB_PATH, charts_output_dir)
            prepared_data, marketcap = prepare_data_for_display(ticker, treasury_yield)
            generate_html_table(prepared_data, ticker)
            valuation_update(ticker, cur, treasury_yield, marketcap, dashboard_data)
            generate_expense_reports(ticker)

        full_dashboard_html, avg_values = generate_dashboard_table(dashboard_data)
        log_average_valuations(avg_values, TICKERS_FILE_PATH)

        spy_qqq_growth_html = index_growth(treasury_yield)

        generate_earnings_tables()

        # ←— NEW: fetch dividends & generate EPS vs Dividend charts
        eps_dividend_generator()

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

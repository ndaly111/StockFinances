# main_remote.py

import os
import sqlite3
import ticker_manager
from datetime import datetime
from annual_and_ttm_update import annual_and_ttm_update
from html_generator import create_html_for_tickers
from balance_sheet_data_fetcher import (
    fetch_balance_sheet_data   as db_fetch_balance_sheet_data,
    check_missing_balance_sheet_data,
    is_balance_sheet_data_outdated,
    fetch_balance_sheet_data_from_yahoo,
    store_fetched_balance_sheet_data
)
from balancesheet_chart import (
    fetch_balance_sheet_data   as chart_fetch_balance_sheet_data,
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
from eps_dividend_generator import eps_dividend_generator
from generate_earnings_tables import generate_earnings_tables
import yfinance as yf

# Constants
TICKERS_FILE_PATH  = 'tickers.csv'
DB_PATH            = 'Stock Data.db'
UPDATE_CSV_PATH    = "update_growth.csv"
CHARTS_DIR         = 'charts/'
FORWARD_TABLE_NAME = 'ForwardFinancialData'


def manage_tickers(tickers_file_path, is_remote=False):
    current = ticker_manager.read_tickers(tickers_file_path)
    updated = ticker_manager.modify_tickers(current, is_remote)
    sorted_tocks = sorted(updated)
    ticker_manager.write_tickers(sorted_tocks, tickers_file_path)
    return sorted_tocks


def establish_database_connection(db_path):
    full = os.path.abspath(db_path)
    if not os.path.exists(full):
        print(f"❌ Database file not found: {full}")
        return None
    return sqlite3.connect(full)


def log_average_valuations(avg_vals, tickers_file_path):
    if tickers_file_path != TICKERS_FILE_PATH:
        return
    today = datetime.now().strftime('%Y-%m-%d')
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS AverageValuations(
              date TEXT PRIMARY KEY,
              avg_ttm_valuation REAL,
              avg_forward_valuation REAL,
              avg_finviz_valuation REAL
            );
        ''')
        c.execute('SELECT 1 FROM AverageValuations WHERE date=?', (today,))
        if not c.fetchone():
            c.execute('''
                INSERT INTO AverageValuations
                  (date, avg_ttm_valuation, avg_forward_valuation, avg_finviz_valuation)
                VALUES (?, ?, ?, ?)
            ''', (
                today,
                avg_vals['Nicks_TTM_Value_Average'],
                avg_vals['Nicks_Forward_Value_Average'],
                avg_vals['Finviz_TTM_Value_Average']
            ))
            conn.commit()


def balancesheet_chart(ticker, charts_output_dir):
    data = chart_fetch_balance_sheet_data(ticker)
    if data is None:
        return

    plot_chart(data, charts_output_dir, ticker)

    debt   = data.get('Total_Debt')
    equity = data.get('Total_Equity')
    if debt is None or equity in (None, 0) or pd.isna(debt) or pd.isna(equity):
        data['Debt_to_Equity_Ratio'] = None
    else:
        data['Debt_to_Equity_Ratio'] = debt / equity

    create_and_save_table(data, charts_output_dir, ticker)


def fetch_and_update_balance_sheet_data(ticker, cursor):
    existing = db_fetch_balance_sheet_data(ticker, cursor)
    if check_missing_balance_sheet_data(ticker, cursor) or is_balance_sheet_data_outdated(existing):
        fresh = fetch_balance_sheet_data_from_yahoo(ticker)
        if fresh:
            store_fetched_balance_sheet_data(cursor, fresh)


def fetch_10_year_treasury_yield():
    try:
        tnx = yf.Ticker("^TNX")
        return tnx.info.get('regularMarketPrice')  # in tenths of a percent
    except Exception as e:
        print("YF TNX error:", e)
        return None


def main():
    dashboard_data = []
    treasury_yield = fetch_10_year_treasury_yield()

    tickers = manage_tickers(TICKERS_FILE_PATH, is_remote=True)
    conn    = establish_database_connection(DB_PATH)
    if conn is None:
        return

    try:
        cur = conn.cursor()
        process_update_growth_csv(UPDATE_CSV_PATH, DB_PATH)

        for tkr in tickers:
            annual_and_ttm_update(tkr, DB_PATH)
            fetch_and_update_balance_sheet_data(tkr, cur)
            balancesheet_chart(tkr, CHARTS_DIR)
            scrape_forward_data(tkr, DB_PATH, FORWARD_TABLE_NAME)
            generate_forecast_charts_and_tables(tkr, DB_PATH, CHARTS_DIR)
            pdata, mcap = prepare_data_for_display(tkr, treasury_yield)
            generate_html_table(pdata, tkr)
            valuation_update(tkr, cur, treasury_yield, mcap, dashboard_data)

            # <-- Correct call with ticker argument
            generate_expense_reports(tkr)

        eps_dividend_generator()

        full_dashboard_html, avg_vals = generate_dashboard_table(dashboard_data)
        log_average_valuations(avg_vals, TICKERS_FILE_PATH)

        spy_qqq_html = index_growth(treasury_yield)

        generate_earnings_tables()

        html_generator2(
            tickers,
            {},  # financial_data placeholder
            full_dashboard_html,
            avg_vals,
            spy_qqq_html
        )

    finally:
        conn.close()


if __name__ == "__main__":
    main()

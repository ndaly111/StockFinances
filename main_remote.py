#!/usr/bin/env python3
# main_remote.py – 2025-08-08  (calls economic-data generator first)
# ────────────────────────────────────────────────────────────────────
import os, sqlite3, pandas as pd, yfinance as yf, math
from datetime import datetime, timezone
from pathlib import Path
import time  # ← added for optional polite delay

import ticker_manager
from generate_economic_data    import generate_economic_data
from annual_and_ttm_update     import annual_and_ttm_update
from html_generator            import create_html_for_tickers
from balance_sheet_data_fetcher import (
    fetch_balance_sheet_data, check_missing_balance_sheet_data,
    is_balance_sheet_data_outdated, fetch_balance_sheet_data_from_yahoo,
    store_fetched_balance_sheet_data
)
from balancesheet_chart        import (
    fetch_balance_sheet_data as fetch_bs_for_chart,
    plot_chart, create_and_save_table
)
from implied_growth_summary    import generate_all_summaries
from Forward_data              import scrape_forward_data
from forecasted_earnings_chart import generate_forecast_charts_and_tables
from ticker_info               import prepare_data_for_display, generate_html_table
from expense_reports           import generate_expense_reports
from html_generator2           import html_generator2, generate_dashboard_table
from valuation_update          import valuation_update, process_update_growth_csv
from index_growth_table        import index_growth
from eps_dividend_generator    import eps_dividend_generator
from index_growth_charts       import render_index_growth_charts
from generate_earnings_tables  import generate_earnings_tables
from backfill_index_growth import backfill_index_growth
from generate_index_growth_pages import generate_index_growth_pages


from generate_segment_charts import generate_segment_charts_for_ticker
# Constants
# ────────────────────────────────────────────────────────────────────
TICKERS_FILE_PATH = "tickers.csv"
DB_PATH           = "Stock Data.db"
UPDATE_GROWTH_CSV = "update_growth.csv"
CHARTS_DIR        = "charts/"
TABLE_NAME        = "ForwardFinancialData"

# ────────────────────────────────────────────────────────────────────
# Build-stamp helper (for run-wide freshness checks)
# ────────────────────────────────────────────────────────────────────
def write_build_stamp(stamp_path="charts/_build_stamp.txt") -> str:
    Path("charts").mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    Path(stamp_path).write_text(ts, encoding="utf-8")
    print(f"[build-stamp] {stamp_path} = {ts}")
    return ts

# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────
def manage_tickers(tickers_file, is_remote=False):
    tickers = ticker_manager.read_tickers(tickers_file)
    tickers = ticker_manager.modify_tickers(tickers, is_remote)
    tickers = sorted(tickers)
    ticker_manager.write_tickers(tickers, tickers_file)
    return tickers

def establish_database_connection(db_path):
    if not os.path.exists(db_path):
        print(f"[ERROR] Database not found at {db_path}")
        return None
    return sqlite3.connect(db_path)

def log_average_valuations(avg_values, tickers_file):
    if tickers_file != "tickers.csv":
        return
    req = ("Nicks_TTM_Value_Average",
           "Nicks_Forward_Value_Average",
           "Finviz_TTM_Value_Average")
    if not all(k in avg_values for k in req):
        print("[WARNING] Missing keys in avg_values; skipping DB insert.")
        return
    today = datetime.now().strftime("%Y-%m-%d")
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS AverageValuations (
            date DATE PRIMARY KEY,
            avg_ttm_valuation REAL,
            avg_forward_valuation REAL,
            avg_finviz_valuation REAL
        );
        """)
        cur.execute("SELECT 1 FROM AverageValuations WHERE date = ?", (today,))
        if not cur.fetchone():
            cur.execute("""
            INSERT INTO AverageValuations
              (date, avg_ttm_valuation, avg_forward_valuation, avg_finviz_valuation)
            VALUES (?, ?, ?, ?);
            """, (
                today,
                avg_values["Nicks_TTM_Value_Average"],
                avg_values["Nicks_Forward_Value_Average"],
                avg_values["Finviz_TTM_Value_Average"]
            ))
            conn.commit()

def balancesheet_chart(ticker):
    data = fetch_bs_for_chart(ticker)
    if data is None:
        return
    plot_chart(data, CHARTS_DIR, ticker)

    debt   = data.get("Total_Debt")
    equity = data.get("Total_Equity")
    def _is_missing(x):
        return x is None or (isinstance(x, (float,int)) and math.isnan(x)) or pd.isna(x)
    if _is_missing(debt) or _is_missing(equity) or equity == 0:
        print(f"[INFO] Skipping Debt/Equity ratio for {ticker}")
        data["Debt_to_Equity_Ratio"] = None
    else:
        data["Debt_to_Equity_Ratio"] = debt / equity
    create_and_save_table(data, CHARTS_DIR, ticker)

def fetch_and_update_balance_sheet_data(ticker, cursor):
    current = fetch_balance_sheet_data(ticker, cursor)
    if (check_missing_balance_sheet_data(ticker, cursor) or
        is_balance_sheet_data_outdated(current)):
        fresh = fetch_balance_sheet_data_from_yahoo(ticker)
        if fresh:
            store_fetched_balance_sheet_data(cursor, fresh)

def fetch_10_year_treasury_yield():
    try:
        return yf.Ticker("^TNX").info.get("regularMarketPrice")
    except Exception as e:
        print(f"[YF] Error fetching 10Y Treasury Yield: {e}")
        return None

# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────
def mini_main():
    # Build stamp first so we can verify freshness of artifacts created in this run
    write_build_stamp()

    # ─── Build the economic-indicator HTML first
    generate_economic_data()

    financial_data, dashboard_data = {}, []
    treasury = fetch_10_year_treasury_yield()

    tickers = manage_tickers(TICKERS_FILE_PATH, is_remote=True)
    conn = establish_database_connection(DB_PATH)
    if not conn:
        return

    try:
        cursor = conn.cursor()
        process_update_growth_csv(UPDATE_GROWTH_CSV, DB_PATH)

        missing_segments = []  # ← collect any tickers that didn't produce a table

        for ticker in tickers:
            print(f"[main] Processing {ticker}")
            try:
                annual_and_ttm_update(ticker, DB_PATH)
                fetch_and_update_balance_sheet_data(ticker, cursor)
                balancesheet_chart(ticker)
                scrape_forward_data(ticker)
                generate_forecast_charts_and_tables(ticker, DB_PATH, CHARTS_DIR)

                prepared, mktcap = prepare_data_for_display(ticker, treasury)
                generate_html_table(prepared, ticker)
                valuation_update(ticker, cursor, treasury, mktcap, dashboard_data)
                generate_expense_reports(ticker, rebuild_schema=False, conn=conn)
                if not segments_table.exists():
                    print(f"[segments] WARN: {segments_table} not created")
                    missing_segments.append(ticker)

                # Optional: be polite to external sources (SEC, etc.)
                time.sleep(1)
                # ───────────────────────────────────────────────────────────────────

            except Exception as e:
                # Prevent one ticker failure (e.g., yfinance timeout) from killing the run
                print(f"[WARN] Skipping remaining steps for {ticker} due to error: {e}")
                continue

        # Post-run generators
        eps_dividend_generator()
        generate_all_summaries()

        full_html, avg_vals = generate_dashboard_table(dashboard_data)
        log_average_valuations(avg_vals, TICKERS_FILE_PATH)
        spy_qqq_html = index_growth(treasury)
        generate_earnings_tables()
        render_index_growth_charts()

        # Pages (includes segment table read-in)
        backfill_index_growth()
        generate_index_growth_pages()


        html_generator2(
            tickers,
            financial_data,
            full_html,
            avg_vals,
            spy_qqq_html
        )

        # Quick visibility: which tickers are missing segment tables?
        if missing_segments:
            head = ", ".join(missing_segments[:30])
            tail = " …" if len(missing_segments) > 30 else ""
            print(f"[segments] Missing tables for {len(missing_segments)} tickers: {head}{tail}")

        # ── Generate a one-page freshness report under charts/ (optional) ──
        try:
            from freshness_check import read_stamp, scan_charts, write_report
            stamp = read_stamp()
            stale = scan_charts(stamp)
            write_report(stale)  # writes charts/freshness_report.html
        except Exception as e:
            print(f"[freshness] WARN: could not write freshness report: {e}")

    finally:
        conn.close()

# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mini_main()

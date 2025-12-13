#!/usr/bin/env python3
# main_remote.py – 2025-08-27  (segments first; canonical table path)
import sqlite3, pandas as pd, yfinance as yf, math, os, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

import ticker_manager
from generate_market_summary import generate_market_summary
from generate_economic_data    import generate_economic_data
from annual_and_ttm_update     import annual_and_ttm_update, get_db_connection
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
from backfill_index_growth     import backfill_index_growth

# We no longer call the second table generator; chart writer owns the table.
from generate_segment_charts   import generate_segment_charts_for_ticker

# Constants
TICKERS_FILE_PATH = "tickers.csv"
DB_PATH           = "Stock Data.db"
UPDATE_GROWTH_CSV = "update_growth.csv"
CHARTS_DIR        = "charts/"
TABLE_NAME        = "ForwardFinancialData"
HISTORICAL_CUTOFF = "2025-07-01"

def write_build_stamp(stamp_path=Path(CHARTS_DIR) / "_build_stamp.txt") -> str:
    Path(CHARTS_DIR).mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    Path(stamp_path).write_text(ts, encoding="utf-8")
    print(f"[build-stamp] {stamp_path} = {ts}")
    return ts

def manage_tickers(tickers_file, is_remote=False):
    tickers = ticker_manager.read_tickers(tickers_file)
    tickers = ticker_manager.modify_tickers(tickers, is_remote)
    ticker_manager.write_tickers(tickers, tickers_file)
    return tickers

def establish_database_connection(db_path):
    # Ensure schema once and return a ready connection
    return get_db_connection(db_path)

def log_average_valuations(avg_values, tickers_file):
    if tickers_file != "tickers.csv":
        return
    req = ("Nicks_TTM_Value_Average","Nicks_Forward_Value_Average","Finviz_TTM_Value_Average")
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
        return x is None or (isinstance(x, (float, int)) and math.isnan(x)) or pd.isna(x)

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


def maybe_load_sp500_index_series(db_path: str = DB_PATH) -> None:
    """Run the one-time SP500 loader script when historical data is missing."""

    try:
        with sqlite3.connect(db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT 1
                  FROM Index_PE_History
                 WHERE Ticker = ? AND PE_Type = ? AND Date < ?
                 LIMIT 1;
                """,
                ("SPY", "TTM", HISTORICAL_CUTOFF),
            )
            if cur.fetchone():
                print("[SP500 loader] Historical SPY P/E data already present; skipping loader.")
                return
    except sqlite3.Error as exc:
        print(f"[SP500 loader] Unable to inspect Index_PE_History: {exc}")
        return

    pe_csv = Path(os.environ.get("SP500_PE_CSV", "data/sp500_daily_pe_filled.csv"))
    yield_csv = Path(
        os.environ.get("TREASURY_YIELD_CSV", "data/treasury_10y_yield.csv")
    )
    missing_inputs = [str(p) for p in (pe_csv, yield_csv) if not p.exists()]
    if missing_inputs:
        print(
            "[SP500 loader] Missing required CSV files; skipping loader: "
            + ", ".join(missing_inputs)
        )
        return

    script_path = Path("scripts") / "load_sp500_index_series.py"
    if not script_path.exists():
        print(f"[SP500 loader] Loader script not found at {script_path}")
        return

    print("[SP500 loader] Running load_sp500_index_series.py for SPY")
    try:
        subprocess.run([sys.executable, str(script_path)], check=True)
    except subprocess.CalledProcessError as exc:
        print(f"[SP500 loader] Loader script failed with exit code {exc.returncode}")


# ───────────────────────────────────────────────────────────
# Segments: charts + table (canonical)
# ───────────────────────────────────────────────────────────
def build_segments_for_ticker(ticker: str) -> bool:
    out_dir = Path(CHARTS_DIR) / ticker
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Charts (+ tables emitted by generate_segment_charts_for_ticker)
    generate_segment_charts_for_ticker(ticker, out_dir)

    axis1 = out_dir / f"axis1_{ticker}_segments_table.html"
    axis2 = out_dir / f"axis2_{ticker}_segments_table.html"
    ok1 = axis1.exists() and axis1.is_file()
    ok2 = axis2.exists() and axis2.is_file()
    if not ok1 or not ok2:
        print(f"[segments] Missing segment tables for {ticker}: axis1={ok1} axis2={ok2}")
    return ok1 and ok2

# ───────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────
def mini_main():
    write_build_stamp()
    generate_economic_data()

    financial_data, dashboard_data = {}, []
    treasury = fetch_10_year_treasury_yield()
    market_summary_embed = (
        '<p>Daily market summary unavailable. '
        '<a href="daily-market-summary.html">View latest generated file</a></p>'
    )

    tickers = manage_tickers(TICKERS_FILE_PATH, is_remote=True)
    # below is the inserted code
    maybe_load_sp500_index_series(DB_PATH)
    # above is the inserted code
    conn = establish_database_connection(DB_PATH)
    if not conn:
        return

    try:
        cursor = conn.cursor()
        process_update_growth_csv(UPDATE_GROWTH_CSV, DB_PATH)

        missing_segments = []

        for ticker in tickers:
            print(f"[main] Processing {ticker}")
            try:
                # 1) Core financial data
                annual_and_ttm_update(ticker, cursor)
                scrape_forward_data(ticker)
                generate_forecast_charts_and_tables(ticker, DB_PATH, CHARTS_DIR)

                # 2) Balance sheet
                fetch_and_update_balance_sheet_data(ticker, cursor)
                balancesheet_chart(ticker)

                # 3) Segments (run within loop so no separate pass/pause)
                ok = build_segments_for_ticker(ticker)
                if not ok:
                    missing_segments.append(ticker)

                # 4) Valuation + reporting
                prepared, mktcap = prepare_data_for_display(ticker, treasury)
                generate_html_table(prepared, ticker)
                valuation_update(ticker, cursor, treasury, mktcap, dashboard_data)
                generate_expense_reports(ticker, rebuild_schema=False, conn=conn)

            except Exception as e:
                print(f"[WARN] Skipping remaining steps for {ticker} due to error: {e}")
                continue

        if missing_segments:
            msg = "Missing segment tables for: " + ", ".join(missing_segments)
            print(f"[WARN] {msg}")

        eps_dividend_generator()
        generate_all_summaries()

        full_html, avg_vals = generate_dashboard_table(dashboard_data)
        log_average_valuations(avg_vals, TICKERS_FILE_PATH)
        spy_qqq_html = index_growth(treasury)
        generate_earnings_tables()
        # Generate index growth charts for both SPY and QQQ so that
        # their growth pages share the same styled summaries.
        for idx in ("SPY", "QQQ"):
            render_index_growth_charts(idx)
        backfill_index_growth()

        try:
            generate_market_summary()
            market_summary_embed = (
                '<iframe src="daily-market-summary.html" '
                'title="Daily Market Summary" '
                'style="width:100%;height:740px;border:2px inset #C0C0C0;'
                'background:#FFFFFF" loading="lazy"></iframe>'
                '<p style="text-align:right;margin-top:6px;">'
                '<a href="daily-market-summary.html">Open full summary</a>'
                '</p>'
            )
        except Exception as exc:
            print(f"[WARN] Daily market summary generation failed: {exc}")

        html_generator2(
            tickers,
            financial_data,
            full_html,
            avg_vals,
            spy_qqq_html,
            daily_market_summary_html=market_summary_embed
        )
    finally:
        conn.close()

if __name__ == "__main__":
    mini_main()

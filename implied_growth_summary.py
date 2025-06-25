import math
import sqlite3
import yfinance as yf
from datetime import datetime

# ───────────────────────────────────────────────
# Configuration
# ───────────────────────────────────────────────
DB_PATH     = "Stock Data.db"
TABLE_NAME  = "Implied_Growth_History"

# ───────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────
def fetch_pe_ratios(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        trailing_pe = info.get("trailingPE")
        forward_pe = info.get("forwardPE")
        return trailing_pe, forward_pe
    except Exception as e:
        print(f"[fetch_pe_ratios] Failed for {ticker}: {e}")
        return None, None

def calculate_implied_growth(pe_ratio, treasury_yield):
    if pe_ratio is None or pe_ratio == 0:
        return None
    try:
        return treasury_yield * pe_ratio
    except Exception as e:
        print(f"[calculate_implied_growth] Error: {e}")
        return None

def try_insert(tkr, typ, val, date):
    # Reject invalid values (complex, None, NaN, inf)
    if not isinstance(val, (int, float)) or isinstance(val, complex) or not math.isfinite(val):
        print(f"[try_insert] Skipping invalid growth value for {tkr} ({typ}): {val}")
        return

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(f'''
            INSERT INTO {TABLE_NAME} (ticker, growth_type, growth_value, date_recorded)
            VALUES (?, ?, ?, ?)
        ''', (tkr, typ, round(val, 6), date))
        conn.commit()

def record_implied_growth_history(ticker, date_str, ttm_growth, forward_growth):
    try_insert(ticker, 'TTM', ttm_growth, date_str)
    try_insert(ticker, 'Forward', forward_growth, date_str)

# ───────────────────────────────────────────────
# Public Entrypoint (Mini-Main)
# ───────────────────────────────────────────────
def prepare_data_for_display(ticker, treasury_yield):
    """
    Main callable function. Fetches PE ratios, calculates implied growth,
    and stores them into the Implied_Growth_History table.
    Returns summary data and placeholder for market cap (if needed later).
    """
    today = datetime.today().strftime("%Y-%m-%d")

    trailing_pe, forward_pe = fetch_pe_ratios(ticker)
    ttm_growth = calculate_implied_growth(trailing_pe, treasury_yield)
    forward_growth = calculate_implied_growth(forward_pe, treasury_yield)

    record_implied_growth_history(ticker, today, ttm_growth, forward_growth)

    prepared_data = {
        "Ticker": ticker,
        "Trailing P/E": trailing_pe,
        "Forward P/E": forward_pe,
        "TTM Implied Growth": ttm_growth,
        "Forward Implied Growth": forward_growth
    }

    return prepared_data, None

# log_index_growth.py
# --------------------------------------------------------------------
# Logs implied growth rates for SPY and QQQ into Index_Growth_History
# --------------------------------------------------------------------

import sqlite3
from datetime import datetime
import yfinance as yf

DB_PATH = "Stock Data.db"
TABLE_NAME = "Index_Growth_History"
INDEXES = ["SPY", "QQQ"]

def compute_growth(ttm_pe, fwd_pe, treasury_yield=0.045):
    """ Gordon Growth Model Rearranged: Growth = ROE - Dividend Yield """
    if ttm_pe == 0 or fwd_pe == 0:
        return None, None
    ttm_growth = treasury_yield * ttm_pe - 1
    fwd_growth = treasury_yield * fwd_pe - 1
    return ttm_growth, fwd_growth

def fetch_pe_ratios(ticker):
    """ Get trailing and forward P/E ratios using yfinance """
    data = yf.Ticker(ticker).info
    return data.get("trailingPE"), data.get("forwardPE")

def ensure_table_exists(conn):
    """ Create table if it doesn't exist """
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            Date TEXT,
            Ticker TEXT,
            Growth_Type TEXT, -- TTM or Forward
            Implied_Growth REAL,
            PRIMARY KEY (Date, Ticker, Growth_Type)
        )
    """)
    conn.commit()

def log_index_growth():
    today = datetime.today().strftime("%Y-%m-%d")
    conn = sqlite3.connect(DB_PATH)
    ensure_table_exists(conn)
    cursor = conn.cursor()

    for ticker in INDEXES:
        ttm_pe, fwd_pe = fetch_pe_ratios(ticker)
        ttm_growth, fwd_growth = compute_growth(ttm_pe, fwd_pe)

        if ttm_growth is not None:
            cursor.execute(f"""
                INSERT OR REPLACE INTO {TABLE_NAME}
                (Date, Ticker, Growth_Type, Implied_Growth)
                VALUES (?, ?, ?, ?)
            """, (today, ticker, "TTM", ttm_growth))

        if fwd_growth is not None:
            cursor.execute(f"""
                INSERT OR REPLACE INTO {TABLE_NAME}
                (Date, Ticker, Growth_Type, Implied_Growth)
                VALUES (?, ?, ?, ?)
            """, (today, ticker, "Forward", fwd_growth))

        print(f"Logged growth for {ticker} on {today}")

    conn.commit()
    conn.close()

# Mini-main
if __name__ == "__main__":
    log_index_growth()

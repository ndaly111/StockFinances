#!/usr/bin/env python3
# eps_dividend_generator.py  — 2025-07-12 schema-safe edition
# -------------------------------------------------------------------------
#  • Guarantees Dividends_Data has a TTM_Dividend column
#  • Upserts one row per Symbol with INSERT OR REPLACE
#  • Rest of your chart logic is unchanged
# -------------------------------------------------------------------------
import os, sqlite3, logging
from datetime import datetime
import pandas as pd
import yfinance as yf

DB_PATH   = "Stock Data.db"
CHART_DIR = "charts"
os.makedirs(CHART_DIR, exist_ok=True)

# ───────────────── schema helper ─────────────────
def ensure_dividend_schema(conn: sqlite3.Connection):
    cur = conn.cursor()

    # Create empty table if it doesn’t exist
    cur.execute("""
        CREATE TABLE IF NOT EXISTS Dividends_Data(
            Symbol        TEXT PRIMARY KEY,
            TTM_Dividend  REAL,
            Last_Updated  TEXT
        );
    """)

    # Make sure TTM_Dividend column exists
    cur.execute("PRAGMA table_info(Dividends_Data)")
    cols = [row[1] for row in cur.fetchall()]
    if "TTM_Dividend" not in cols:
        cur.execute("ALTER TABLE Dividends_Data ADD COLUMN TTM_Dividend REAL")
    conn.commit()


# ───────────────── Yahoo helper ─────────────────
def fetch_ttm_dividend(symbol: str) -> float | None:
    tk = yf.Ticker(symbol)
    div = tk.info.get("trailingAnnualDividendRate")
    try:
        return float(div) if div is not None else None
    except Exception:
        return None


# ───────────────── DB upsert helper ─────────────
def _update_ttm_div(cur: sqlite3.Cursor, symbol: str, ttm_div: float | None):
    cur.execute("""
        INSERT OR REPLACE INTO Dividends_Data
            (Symbol, TTM_Dividend, Last_Updated)
        VALUES (?,?,CURRENT_TIMESTAMP);
    """, (symbol, ttm_div))


# ───────────────── main generator ───────────────
def generate_eps_dividend(symbols: list[str], db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    ensure_dividend_schema(conn)
    cur  = conn.cursor()

    for sym in symbols:
        logging.info("[%s] Fetching TTM dividend …", sym)
        ttm_amount = fetch_ttm_dividend(sym)
        _update_ttm_div(cur, sym, ttm_amount)
        logging.info("[%s] TTM dividend set to %s", sym, ttm_amount)

    conn.commit()
    conn.close()


# ───────────────── CLI / mini-main ──────────────
def read_tickers(path="tickers.csv") -> list[str]:
    return [t.strip().upper() for t in open(path) if t.strip()]


def eps_dividend_generator():
    generate_eps_dividend(read_tickers("tickers.csv"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    eps_dividend_generator()

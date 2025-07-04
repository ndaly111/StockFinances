# log_index_growth.py
# --------------------------------------------------------------------
# Logs implied growth rates for SPY and QQQ into Index_Growth_History
# --------------------------------------------------------------------

import sqlite3
from datetime import datetime
import yfinance as yf

DB_PATH      = "Stock Data.db"
TABLE_NAME   = "Index_Growth_History"
INDEXES      = ["SPY", "QQQ"]
TREASURY_YLD = 0.045            # 10-yr yield used in Gordon model

# ───────────────────────────────────────────────────────────
#  Public helper functions (names unchanged from your original)
# ───────────────────────────────────────────────────────────
def compute_growth(ttm_pe: float, fwd_pe: float):
    """Implied growth via simple Gordon Growth rearrangement."""
    if not ttm_pe or not fwd_pe:
        return None, None
    return TREASURY_YLD * ttm_pe - 1, TREASURY_YLD * fwd_pe - 1

def fetch_pe_ratios(ticker: str):
    """Return (trailingPE, forwardPE) from yfinance."""
    info = yf.Ticker(ticker).info
    return info.get("trailingPE"), info.get("forwardPE")

def ensure_table_exists(conn: sqlite3.Connection):
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            Date           TEXT,
            Ticker         TEXT,
            Growth_Type    TEXT,   -- 'TTM' | 'Forward'
            Implied_Growth REAL,
            PRIMARY KEY (Date, Ticker, Growth_Type)
        )
    """)
    conn.commit()

# ───────────────────────────────────────────────────────────
#  Main routine
# ───────────────────────────────────────────────────────────
def log_index_growth():
    """Store today’s implied growth for SPY & QQQ."""
    today = datetime.today().strftime("%Y-%m-%d")
    conn  = sqlite3.connect(DB_PATH)
    ensure_table_exists(conn)
    cur   = conn.cursor()

    for tk in INDEXES:
        ttm_pe, fwd_pe         = fetch_pe_ratios(tk)
        ttm_growth, fwd_growth = compute_growth(ttm_pe, fwd_pe)

        if ttm_growth is not None:
            cur.execute(f"""
                INSERT OR REPLACE INTO {TABLE_NAME}
                (Date, Ticker, Growth_Type, Implied_Growth)
                VALUES (?, ?, 'TTM', ?)
            """, (today, tk, ttm_growth))

        if fwd_growth is not None:
            cur.execute(f"""
                INSERT OR REPLACE INTO {TABLE_NAME}
                (Date, Ticker, Growth_Type, Implied_Growth)
                VALUES (?, ?, 'Forward', ?)
            """, (today, tk, fwd_growth))

        print(f"Logged {tk} — TTM: {ttm_growth:.2%} | Fwd: {fwd_growth:.2%}")

    conn.commit()
    conn.close()

# Mini-main for manual test runs
if __name__ == "__main__":
    log_index_growth()

#!/usr/bin/env python3
# annual_and_ttm_update.py  — 2025-07-12 “single-row TTM” (failsafe edition)
# ---------------------------------------------------------------------------
#  • Ensures exactly ONE TTM row per Symbol
#  • If an old composite-PK table exists it is renamed, not dropped
#  • TTM row is written with INSERT OR REPLACE (robust to any PK/UNIQUE flags)
# ---------------------------------------------------------------------------
import logging, os, sqlite3, time
from datetime import datetime
from functools import lru_cache

import numpy as np
import pandas as pd
import yfinance as yf

DB_PATH   = "Stock Data.db"
CHART_DIR = "charts"
os.makedirs(CHART_DIR, exist_ok=True)

# ─────────────────────────── DB helpers ────────────────────────────
def _rename_legacy_ttm(cur: sqlite3.Cursor):
    """Detect legacy (composite-PK) TTM_Data and rename it."""
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='TTM_Data'")
    if not cur.fetchone():
        return  # no table → nothing to rename

    cur.execute("PRAGMA table_info(TTM_Data)")
    pk_cols = [row for row in cur.fetchall() if row[5] > 0]  # col[5] == pk flag
    if len(pk_cols) == 1 and pk_cols[0][1] == "Symbol":
        return  # already new schema

    ts = int(time.time())
    cur.execute(f"ALTER TABLE TTM_Data RENAME TO TTM_Data_OLD_{ts}")
    logging.info("Legacy TTM_Data table renamed to TTM_Data_OLD_%s", ts)


def get_db_connection(db_path: str) -> sqlite3.Connection:
    """Open the DB and ensure schemas + UNIQUE constraint are correct."""
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()

    # ---- make sure legacy table (if any) is renamed first ----
    _rename_legacy_ttm(cur)

    # Annual_Data (never changed)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS Annual_Data(
            Symbol TEXT,
            Date   TEXT,
            Revenue      REAL,
            Net_Income   REAL,
            EPS          REAL,
            Last_Updated TEXT,
            PRIMARY KEY(Symbol, Date)
        );
    """)

    # Clean one-row TTM table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS TTM_Data(
            Symbol TEXT PRIMARY KEY,            -- ← the only PK
            TTM_Revenue        REAL,
            TTM_Net_Income     REAL,
            TTM_EPS            REAL,
            Shares_Outstanding REAL,
            Quarter            TEXT,
            Last_Updated       TEXT
        );
    """)
    # Extra safeguard: UNIQUE(Symbol) even if an odd PK state remains
    cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_ttm_symbol_unique ON TTM_Data(Symbol)")

    cur.execute("CREATE INDEX IF NOT EXISTS idx_annual_symbol ON Annual_Data(Symbol)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_ttm_symbol    ON TTM_Data(Symbol)")
    conn.commit()
    return conn

# ───────────────────────── utilities ─────────────────────────
def _to_float(v):
    try:
        return float(v)
    except Exception:
        return np.nan

def _clean_financial_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ("Revenue", "Net_Income", "EPS"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    df.dropna(how="all", subset=["Revenue", "Net_Income", "EPS"], inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df

# ─────────────────── fetch from Yahoo Finance ─────────────────
@lru_cache(maxsize=32)
def _fetch_annual(tkr: str) -> pd.DataFrame:
    logging.info("Fetching ANNUAL data from Yahoo Finance")
    fin = yf.Ticker(tkr).financials
    if fin is None or fin.empty:
        return pd.DataFrame()
    fin = fin.T
    fin["Date"] = fin.index
    fin.rename(columns={"Total Revenue": "Revenue",
                        "Net Income":    "Net_Income",
                        "Basic EPS":     "EPS"}, inplace=True)
    return _clean_financial_df(fin)

@lru_cache(maxsize=32)
def _fetch_ttm(tkr: str) -> dict | None:
    logging.info("Fetching TTM data from Yahoo Finance")
    tk = yf.Ticker(tkr)
    q = tk.quarterly_financials
    if q is None or q.empty:
        return None
    return {
        "TTM_Revenue":        q.loc["Total Revenue"].iloc[:4].sum(),
        "TTM_Net_Income":     q.loc["Net Income"].iloc[:4].sum(),
        "TTM_EPS":            tk.info.get("trailingEps", np.nan),
        "Shares_Outstanding": tk.info.get("sharesOutstanding", np.nan),
        "Quarter":            q.columns[0].strftime("%Y-%m-%d"),
    }

# ──────────────────── storage helpers ────────────────────────
def _store_annual(tkr: str, df: pd.DataFrame, cur: sqlite3.Cursor):
    for _, row in df.iterrows():
        d = row["Date"].strftime("%Y-%m-%d") if hasattr(row["Date"], "strftime") else row["Date"]
        cur.execute("""
            INSERT OR REPLACE INTO Annual_Data
                (Symbol, Date, Revenue, Net_Income, EPS, Last_Updated)
            VALUES (?,?,?,?,?,CURRENT_TIMESTAMP);
        """, (tkr, d,
              _to_float(row["Revenue"]),
              _to_float(row["Net_Income"]),
              _to_float(row["EPS"])))

def _store_ttm(tkr: str, d: dict, cur: sqlite3.Cursor):
    cur.execute("""
        INSERT OR REPLACE INTO TTM_Data
            (Symbol, TTM_Revenue, TTM_Net_Income, TTM_EPS,
             Shares_Outstanding, Quarter, Last_Updated)
        VALUES (?,?,?,?,?,?,CURRENT_TIMESTAMP);
    """, (
        tkr,
        _to_float(d.get("TTM_Revenue")),
        _to_float(d.get("TTM_Net_Income")),
        _to_float(d.get("TTM_EPS")),
        _to_float(d.get("Shares_Outstanding")),
        d.get("Quarter"),
    ))

# ───────────────────────── main entry ─────────────────────────
def annual_and_ttm_update(tkr: str, cur: sqlite3.Cursor):
    """Update annual and TTM data for ``tkr`` using an existing cursor."""

    # Annual (only if none exist)
    cur.execute("SELECT 1 FROM Annual_Data WHERE Symbol=? LIMIT 1", (tkr,))
    if not cur.fetchone():
        df = _fetch_annual(tkr)
        if not df.empty:
            _store_annual(tkr, df, cur)

    # TTM (overwrite with latest every run)
    ttm = _fetch_ttm(tkr)
    if ttm:
        _store_ttm(tkr, ttm, cur)

    # Commit changes but leave connection management to caller
    cur.connection.commit()
    logging.info("[%s] annual+TTM update complete", tkr)

# stand-alone sanity test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    conn = get_db_connection(DB_PATH)
    cur = conn.cursor()
    annual_and_ttm_update("AAPL", cur)
    conn.close()

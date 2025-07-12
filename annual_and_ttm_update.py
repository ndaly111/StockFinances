#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────
#  annual_and_ttm_update.py   —  2025-07-11 single-row TTM patch
#     • TTM_Data uses Symbol as the sole PRIMARY KEY
#     • On first run it drops the old composite-PK table if present
#     • Subsequent runs just upsert the one latest TTM row
# ─────────────────────────────────────────────────────────────
import logging
import os
import sqlite3
from datetime import datetime
from functools import lru_cache

import numpy as np
import pandas as pd
import yfinance as yf

DB_PATH   = "Stock Data.db"
CHART_DIR = "charts"
os.makedirs(CHART_DIR, exist_ok=True)

# ───────────────────────── DB helpers ─────────────────────────
def get_db_connection(db_path: str) -> sqlite3.Connection:
    """
    Ensure the tables exist with correct schemas.
    If TTM_Data still has the old composite PK, rebuild it with Symbol-only PK.
    """
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()

    # ---------- Annual_Data (unchanged) ----------
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

    # ---------- TTM_Data  (migrate if needed) ----------
    def _is_new_schema():
        cur.execute("PRAGMA table_info(TTM_Data)")
        cols = cur.fetchall()
        if not cols:                 # table doesn’t exist yet
            return False
        pk_cols = [c for c in cols if c[5] == 1]
        return len(pk_cols) == 1 and pk_cols[0][1] == "Symbol"

    if not _is_new_schema():
        cur.execute("DROP TABLE IF EXISTS TTM_Data")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS TTM_Data(
            Symbol TEXT PRIMARY KEY,            -- single-column PK
            TTM_Revenue        REAL,
            TTM_Net_Income     REAL,
            TTM_EPS            REAL,
            Shares_Outstanding REAL,
            Quarter            TEXT,
            Last_Updated       TEXT
        );
    """)

    # Helpful indexes for reads
    cur.execute("CREATE INDEX IF NOT EXISTS idx_annual_symbol ON Annual_Data(Symbol)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_ttm_symbol   ON TTM_Data(Symbol)")

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
def _fetch_annual_from_yahoo(tkr: str) -> pd.DataFrame:
    logging.info("Fetching ANNUAL data from Yahoo Finance")
    fin = yf.Ticker(tkr).financials
    if fin is None or fin.empty:
        return pd.DataFrame()
    fin = fin.T
    fin["Date"] = fin.index
    fin.rename(columns={
        "Total Revenue": "Revenue",
        "Net Income":    "Net_Income",
        "Basic EPS":     "EPS",
    }, inplace=True)
    return _clean_financial_df(fin)

@lru_cache(maxsize=32)
def _fetch_ttm_from_yahoo(tkr: str) -> dict | None:
    logging.info("Fetching TTM data from Yahoo Finance")
    ticker = yf.Ticker(tkr)
    q = ticker.quarterly_financials
    if q is None or q.empty:
        return None
    return {
        "TTM_Revenue":        q.loc["Total Revenue"].iloc[:4].sum(),
        "TTM_Net_Income":     q.loc["Net Income"].iloc[:4].sum(),
        "TTM_EPS":            ticker.info.get("trailingEps", np.nan),
        "Shares_Outstanding": ticker.info.get("sharesOutstanding", np.nan),
        "Quarter":            q.columns[0].strftime("%Y-%m-%d"),
    }

# ───────────────────── storage helpers ───────────────────────
def _store_annual(tkr: str, df: pd.DataFrame, cur: sqlite3.Cursor):
    for _, row in df.iterrows():
        d = row["Date"].strftime("%Y-%m-%d") if hasattr(row["Date"], "strftime") else row["Date"]
        cur.execute("""
            INSERT INTO Annual_Data
                (Symbol, Date, Revenue, Net_Income, EPS, Last_Updated)
            VALUES (?,?,?,?,?,CURRENT_TIMESTAMP)
            ON CONFLICT(Symbol,Date) DO UPDATE SET
                Revenue      = COALESCE(EXCLUDED.Revenue,      Revenue),
                Net_Income   = COALESCE(EXCLUDED.Net_Income,   Net_Income),
                EPS          = COALESCE(EXCLUDED.EPS,          EPS),
                Last_Updated = CURRENT_TIMESTAMP
        """, (tkr, d,
              _to_float(row["Revenue"]),
              _to_float(row["Net_Income"]),
              _to_float(row["EPS"])))

def _store_ttm(tkr: str, dct: dict, cur: sqlite3.Cursor):
    """
    Overwrite the single TTM row for <tkr>.
    """
    cur.execute("""
        INSERT INTO TTM_Data
            (Symbol, TTM_Revenue, TTM_Net_Income, TTM_EPS,
             Shares_Outstanding, Quarter, Last_Updated)
        VALUES (?,?,?,?,?,?,CURRENT_TIMESTAMP)
        ON CONFLICT(Symbol) DO UPDATE SET
            TTM_Revenue        = COALESCE(EXCLUDED.TTM_Revenue,        TTM_Revenue),
            TTM_Net_Income     = COALESCE(EXCLUDED.TTM_Net_Income,     TTM_Net_Income),
            TTM_EPS            = COALESCE(EXCLUDED.TTM_EPS,            TTM_EPS),
            Shares_Outstanding = COALESCE(EXCLUDED.Shares_Outstanding, Shares_Outstanding),
            Quarter            = COALESCE(EXCLUDED.Quarter,            Quarter),
            Last_Updated       = CURRENT_TIMESTAMP
    """, (
        tkr,
        _to_float(dct.get("TTM_Revenue")),
        _to_float(dct.get("TTM_Net_Income")),
        _to_float(dct.get("TTM_EPS")),
        _to_float(dct.get("Shares_Outstanding")),
        dct.get("Quarter"),
    ))

# ─────────────────────── main updater ────────────────────────
def annual_and_ttm_update(tkr: str, db_path: str = DB_PATH):
    conn = get_db_connection(db_path)
    cur  = conn.cursor()

    # ---- Annual (only fetch if empty) ------------------------
    cur.execute("SELECT 1 FROM Annual_Data WHERE Symbol=? LIMIT 1", (tkr,))
    if not cur.fetchone():
        df = _fetch_annual_from_yahoo(tkr)
        if not df.empty:
            _store_annual(tkr, df, cur)

    # ---- TTM (always save latest, overwriting prior row) -----
    ttm_dict = _fetch_ttm_from_yahoo(tkr)
    if ttm_dict:                             # None if Yahoo failed
        _store_ttm(tkr, ttm_dict, cur)

    conn.commit()
    conn.close()
    logging.info("[%s] annual + TTM update complete", tkr)

# stand-alone sanity test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    annual_and_ttm_update("AAPL")

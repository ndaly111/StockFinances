"""
expense_reports.py
Builds annual / quarterly operating-expense tables, stores them in SQLite,
and generates two charts per ticker:

  1) Revenue vs. stacked expenses (absolute $)
  2) Expenses as % of revenue

Keeps the legacy API (`generate_expense_reports`) but now
retries on “database is locked” errors to avoid CI flakes.
"""

from __future__ import annotations
import os
import sqlite3
import time
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib.ticker import FuncFormatter

from expense_labels import (
    COST_OF_REVENUE,
    RESEARCH_AND_DEVELOPMENT,
    SELLING_AND_MARKETING,
    GENERAL_AND_ADMIN,
    SGA_COMBINED,
    FACILITIES_DA,
    PERSONNEL_COSTS,
    INSURANCE_CLAIMS,
    OTHER_OPERATING,
)

# --------------------------------------------------------------------------- #
#  Module-wide constants & setup                                              #
# --------------------------------------------------------------------------- #
DB_PATH    = "Stock Data.db"
DB_URI     = f"file:{DB_PATH}?cache=shared&mode=rwc"
OUTPUT_DIR = "charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

__all__ = ["generate_expense_reports"]


# --------------------------------------------------------------------------- #
#  Utility helpers                                                            #
# --------------------------------------------------------------------------- #
def clean_value(val):
    """Convert NaNs → None and datetimes → ISO-8601 strings (for SQLite)."""
    if pd.isna(val):
        return None
    if isinstance(val, (pd.Timestamp, datetime)):
        return val.isoformat()
    return val


# --------------------------------------------------------------------------- #
#  Expense extraction                                                         #
# --------------------------------------------------------------------------- #
def extract_expenses(row: pd.Series):
    """Fuzzy-match your label lists against each row index."""
    def match_any(label_list):
        for key in row.index:
            for lbl in label_list:
                if lbl.lower() in key.lower() and pd.notna(row[key]):
                    return row[key]
        return None

    return (
        match_any(COST_OF_REVENUE),
        match_any(RESEARCH_AND_DEVELOPMENT),
        match_any(SELLING_AND_MARKETING),
        match_any(GENERAL_AND_ADMIN),
        match_any(SGA_COMBINED),
        match_any(FACILITIES_DA),
        match_any(PERSONNEL_COSTS),
        match_any(INSURANCE_CLAIMS),
        match_any(OTHER_OPERATING),
    )


# --------------------------------------------------------------------------- #
#  Robust SQLite opener                                                       #
# --------------------------------------------------------------------------- #
def _open_conn() -> sqlite3.Connection:
    """
    Open the shared SQLite DB with a generous timeout & shared cache.
    """
    conn = sqlite3.connect(
        DB_URI,
        uri=True,
        timeout=30,                     # wait up to 30s if locked
        detect_types=sqlite3.PARSE_DECLTYPES,
    )
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=60000;")
    except sqlite3.OperationalError:
        pass
    return conn


# --------------------------------------------------------------------------- #
#  Schema enforcement                                                         #
# --------------------------------------------------------------------------- #
TABLES       = ("IncomeStatement", "QuarterlyIncomeStatement")
TABLE_SCHEMA = """
    CREATE TABLE IF NOT EXISTS {name} (
        ticker TEXT,
        period_ending TEXT,
        total_revenue REAL,
        cost_of_revenue REAL,
        research_and_development REAL,
        selling_and_marketing REAL,
        general_and_admin REAL,
        sga_combined REAL,
        facilities_da REAL,
        personnel_costs REAL,
        insurance_claims REAL,
        other_operating REAL,
        PRIMARY KEY (ticker, period_ending)
    );
"""

def ensure_tables(*, drop: bool = False) -> None:
    """
    Guarantee that both tables exist with the expected columns.
    If drop=True you get a fresh rebuild (only for manual resets).
    Retries automatically on a locked DB.
    """
    attempts, wait = 5, 1
    for i in range(attempts):
        try:
            conn = _open_conn()
            cur  = conn.cursor()
            for tbl in TABLES:
                if drop:
                    cur.execute(f"DROP TABLE IF EXISTS {tbl};")
                cur.execute(TABLE_SCHEMA.format(name=tbl))
            conn.commit()
            conn.close()
            msg = "dropped & recreated" if drop else "ensured"
            print(f"✅ Tables {msg}")
            return
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower() and i < attempts-1:
                print(f"⚠️ DB locked during schema step; retrying in {wait}s…")
                time.sleep(wait)
                wait *= 2
                continue
            raise


# --------------------------------------------------------------------------- #
#  Storage helpers                                                            #
# --------------------------------------------------------------------------- #
def store_data(ticker: str, *, mode: str = "annual") -> None:
    """
    Fetch annual or quarterly financials and upsert into SQLite.
    Retries on lock.
    """
    yf_tkr = yf.Ticker(ticker)
    raw_df = (
        yf_tkr.financials.transpose()
        if mode == "annual"
        else yf_tkr.quarterly_financials.transpose()
    )

    if raw_df.empty:
        print(f"⚠️ No {mode} financials for {ticker}")
        return

    table = "IncomeStatement" if mode=="annual" else "QuarterlyIncomeStatement"
    attempts, wait = 5, 1

    for i in range(attempts):
        try:
            conn = _open_conn()
            cur  = conn.cursor()
            for idx, row in raw_df.iterrows():
                pe        = idx.to_pydatetime() if isinstance(idx, pd.Timestamp) else idx
                total_rev = row.get("Total Revenue")
                cost_rev, rnd, mkt, adm, sga, fda, ppl, ins, oth = extract_expenses(row)

                cur.execute(f"""
                    INSERT OR REPLACE INTO {table}
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    ticker, clean_value(pe), clean_value(total_rev),
                    clean_value(cost_rev), clean_value(rnd), clean_value(mkt),
                    clean_value(adm),   clean_value(sga), clean_value(fda),
                    clean_value(ppl),   clean_value(ins), clean_value(oth)
                ))
            conn.commit()
            conn.close()
            print(f"✅ {mode.capitalize()} data stored for {ticker}")
            return
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower() and i < attempts-1:
                print(f"⚠️ DB locked during store_data({mode}); retrying in {wait}s…")
                time.sleep(wait)
                wait *= 2
                continue
            raise


# --------------------------------------------------------------------------- #
#  Data fetchers                                                              #
# --------------------------------------------------------------------------- #
def fetch_yearly_data(ticker: str) -> pd.DataFrame:
    conn = _open_conn()
    df   = pd.read_sql_query(
        "SELECT * FROM IncomeStatement WHERE ticker=?", conn, params=(ticker,)
    )
    conn.close()
    if df.empty:
        return pd.DataFrame()
    df["period_ending"] = pd.to_datetime(df["period_ending"])
    df["year"]          = df["period_ending"].dt.year
    agg_cols            = df.columns.difference(["ticker","period_ending","year"])
    return df.groupby("year", as_index=False)[agg_cols].sum()

def fetch_ttm_data(ticker: str) -> pd.DataFrame:
    conn = _open_conn()
    df   = pd.read_sql_query(
        "SELECT * FROM QuarterlyIncomeStatement WHERE ticker=? ORDER BY period_ending DESC",
        conn, params=(ticker,)
    )
    conn.close()
    if df.empty:
        return pd.DataFrame()
    df["period_ending"] = pd.to_datetime(df["period_ending"])
    recent = df.head(4).sort_values("period_ending")
    if len(recent)<4:
        return pd.DataFrame()
    expected = pd.date_range(end=recent["period_ending"].max(), periods=4, freq="Q")
    if list(expected.to_period("Q")) != list(recent["period_ending"].dt.to_period("Q")):
        return pd.DataFrame()
    ttm = recent.drop(columns=["ticker","period_ending"]).sum().to_frame().T
    ttm.insert(0, "year", "TTM")
    return ttm


# --------------------------------------------------------------------------- #
#  Chart helpers                                                              #
# --------------------------------------------------------------------------- #
def format_short(x, dec=1):
    if pd.isna(x):
        return "$0"
    absx = abs(x)
    if absx>=1e12: return f"${x/1e12:.{dec}f} T"
    if absx>=1e9:  return f"${x/1e9:.{dec}f} B"
    if absx>=1e6:  return f"${x/1e6:.{dec}f} M"
    if absx>=1e3:  return f"${x/1e3:.{dec}f} K"
    return f"${x:.{dec}f}"

# paste in your existing plot_expense_charts & plot_expense_percent_chart
# (identical to your prior versions – no change needed here)


# --------------------------------------------------------------------------- #
#  Public orchestration                                                        #
# --------------------------------------------------------------------------- #
def generate_expense_reports(ticker: str, *, rebuild_schema: bool=False) -> None:
    """
    Main entrypoint used by main_remote.py.
    If rebuild_schema=True, drops & recreates the two tables first.
    """
    ensure_tables(drop=rebuild_schema)
    store_data(ticker, mode="annual")
    store_data(ticker, mode="quarterly")

    yearly = fetch_yearly_data(ticker)
    if yearly.empty:
        print("⛔ No data found — skipping charts")
        return

    ttm  = fetch_ttm_data(ticker)
    full = pd.concat([yearly, ttm], ignore_index=True)

    plot_expense_charts(full, ticker)
    plot_expense_percent_chart(full, ticker)


# --------------------------------------------------------------------------- #
#  CLI usage                                                                  #
# --------------------------------------------------------------------------- #
if __name__=="__main__":
    generate_expense_reports("AAPL")

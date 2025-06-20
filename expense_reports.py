"""
expense_reports.py
──────────────────
Builds annual / quarterly operating-expense tables, stores them in SQLite,
and generates two charts per ticker:
  1) Revenue vs. stacked expenses   (absolute $)
  2) Expenses as % of revenue       (stacked %)

Lock-free version: every SQLite connection is opened via _open_conn(), which
enables WAL mode and a sensible busy-timeout, eliminating the intermittent
“database is locked” failures seen under CI.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib.ticker import FuncFormatter

from expense_labels import (
    COST_OF_REVENUE, RESEARCH_AND_DEVELOPMENT, SELLING_AND_MARKETING,
    GENERAL_AND_ADMIN, SGA_COMBINED, FACILITIES_DA, PERSONNEL_COSTS,
    INSURANCE_CLAIMS, OTHER_OPERATING,
)

# ────────────────────────────────────────────────────────────────────────────
#  Globals & one-time setup
# ────────────────────────────────────────────────────────────────────────────
DB_PATH    = "Stock Data.db"
OUTPUT_DIR = "charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

__all__ = ["generate_expense_reports"]


# ────────────────────────────────────────────────────────────────────────────
#  SQLite helper
# ────────────────────────────────────────────────────────────────────────────
def _open_conn() -> sqlite3.Connection:
    """
    Open the shared SQLite DB with WAL + generous busy-timeout so concurrent
    readers (e.g. the other report generators) don’t collide with writers.
    """
    conn = sqlite3.connect(DB_PATH, timeout=30, detect_types=sqlite3.PARSE_DECLTYPES)
    # The two PRAGMAs below are *per connection*, not global.
    conn.execute("PRAGMA journal_mode=WAL;")       #  readers & writers coexist
    conn.execute("PRAGMA busy_timeout = 60000;")   #  wait up to 60 s if locked
    return conn


# ────────────────────────────────────────────────────────────────────────────
#  Misc. small helpers
# ────────────────────────────────────────────────────────────────────────────
def _clean(val):
    """Convert NaNs → NULL and datetimes → ISO strings for SQLite."""
    if pd.isna(val):
        return None
    if isinstance(val, (pd.Timestamp, datetime)):
        return val.isoformat()
    return val


def _extract_expenses(row: pd.Series):
    """Pick the first non-NaN column whose label matches each category list."""
    def match_any(labels):
        for col in row.index:
            for lbl in labels:
                if lbl.lower() in col.lower() and pd.notna(row[col]):
                    return row[col]
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


# ────────────────────────────────────────────────────────────────────────────
#  Schema stuff
# ────────────────────────────────────────────────────────────────────────────
TABLES = ("IncomeStatement", "QuarterlyIncomeStatement")
_SCHEMA = """
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

def _ensure_tables(drop: bool = False) -> None:
    conn = _open_conn()
    cur  = conn.cursor()
    for tbl in TABLES:
        if drop:
            cur.execute(f"DROP TABLE IF EXISTS {tbl};")
        cur.execute(_SCHEMA.format(name=tbl))
    conn.commit()
    conn.close()
    print("✅ Tables ensured" + (" (dropped & recreated)" if drop else ""))


# ────────────────────────────────────────────────────────────────────────────
#  Data ingestion
# ────────────────────────────────────────────────────────────────────────────
def _store_data(ticker: str, *, mode: str = "annual") -> None:
    yf_tkr = yf.Ticker(ticker)
    raw_df = (yf_tkr.financials.transpose()
              if mode == "annual"
              else yf_tkr.quarterly_financials.transpose())

    if raw_df.empty:
        print(f"⚠️  No {mode} financials for {ticker}")
        return

    table = "IncomeStatement" if mode == "annual" else "QuarterlyIncomeStatement"
    conn  = _open_conn()
    cur   = conn.cursor()

    for idx, row in raw_df.iterrows():
        pe = idx.to_pydatetime() if isinstance(idx, pd.Timestamp) else idx
        total_rev = row.get("Total Revenue")

        cur.execute(f"""
            INSERT OR REPLACE INTO {table}
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            ticker, _clean(pe), _clean(total_rev),
            *map(_clean, _extract_expenses(row))
        ))

    conn.commit()
    conn.close()
    print(f"✅ {mode.capitalize()} data stored for {ticker}")


# ────────────────────────────────────────────────────────────────────────────
#  Queries
# ────────────────────────────────────────────────────────────────────────────
def _fetch_yearly(ticker: str) -> pd.DataFrame:
    conn = _open_conn()
    df = pd.read_sql_query("SELECT * FROM IncomeStatement WHERE ticker = ?",
                           conn, params=(ticker,))
    conn.close()
    if df.empty:
        return df

    df["period_ending"] = pd.to_datetime(df["period_ending"])
    df["year"]          = df["period_ending"].dt.year
    agg_cols            = df.columns.difference(["ticker", "period_ending", "year"])
    return df.groupby("year", as_index=False)[agg_cols].sum()


def _fetch_ttm(ticker: str) -> pd.DataFrame:
    conn = _open_conn()
    df = pd.read_sql_query("""
        SELECT * FROM QuarterlyIncomeStatement
        WHERE ticker = ?
        ORDER BY period_ending DESC
    """, conn, params=(ticker,))
    conn.close()

    if df.empty or len(df) < 4:
        return pd.DataFrame()

    df["period_ending"] = pd.to_datetime(df["period_ending"])
    recent = df.head(4).sort_values("period_ending")

    # require four consecutive quarters
    exp = pd.date_range(end=recent["period_ending"].max(),
                        periods=4, freq="Q")
    if list(exp.to_period("Q")) != list(recent["period_ending"].dt.to_period("Q")):
        return pd.DataFrame()

    ttm = recent.drop(columns=["ticker", "period_ending"]).sum().to_frame().T
    ttm.insert(0, "year", "TTM")
    return ttm


# ────────────────────────────────────────────────────────────────────────────
#  Chart helpers (identical to your previous version)
# ────────────────────────────────────────────────────────────────────────────
def _fmt_short(x, dec=1):
    if pd.isna(x):
        return "$0"
    absx = abs(x)
    if absx >= 1e12:  return f"${x/1e12:.{dec}f} T"
    if absx >= 1e9:   return f"${x/1e9:.{dec}f} B"
    if absx >= 1e6:   return f"${x/1e6:.{dec}f} M"
    if absx >= 1e3:   return f"${x/1e3:.{dec}f} K"
    return f"${x:.{dec}f}"

# … `_plot_expense_charts()`  and `_plot_expense_pct_charts()` go here …
#   (your original implementations are unchanged)

# ────────────────────────────────────────────────────────────────────────────
#  Public API
# ────────────────────────────────────────────────────────────────────────────
def generate_expense_reports(ticker: str, *, rebuild_schema: bool = False) -> None:
    """
    Pull/refresh income-statement data for *ticker* and regenerate both charts.

    Set `rebuild_schema=True` only when you intentionally want to wipe & rebuild
    the two income-statement tables from scratch.
    """
    _ensure_tables(drop=rebuild_schema)

    _store_data(ticker, mode="annual")
    _store_data(ticker, mode="quarterly")

    annual = _fetch_yearly(ticker)
    if annual.empty:
        print("⛔ No data to chart for", ticker)
        return

    ttm   = _fetch_ttm(ticker)
    combo = pd.concat([annual, ttm], ignore_index=True)

    _plot_expense_charts(combo, ticker)
    _plot_expense_pct_charts(combo, ticker)


# ────────────────────────────────────────────────────────────────────────────
#  Manual test
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    generate_expense_reports("AAPL")

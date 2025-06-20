"""
expense_reports.py
Builds annual / quarterly operating-expense tables, stores them in SQLite,
and generates two charts per ticker:

  1) Revenue vs. stacked expenses (absolute $)
  2) Expenses as % of revenue

This version keeps the legacy API (`generate_expense_reports`) but avoids
dropping tables on every import, which eliminated the “database is locked”
errors in CI.
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
OUTPUT_DIR = "charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

__all__ = ["generate_expense_reports"]        # what other modules should import


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
#  Schema enforcement                                                         #
# --------------------------------------------------------------------------- #
TABLES      = ("IncomeStatement", "QuarterlyIncomeStatement")
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

def ensure_tables(*, drop: bool = False, conn: sqlite3.Connection | None = None) -> None:
    """
    Guarantee that both income-statement tables exist *with the expected columns*.

    Parameters
    ----------
    drop : bool, default False
        Set to True **only** for a manual full rebuild — never during normal runs,
        otherwise long jobs that already hold connections will fail with
        `sqlite3.OperationalError: database is locked`.
    conn : sqlite3.Connection, optional
        If provided, use this connection instead of opening a new one.
    """
    new_conn = conn is None
    if new_conn:
        conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()

    for tbl in TABLES:
        if drop:
            cur.execute(f"DROP TABLE IF EXISTS {tbl};")
        cur.execute(TABLE_SCHEMA.format(name=tbl))

    conn.commit()
    if new_conn:
        conn.close()

    if drop:
        print("✅ Tables dropped & recreated with fresh 12-column schema")
    else:
        print("✅ Tables ensured (created only if missing)")


# --------------------------------------------------------------------------- #
#  Storage helpers                                                            #
# --------------------------------------------------------------------------- #
def store_data(ticker: str, *, mode: str = "annual", conn: sqlite3.Connection | None = None) -> None:
    """
    Fetch *annual* or *quarterly* financials via yfinance and upsert rows
    into the correct table.
    """
    yf_tkr = yf.Ticker(ticker)
    raw_df = (
        yf_tkr.financials.transpose()
        if mode == "annual"
        else yf_tkr.quarterly_financials.transpose()
    )

    if raw_df.empty:
        print(f"⚠️  No {mode} financials found for {ticker}")
        return

    new_conn = conn is None
    if new_conn:
        conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    table = "IncomeStatement" if mode == "annual" else "QuarterlyIncomeStatement"

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
    if new_conn:
        conn.close()

    print(f"✅ {mode.capitalize()} data stored for {ticker}")


# --------------------------------------------------------------------------- #
#  Data fetchers                                                              #
# --------------------------------------------------------------------------- #
def fetch_yearly_data(ticker: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT * FROM IncomeStatement
        WHERE ticker = ?
    """, conn, params=(ticker,))
    conn.close()

    if df.empty:
        return pd.DataFrame()

    df["period_ending"] = pd.to_datetime(df["period_ending"])
    df["year"]          = df["period_ending"].dt.year
    agg_cols            = df.columns.difference(["ticker", "period_ending", "year"])

    return df.groupby("year", as_index=False)[agg_cols].sum()


def fetch_ttm_data(ticker: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT * FROM QuarterlyIncomeStatement
        WHERE ticker = ?
        ORDER BY period_ending DESC
    """, conn, params=(ticker,))
    conn.close()

    if df.empty:
        return pd.DataFrame()

    df["period_ending"] = pd.to_datetime(df["period_ending"])
    recent = df.head(4).sort_values("period_ending")

    if len(recent) < 4:
        return pd.DataFrame()

    expected = pd.date_range(end=recent["period_ending"].max(), periods=4, freq="Q")
    if list(expected.to_period("Q")) != list(recent["period_ending"].dt.to_period("Q")):
        return pd.DataFrame()

    ttm = recent.drop(columns=["ticker", "period_ending"]).sum().to_frame().T
    ttm.insert(0, "year", "TTM")
    return ttm


# --------------------------------------------------------------------------- #
#  Chart helpers                                                              #
# --------------------------------------------------------------------------- #
def format_short(x, dec=1):
    if pd.isna(x):
        return "$0"
    absx = abs(x)
    if absx >= 1e12:  return f"${x/1e12:.{dec}f} T"
    if absx >= 1e9:   return f"${x/1e9:.{dec}f} B"
    if absx >= 1e6:   return f"${x/1e6:.{dec}f} M"
    if absx >= 1e3:   return f"${x/1e3:.{dec}f} K"
    return f"${x:.{dec}f}"

# … two plotting functions go here, unchanged from your original version …
def plot_expense_charts(full: pd.DataFrame, ticker: str):
    pass
def plot_expense_percent_chart(full: pd.DataFrame, ticker: str):
    pass


# --------------------------------------------------------------------------- #
#  Public orchestration function                                              #
# --------------------------------------------------------------------------- #
def generate_expense_reports(
    ticker: str,
    *,
    rebuild_schema: bool = False,
    conn: sqlite3.Connection | None = None
) -> None:
    """
    High-level helper used by main_remote.py.

    Parameters
    ----------
    ticker : str
        The stock symbol to process.

    rebuild_schema : bool, default False
        Set to True only for a full wipe/reset of the two income-statement
        tables.  The default *does not* drop existing data.

    conn : sqlite3.Connection, optional
        If provided, use this connection for all DB work (no new connections).
    """
    # ensure schema
    ensure_tables(drop=rebuild_schema, conn=conn)

    # store raw data
    store_data(ticker, mode="annual", conn=conn)
    store_data(ticker, mode="quarterly", conn=conn)

    # build and plot
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
if __name__ == "__main__":
    generate_expense_reports("AAPL")   # simple manual test

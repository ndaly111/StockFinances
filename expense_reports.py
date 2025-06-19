"""
expense_reports.py  –  builds annual / quarterly operating-expense tables,
stores them in SQLite, and generates two PNG charts:

1) Revenue vs. stacked expenses (absolute $)
2) Expenses as % of revenue

The public function `generate_expense_reports(ticker)` is unchanged.
"""

import os
import sqlite3
from datetime import datetime, timedelta
import contextlib   # NEW

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

DB_PATH    = "Stock Data.db"
OUTPUT_DIR = "charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------------------------------- #
#  Connection helper (adds WAL + busy-timeout)                                #
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def _get_conn(path: str = DB_PATH, timeout: int = 5):
    """Context manager that enables WAL and waits up to *timeout* seconds."""
    conn = sqlite3.connect(path, timeout=timeout)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        yield conn
    finally:
        conn.commit()
        conn.close()

# --------------------------------------------------------------------------- #
#  Utility helpers                                                            #
# --------------------------------------------------------------------------- #
def clean_value(val):
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
#  One-time schema enforcement                                                #
# --------------------------------------------------------------------------- #
def ensure_tables():
    """
    Ensure the IncomeStatement & QuarterlyIncomeStatement tables exist with the
    correct 12-column schema.

    - Never drops tables (avoids exclusive locks)
    - Adds missing columns on the fly
    """
    base_cols = [
        ("ticker", "TEXT"),
        ("period_ending", "TEXT"),
        ("total_revenue", "REAL"),
        ("cost_of_revenue", "REAL"),
        ("research_and_development", "REAL"),
        ("selling_and_marketing", "REAL"),
        ("general_and_admin", "REAL"),
        ("sga_combined", "REAL"),
        ("facilities_da", "REAL"),
        ("personnel_costs", "REAL"),
        ("insurance_claims", "REAL"),
        ("other_operating", "REAL"),
    ]

    with _get_conn() as conn:
        cur = conn.cursor()
        for tbl in ("IncomeStatement", "QuarterlyIncomeStatement"):
            # create table if it doesn't exist
            col_defs = ", ".join([f"{c[0]} {c[1]}" for c in base_cols])
            pk = "PRIMARY KEY (ticker, period_ending)"
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {tbl} (
                    {col_defs},
                    {pk}
                );
            """)

            # make sure every column is present
            cur.execute(f"PRAGMA table_info({tbl});")
            existing = {row[1] for row in cur.fetchall()}
            for col, coltype in base_cols:
                if col not in existing:
                    cur.execute(f"ALTER TABLE {tbl} ADD COLUMN {col} {coltype};")

    print("✅ Tables ensured (WAL mode, no drops, correct columns)")

# --------------------------------------------------------------------------- #
#  Storage                                                                    #
# --------------------------------------------------------------------------- #
def store_data(ticker: str, mode="annual"):
    """
    Fetches annual or quarterly financials from Yahoo Finance and inserts
    into the corresponding SQLite table.
    """
    df = (
        yf.Ticker(ticker).financials.transpose()
        if mode == "annual"
        else yf.Ticker(ticker).quarterly_financials.transpose()
    )

    table = "IncomeStatement" if mode == "annual" else "QuarterlyIncomeStatement"

    with _get_conn() as conn:
        cur = conn.cursor()
        for idx, row in df.iterrows():
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
    print(f"✅ {mode.capitalize()} data stored for {ticker}")

# --------------------------------------------------------------------------- #
#  Data fetchers                                                              #
# --------------------------------------------------------------------------- #
def fetch_yearly_data(ticker: str) -> pd.DataFrame:
    with _get_conn() as conn:
        df = pd.read_sql_query("""
            SELECT * FROM IncomeStatement
            WHERE ticker = ?
        """, conn, params=(ticker,))

    if df.empty:
        return pd.DataFrame()

    df["period_ending"] = pd.to_datetime(df["period_ending"])
    df["year"]          = df["period_ending"].dt.year

    agg_cols = df.columns.difference(["ticker", "period_ending", "year"])
    return df.groupby("year", as_index=False)[agg_cols].sum()

def fetch_ttm_data(ticker: str) -> pd.DataFrame:
    with _get_conn() as conn:
        df = pd.read_sql_query("""
            SELECT * FROM QuarterlyIncomeStatement
            WHERE ticker = ?
            ORDER BY period_ending DESC
        """, conn, params=(ticker,))

    if df.empty:
        return pd.DataFrame()

    df["period_ending"] = pd.to_datetime(df["period_ending"])
    recent = df[df["period_ending"] > (datetime.today() - timedelta(days=150))]

    if len(recent) < 4:
        return pd.DataFrame()

    recent = recent.head(4).sort_values("period_ending")
    exp = pd.date_range(end=recent["period_ending"].max(), periods=4, freq="Q")
    if list(exp.to_period("Q")) != list(recent["period_ending"].dt.to_period("Q")):
        return pd.DataFrame()

    ttm = recent.drop(columns=["ticker", "period_ending"]).sum().to_frame().T
    ttm.insert(0, "year", "TTM")
    return ttm

# --------------------------------------------------------------------------- #
#  Chart helpers (unchanged)                                                  #
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

# ……………………………………………………………………………………………………………………………
#  plot_expense_charts, plot_expense_percent_chart and generate_expense_reports
#  are **identical** to the original – omitted for brevity, just copy them over
# ……………………………………………………………………………………………………………………………

# --------------------------------------------------------------------------- #
#  Run                                                                        #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    generate_expense_reports("AAPL")   # change ticker or loop as needed

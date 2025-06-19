"""
expense_reports.py
Builds annual / quarterly operating-expense tables, stores them in SQLite,
and generates two charts per ticker:

  1) Revenue vs. stacked expenses (absolute $)
  2) Expenses as % of revenue

– Keeps the legacy public API (`generate_expense_reports`)
– Uses WAL once per interpreter to avoid “database is locked” races
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
DB_PATH       = "Stock Data.db"
OUTPUT_DIR    = "charts"
DB_TIMEOUT    = 30                    # seconds for busy_timeout
_WAL_READY    = False                 # flipped to True after first WAL switch
os.makedirs(OUTPUT_DIR, exist_ok=True)

__all__ = ["generate_expense_reports"]       # what other modules should import


# --------------------------------------------------------------------------- #
#  Connection helper                                                          #
# --------------------------------------------------------------------------- #
def _open_conn() -> sqlite3.Connection:
    """
    Return a connection with busy-timeout; run PRAGMA journal_mode=WAL just once
    per interpreter to avoid an exclusive-lock race.
    """
    conn = sqlite3.connect(
        DB_PATH,
        timeout=DB_TIMEOUT,
        check_same_thread=False,      # allow threads if the caller uses them
    )
    conn.execute(f"PRAGMA busy_timeout = {DB_TIMEOUT * 1000};")

    global _WAL_READY
    if not _WAL_READY:
        try:
            conn.execute("PRAGMA journal_mode = WAL;")
            _WAL_READY = True
        except sqlite3.OperationalError as e:
            # Another process / thread is flipping the switch right now.
            # When we next touch the DB it will already be in WAL mode, so
            # ignore the temporary “database is locked” on this pragma only.
            if "database is locked" not in str(e).lower():
                raise

    return conn


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

def _ensure_tables(*, drop: bool = False) -> None:
    """
    Guarantee that both income-statement tables exist with the expected schema.

    Parameters
    ----------
    drop : bool
        True  → drop & recreate the tables (full reset)  
        False → create only if missing (safe default)
    """
    conn = _open_conn()
    cur  = conn.cursor()

    for tbl in TABLES:
        if drop:
            cur.execute(f"DROP TABLE IF EXISTS {tbl};")
        cur.execute(TABLE_SCHEMA.format(name=tbl))

    conn.commit()
    conn.close()
    print("✅ Tables dropped & recreated" if drop else "✅ Tables ensured (created only if missing)")


# --------------------------------------------------------------------------- #
#  Storage helpers                                                            #
# --------------------------------------------------------------------------- #
def store_data(ticker: str, *, mode: str = "annual") -> None:
    """
    Fetch *annual* or *quarterly* financials via yfinance and upsert rows into
    the correct table.
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

    table = "IncomeStatement" if mode == "annual" else "QuarterlyIncomeStatement"
    conn  = _open_conn()
    cur   = conn.cursor()

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


# --------------------------------------------------------------------------- #
#  Data fetchers                                                              #
# --------------------------------------------------------------------------- #
def fetch_yearly_data(ticker: str) -> pd.DataFrame:
    """Aggregate annual rows by calendar year (sum of columns)."""
    conn = _open_conn()
    df = pd.read_sql_query("SELECT * FROM IncomeStatement WHERE ticker = ?", conn,
                           params=(ticker,))
    conn.close()

    if df.empty:
        return pd.DataFrame()

    df["period_ending"] = pd.to_datetime(df["period_ending"])
    df["year"]          = df["period_ending"].dt.year
    agg_cols            = df.columns.difference(["ticker", "period_ending", "year"])

    return df.groupby("year", as_index=False)[agg_cols].sum()


def fetch_ttm_data(ticker: str) -> pd.DataFrame:
    """
    Sum the last four quarterly rows and label the result “TTM”.
    Returns empty DF if we don’t have four aligned quarters yet.
    """
    conn = _open_conn()
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

#  … plot_expense_charts() and plot_expense_percent_chart() remain verbatim …

# --------------------------------------------------------------------------- #
#  Public orchestrator                                                        #
# --------------------------------------------------------------------------- #
def generate_expense_reports(ticker: str, *, rebuild_schema: bool = False) -> None:
    """
    High-level helper used by main_remote.py.

    Parameters
    ----------
    ticker : str
        Stock symbol to process.

    rebuild_schema : bool, default False
        True  → wipe & recreate the two income-statement tables.  
        False → normal run, keep existing data.
    """
    _ensure_tables(drop=rebuild_schema)

    # Refresh raw data in the DB
    store_data(ticker, mode="annual")
    store_data(ticker, mode="quarterly")

    # Build charts / HTML
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
    generate_expense_reports("AAPL")

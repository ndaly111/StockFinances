"""
expense_reports.py
-------------------------------------------------------------------------------
Builds annual / quarterly operating-expense tables, stores them in SQLite,
and generates two charts per ticker:

  1) Revenue vs. stacked expenses (absolute $)
  2) Expenses as % of revenue
"""

from __future__ import annotations

import math
import os
import sqlite3
from datetime import datetime

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

def ensure_tables(*, drop: bool = False, conn: sqlite3.Connection | None = None) -> None:
    new_conn = conn is None
    if new_conn:
        conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    for tbl in TABLES:
        if drop:
            cur.execute(f"DROP TABLE IF EXISTS {tbl};")
        cur.execute(TABLE_SCHEMA.format(name=tbl))

    conn.commit()
    if new_conn:
        conn.close()


# --------------------------------------------------------------------------- #
#  Storage helpers                                                            #
# --------------------------------------------------------------------------- #
def store_data(ticker: str, *, mode: str = "annual", conn: sqlite3.Connection | None = None) -> None:
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

    # Convert to datetime & extract integer year
    df["period_ending"] = pd.to_datetime(df["period_ending"])
    df["year_int"]      = df["period_ending"].dt.year

    # Sum up all expense columns per year_int
    agg_cols = df.columns.difference(["ticker", "period_ending", "year_int"])
    grouped  = df.groupby("year_int", as_index=False)[agg_cols].sum()

    # Create a display label (string) for each year
    grouped["year_label"] = grouped["year_int"].astype(int).astype(str)
    return grouped


def fetch_ttm_data(ticker: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql_query("""
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

    # Sum trailing-12-months
    ttm = recent.drop(columns=["ticker", "period_ending"]).sum().to_frame().T

    # Insert string label + blank numeric key
    ttm.insert(0, "year_label", "TTM")
    ttm["year_int"] = np.nan
    return ttm


# --------------------------------------------------------------------------- #
#  Chart helpers                                                              #
# --------------------------------------------------------------------------- #
def _format_short(x, _pos=None, dec=1):
    if pd.isna(x):
        return "$0"
    absx = abs(x)
    if absx >= 1e12:  return f"${x/1e12:.{dec}f} T"
    if absx >= 1e9:   return f"${x/1e9:.{dec}f} B"
    if absx >= 1e6:   return f"${x/1e6:.{dec}f} M"
    if absx >= 1e3:   return f"${x/1e3:.{dec}f} K"
    return f"${x:.{dec}f}"


# --------------------------------------------------------------------------- #
#  1) Revenue vs. stacked expenses (absolute $)                               #
# --------------------------------------------------------------------------- #
def plot_expense_charts(full: pd.DataFrame, ticker: str) -> None:
    full = full.copy()
    # Sort by the pure-int key (NaNs for TTM end up at bottom)
    full.sort_values("year_int", inplace=True)
    x_labels = full["year_label"].tolist()

    use_combined = full["sga_combined"].notna().any()
    categories   = [
        ("Cost of Revenue",     "cost_of_revenue",          "#6d6d6d"),
        ("R&D",                 "research_and_development", "blue"),
        ("G&A",                 "general_and_admin",        "#ffb3c6"),
        ("Selling & Marketing", "selling_and_marketing",    "#ffc6e2"),
        ("SG&A",                "sga_combined",             "#c2a5ff"),
        ("Facilities / D&A",    "facilities_da",            "orange"),
    ]
    if use_combined:
        categories = [c for c in categories
                      if c[1] not in ("general_and_admin","selling_and_marketing")]

    fig, ax = plt.subplots(figsize=(11, 6))
    bottoms = np.zeros(len(full))
    for label, col, color in categories:
        vals = full[col].fillna(0).values
        ax.bar(x_labels, vals, bottom=bottoms, label=label, color=color, width=0.6)
        bottoms += vals

    ax.plot(x_labels, full["total_revenue"].values,
            color="black", linewidth=2, marker="o", label="Revenue")

    max_val = max(bottoms.max(), full["total_revenue"].max())
    ax.set_ylim(0, max_val * 1.10)

    ax.set_title(f"Revenue vs. Operating Expenses — {ticker}")
    ax.yaxis.set_major_formatter(FuncFormatter(_format_short))
    ax.set_ylabel("USD")
    ax.legend(frameon=False, ncol=2)
    plt.tight_layout()

    fig.savefig(os.path.join(OUTPUT_DIR, f"{ticker}_expenses_vs_revenue.png"))
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  2) Expenses as % of revenue                                                #
# --------------------------------------------------------------------------- #
def plot_expense_percent_chart(full: pd.DataFrame, ticker: str) -> None:
    full = full.copy()
    full.sort_values("year_int", inplace=True)
    x_labels = full["year_label"].tolist()

    use_combined = full["sga_combined"].notna().any()
    categories   = [
        ("Cost of Revenue",     "cost_of_revenue",          "#6d6d6d"),
        ("R&D",                 "research_and_development", "blue"),
        ("G&A",                 "general_and_admin",        "#ffb3c6"),
        ("Selling & Marketing", "selling_and_marketing",    "#ffc6e2"),
        ("SG&A",                "sga_combined",             "#c2a5ff"),
        ("Facilities / D&A",    "facilities_da",            "orange"),
    ]
    if use_combined:
        categories = [c for c in categories
                      if c[1] not in ("general_and_admin","selling_and_marketing")]

    # Calculate percentages
    for _lbl, col, _c in categories:
        full[col + "_pct"] = (full[col] / full["total_revenue"]) * 100

    fig, ax = plt.subplots(figsize=(11, 6))
    bottoms = np.zeros(len(full))
    for label, col, color in categories:
        pct_vals = full[col + "_pct"].fillna(0).values
        ax.bar(x_labels, pct_vals, bottom=bottoms, label=label, color=color, width=0.6)
        bottoms += pct_vals
        for x, y0, val in zip(x_labels, bottoms - pct_vals, pct_vals):
            if val > 4:
                ax.text(x, y0 + val / 2, f"{val:.1f} %",
                        ha="center", va="center", fontsize=8, color="white")

    max_total = bottoms.max()
    ylim_max  = max(110, math.ceil(max_total / 10) * 10 + 10)
    ax.set_ylim(0, ylim_max)
    ax.axhline(100, linestyle="--", linewidth=1, color="black",
               label="100 % of revenue", zorder=5)

    ax.set_ylabel("Percent of Revenue")
    ax.set_title(f"Expenses as % of Revenue — {ticker}")
    ax.set_yticks(np.arange(0, ylim_max + 1, 10))
    ax.legend(frameon=False, ncol=2)
    plt.tight_layout()

    fig.savefig(os.path.join(OUTPUT_DIR, f"{ticker}_expenses_pct_of_rev.png"))
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  Public orchestration function                                              #
# --------------------------------------------------------------------------- #
def generate_expense_reports(
    ticker: str,
    *,
    rebuild_schema: bool = False,
    conn: sqlite3.Connection | None = None
) -> None:
    ensure_tables(drop=rebuild_schema, conn=conn)

    store_data(ticker, mode="annual",    conn=conn)
    store_data(ticker, mode="quarterly", conn=conn)

    yearly = fetch_yearly_data(ticker)
    if yearly.empty:
        print(f"⛔ No data found for {ticker} — skipping charts")
        return

    full = pd.concat([yearly, fetch_ttm_data(ticker)], ignore_index=True)

    plot_expense_charts(full, ticker)
    plot_expense_percent_chart(full, ticker)


if __name__ == "__main__":
    generate_expense_reports("AAPL")

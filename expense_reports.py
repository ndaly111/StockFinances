"""
expense_reports.py
Builds annual / quarterly operating-expense tables for each ticker, stores them
in SQLite, and generates two PNG charts:

1) Revenue vs. stacked expenses (absolute $)
2) Expenses as % of revenue (stacked)

- Uses authoritative category lists from expense_labels.py
- No overlap between groups
- Excludes "Operating Expense"
"""

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

DB_PATH    = "Stock Data.db"
OUTPUT_DIR = "charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    """Drop & recreate both tables with the correct 12-column schema."""
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()

    schema = """
        CREATE TABLE {name} (
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

    for tbl in ("IncomeStatement", "QuarterlyIncomeStatement"):
        cur.execute(f"DROP TABLE IF EXISTS {tbl};")
        cur.execute(schema.format(name=tbl))

    conn.commit()
    conn.close()
    print("✅ Tables (re)created with correct 12-column schema")

# --------------------------------------------------------------------------- #
#  Storage                                                                    #
# --------------------------------------------------------------------------- #
def store_data(ticker: str, mode="annual"):
    """
    Fetches annual or quarterly financials from Yahoo Finance and inserts
    into the corresponding SQLite table.
    """
    df = (yf.Ticker(ticker).financials.transpose()
          if mode == "annual"
          else yf.Ticker(ticker).quarterly_financials.transpose())

    table = "IncomeStatement" if mode == "annual" else "QuarterlyIncomeStatement"
    conn  = sqlite3.connect(DB_PATH)
    cur   = conn.cursor()

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

    conn.commit()
    conn.close()
    print(f"✅ {mode.capitalize()} data stored for {ticker}")

# --------------------------------------------------------------------------- #
#  Data fetchers                                                              #
# --------------------------------------------------------------------------- #
def fetch_yearly_data(ticker: str) -> pd.DataFrame:
    """
    Retrieve and aggregate annual records from IncomeStatement by year.
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT * FROM IncomeStatement
        WHERE ticker = ?
    """, conn, params=(ticker,))
    conn.close()

    if df.empty:
        return pd.DataFrame()

    df["period_ending"] = pd.to_datetime(df["period_ending"])
    df["year"]         = df["period_ending"].dt.year

    agg_cols = df.columns.difference(["ticker", "period_ending", "year"])
    return df.groupby("year", as_index=False)[agg_cols].sum()

def fetch_ttm_data(ticker: str) -> pd.DataFrame:
    """
    Retrieve the most recent four quarters from QuarterlyIncomeStatement and sum them.
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT * FROM QuarterlyIncomeStatement
        WHERE ticker = ?
        ORDER BY period_ending DESC
    """, conn, params=(ticker,))
    conn.close()

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

def plot_expense_charts(df: pd.DataFrame, ticker: str):
    """
    Create a stacked bar chart of expenses vs. revenue.

    - Drops any year where all numeric columns are zero.
    - Skips any category whose total across all years is zero.
    """
    # Fill NaN with 0, then drop rows (years) where all numeric columns are zero
    numeric_cols = df.columns.difference(["year"])
    df_plot      = df.copy()
    df_plot[numeric_cols] = df_plot[numeric_cols].fillna(0)

    # Remove years where every numeric column is zero
    df_plot = df_plot.loc[~(df_plot[numeric_cols] == 0).all(axis=1)]

    if df_plot.empty:
        print("⛔ Nothing to plot after zero-row filter — skipping chart")
        return

    years = df_plot["year"].astype(str)
    pos   = np.arange(len(df_plot))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bottom = np.zeros(len(df_plot))

    categories = [
        ("cost_of_revenue",         "Cost of Revenue",       "dimgray"),
        ("research_and_development","R&D",                   "blue"),
        ("selling_and_marketing",   "Sales & Marketing",     "purple"),
        ("general_and_admin",       "G&A",                   "pink"),
        ("sga_combined",            "SG&A",                  "mediumpurple"),
        ("facilities_da",           "Facilities / D&A",      "orange"),
        ("personnel_costs",         "Personnel",             "brown"),
        ("insurance_claims",        "Insurance",             "teal"),
        ("other_operating",         "Other Op.",             "gold"),
    ]

    # For each category, draw a stack if its total is > 0
    for col, lbl, color in categories:
        if col in df_plot.columns and df_plot[col].sum() > 0:
            vals = df_plot[col].fillna(0).to_numpy()
            ax.bar(pos - width/2, vals, width, bottom=bottom, label=lbl, color=color)
            bottom += vals

    # Draw revenue bars beside the stacks
    revs = df_plot["total_revenue"].to_numpy()
    bars = ax.bar(pos + width/2, revs, width, label="Revenue", color="green")
    for b in bars:
        ax.text(
            b.get_x() + b.get_width()/2,
            b.get_height(),
            format_short(b.get_height(), 0),
            ha="center", va="bottom", fontsize=8, weight="bold"
        )

    ax.set_xticks(pos)
    ax.set_xticklabels(years)
    ax.set_ylabel("Amount")
    ax.set_title(f"Revenue vs Expenses — {ticker}")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: format_short(x, 0)))
    plt.tight_layout()

    outfile = os.path.join(OUTPUT_DIR, f"{ticker}_rev_expense_chart.png")
    plt.savefig(outfile, dpi=100)
    plt.close()
    print(f"✅ Saved chart → {outfile}")

# --------------------------------------------------------------------------- #
#  Main orchestrator                                                          #
# --------------------------------------------------------------------------- #
def generate_expense_reports(ticker: str):
    ensure_tables()                     # enforce correct schema
    store_data(ticker, mode="annual")
    store_data(ticker, mode="quarterly")

    yearly = fetch_yearly_data(ticker)
    if yearly.empty:
        print("⛔ No data found — skipping charts")
        return

    ttm  = fetch_ttm_data(ticker)
    full = pd.concat([yearly, ttm], ignore_index=True)
    plot_expense_charts(full, ticker)

# --------------------------------------------------------------------------- #
#  Run                                                                        #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    generate_expense_reports("AAPL")      # change ticker or loop as needed

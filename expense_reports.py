"""
expense_reports.py
Builds annual / quarterly operating-expense tables for each ticker, stores them
in SQLite, and generates two PNG charts:

1) Revenue vs. stacked expenses (absolute $)
2) Expenses as % of revenue (stacked)

Fixes in this version
---------------------
• SG&A detection now recognises all Yahoo label variants ( “…& Administrative”
  **and** “…And Administrative”, with or without commas).
• Percent-of-revenue math is fully divide-by-zero-safe.
• .fillna(0) cast to float to suppress future down-cast warning.
"""

import os
import sqlite3
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib.ticker import FuncFormatter

# --------------------------------------------------------------------------- #
#  Config                                                                     #
# --------------------------------------------------------------------------- #
DB_PATH = "Stock Data.db"
OUTPUT_DIR = "charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------------------------------- #
#  Helpers                                                                    #
# --------------------------------------------------------------------------- #
def clean_value(val):
    if pd.isna(val):
        return None
    if isinstance(val, (pd.Timestamp, datetime)):
        return val.isoformat()
    return val


def extract_expenses(row: pd.Series):
    """
    Return   cost_rev, rnd, mkt, adm, sga_comb
    where    mkt+adm  XOR  sga_comb  is populated.
    """

    def first(cols):
        for col in cols:
            for cand in row.index:
                if col.lower() in cand.lower() and pd.notna(row[cand]):
                    return row[cand]
        return None

    # -- base fields ---------------------------------------------------------
    cost_rev = first(["Cost Of Revenue", "Reconciled Cost Of Revenue"])
    rnd = first(["Research & Development", "Research and Development", "R&D"])

    # -- split  --------------------------------------------------------------
    mkt = first(["Selling and Marketing", "Sales and Marketing"])
    adm = first(["General and Administrative"])

    # -- combined SG&A -------------------------------------------------------
    sga_comb = first([
        "Selling General & Administrative",
        "Selling, General & Administrative",
        "Sales, General & Administrative",
        "Selling General And Administration",
        "Selling General and Administration",
        "Selling, General and Administration",
        "Sales, General and Administration",
    ])

    if sga_comb is not None:
        mkt = adm = None
        fmt = "SG&A"
    elif mkt is not None or adm is not None:
        sga_comb = None
        fmt = "Split"
    else:
        fmt = "Unknown"

    print(f"📊 Reporting format detected: {fmt}")
    return cost_rev, rnd, mkt, adm, sga_comb


# --------------------------------------------------------------------------- #
#  SQLite storage                                                             #
# --------------------------------------------------------------------------- #
def store_annual_data(ticker: str):
    print(f"\n--- Fetching ANNUAL financials for {ticker} ---")
    df = yf.Ticker(ticker).financials.transpose()
    print("First two rows:\n", df.head(2))

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS IncomeStatement (
            ticker TEXT,
            period_ending TEXT,
            total_revenue REAL,
            cost_of_revenue REAL,
            research_and_development REAL,
            selling_and_marketing REAL,
            general_and_admin REAL,
            sga_combined REAL,
            PRIMARY KEY (ticker, period_ending)
        );
    """)
    for idx, row in df.iterrows():
        pe = idx.to_pydatetime() if isinstance(idx, pd.Timestamp) else idx
        tot_rev = row.get("Total Revenue")
        cost, rnd, mkt, adm, sga = extract_expenses(row)
        cur.execute("""
            INSERT OR REPLACE INTO IncomeStatement
            VALUES (?,?,?,?,?,?,?,?)
        """, (
            ticker, clean_value(pe), clean_value(tot_rev),
            clean_value(cost), clean_value(rnd),
            clean_value(mkt), clean_value(adm), clean_value(sga)
        ))
    conn.commit()
    conn.close()
    print("✅ Annual data stored.")


def store_quarterly_data(ticker: str):
    print(f"\n--- Fetching QUARTERLY financials for {ticker} ---")
    df = yf.Ticker(ticker).quarterly_financials.transpose()

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS QuarterlyIncomeStatement (
            ticker TEXT,
            period_ending TEXT,
            total_revenue REAL,
            cost_of_revenue REAL,
            research_and_development REAL,
            selling_and_marketing REAL,
            general_and_admin REAL,
            sga_combined REAL,
            PRIMARY KEY (ticker, period_ending)
        );
    """)
    for idx, row in df.iterrows():
        pe = idx.to_pydatetime() if isinstance(idx, pd.Timestamp) else idx
        tot_rev = row.get("Total Revenue")
        cost, rnd, mkt, adm, sga = extract_expenses(row)
        cur.execute("""
            INSERT OR REPLACE INTO QuarterlyIncomeStatement
            VALUES (?,?,?,?,?,?,?,?)
        """, (
            ticker, clean_value(pe), clean_value(tot_rev),
            clean_value(cost), clean_value(rnd),
            clean_value(mkt), clean_value(adm), clean_value(sga)
        ))
    conn.commit()
    conn.close()
    print("✅ Quarterly data stored.")


# --------------------------------------------------------------------------- #
#  Fetch / aggregate helpers                                                  #
# --------------------------------------------------------------------------- #
def fetch_ttm_data(ticker: str) -> pd.DataFrame:
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
    # verify consecutive quarters
    exp = pd.date_range(end=recent["period_ending"].max(), periods=4, freq="Q")
    if list(exp.to_period("Q")) != list(recent["period_ending"].dt.to_period("Q")):
        return pd.DataFrame()

    ttm = recent.drop(columns=["ticker", "period_ending"]).sum().to_frame().T
    ttm.insert(0, "year", "TTM")
    return ttm


def load_yearly_data(ticker: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT period_ending, total_revenue, cost_of_revenue,
               research_and_development, selling_and_marketing,
               general_and_admin, sga_combined
        FROM IncomeStatement
        WHERE ticker = ?
    """, conn, params=(ticker,))
    conn.close()

    if df.empty:
        return pd.DataFrame()

    df["period_ending"] = pd.to_datetime(df["period_ending"])
    df["year"] = df["period_ending"].dt.year
    num_cols = ["total_revenue", "cost_of_revenue",
                "research_and_development", "selling_and_marketing",
                "general_and_admin", "sga_combined"]
    return df.groupby("year", as_index=False)[num_cols].sum()


# --------------------------------------------------------------------------- #
#  Chart helpers                                                              #
# --------------------------------------------------------------------------- #
def format_short(x, dec=1):
    if pd.isna(x):
        return "$0"
    absx = abs(x)
    if absx >= 1e12:
        return f"${x/1e12:.{dec}f} T"
    if absx >= 1e9:
        return f"${x/1e9:.{dec}f} B"
    if absx >= 1e6:
        return f"${x/1e6:.{dec}f} M"
    if absx >= 1e3:
        return f"${x/1e3:.{dec}f} K"
    return f"${x:.{dec}f}"


def plot_chart(df: pd.DataFrame, ticker: str):
    yrs = df["year"].astype(str)
    n = len(df)
    pos = np.arange(n)
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    # stacks
    bottom = np.zeros(n)
    def stack(vals, lbl, col):
        if np.sum(vals) == 0:
            return bottom
        bars = ax.bar(pos - width/2, vals, width,
                      bottom=bottom, label=lbl, color=col)
        for i, b in enumerate(bars):
            if b.get_height() > 0:
                ax.text(b.get_x()+b.get_width()/2, bottom[i]+b.get_height()/2,
                        format_short(b.get_height(), 0),
                        ha="center", va="center", color="white", fontsize=7)
        return bottom + vals

    bottom = stack(df["cost_of_revenue"], "Cost of Revenue", "dimgray")
    bottom = stack(df["research_and_development"], "R&D", "blue")

    if df["selling_and_marketing"].fillna(0).sum() or df["general_and_admin"].fillna(0).sum():
        bottom = stack(df["selling_and_marketing"].fillna(0), "Sales and Marketing", "mediumpurple")
        bottom = stack(df["general_and_admin"].fillna(0), "General and Administrative", "pink")
    else:
        bottom = stack(df["sga_combined"].fillna(0), "SG&A", "mediumpurple")

    # revenue bars
    bars = ax.bar(pos + width/2, df["total_revenue"], width,
                  label="Revenue", color="green")
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height(),
                format_short(b.get_height(), 0),
                ha="center", va="bottom", fontsize=8, weight="bold")

    ax.set_xticks(pos)
    ax.set_xticklabels(yrs)
    ax.set_ylabel("Amount")
    ax.set_title(f"Revenue vs Expenses — {ticker}")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: format_short(x, 0)))
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"{ticker}_rev_expense_chart.png")
    plt.savefig(out, dpi=100)
    plt.close()
    print(f"✅ Saved chart → {out}")


def plot_expense_percent_chart(df: pd.DataFrame, ticker: str):
    df_pct = df.copy()
    use_split = (df_pct["selling_and_marketing"].fillna(0).sum() > 0 or
                 df_pct["general_and_admin"].fillna(0).sum() > 0)

    if use_split:
        cols = ["cost_of_revenue", "research_and_development",
                "selling_and_marketing", "general_and_admin"]
        lbl = ["Cost of Revenue", "R&D", "Sales and Marketing", "General and Administrative"]
        colr = ["dimgray", "blue", "mediumpurple", "pink"]
    else:
        cols = ["cost_of_revenue", "research_and_development", "sga_combined"]
        lbl = ["Cost of Revenue", "R&D", "SG&A"]
        colr = ["dimgray", "blue", "mediumpurple"]

    # safe percentage calc
    tr = df_pct["total_revenue"].replace(0, np.nan)
    df_pct[cols] = df_pct[cols].fillna(0).div(tr, axis=0).fillna(0) * 100

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(df_pct))
    width = 0.6
    bottom = np.zeros(len(df_pct))

    for c, l, cl in zip(cols, lbl, colr):
        vals = df_pct[c].to_numpy()
        bars = ax.bar(x, vals, width, bottom=bottom, label=l, color=cl)
        for i, b in enumerate(bars):
            if b.get_height() > 2:
                ax.text(b.get_x()+width/2, bottom[i]+b.get_height()/2,
                        f"{b.get_height():.1f}%",
                        ha="center", va="center", color="white", fontsize=7)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(df["year"].astype(str))
    ax.set_ylabel("Percent of Revenue")
    ax.set_title(f"Expenses as % of Revenue — {ticker}")
    ax.set_ylim(0, 100)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"{ticker}_expense_percent_chart.png")
    plt.savefig(out, dpi=100)
    plt.close()
    print(f"✅ Saved percent chart → {out}")


# --------------------------------------------------------------------------- #
#  Orchestrator                                                               #
# --------------------------------------------------------------------------- #
def generate_expense_reports(ticker: str):
    print(f"\n=== Generating expense reports for {ticker} ===")
    reset_expense_tables()

# --------------------------------------------------------------------------- #
#  Run                                                                        #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    generate_expense_reports("AAPL")        # change ticker or loop as needed


# --------------------------------------------------------------------------- #
#  One-time table reset (run once, then delete)                               #
# --------------------------------------------------------------------------- #
def reset_expense_tables():
    """Create expense tables if missing, then clear them for a clean re-import."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Ensure both tables exist
    cur.execute("""
        CREATE TABLE IF NOT EXISTS IncomeStatement (
            ticker TEXT,
            period_ending TEXT,
            total_revenue REAL,
            cost_of_revenue REAL,
            research_and_development REAL,
            selling_and_marketing REAL,
            general_and_admin REAL,
            sga_combined REAL,
            PRIMARY KEY (ticker, period_ending)
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS QuarterlyIncomeStatement (
            ticker TEXT,
            period_ending TEXT,
            total_revenue REAL,
            cost_of_revenue REAL,
            research_and_development REAL,
            selling_and_marketing REAL,
            general_and_admin REAL,
            sga_combined REAL,
            PRIMARY KEY (ticker, period_ending)
        );
    """)

    # Now clear existing records
    cur.execute("DELETE FROM IncomeStatement;")
    cur.execute("DELETE FROM QuarterlyIncomeStatement;")
    print("✅ Cleared both IncomeStatement and QuarterlyIncomeStatement")

    conn.commit()
    conn.close()

# Uncomment this to run the reset ONCE, then delete or comment it again
# reset_expense_tables()

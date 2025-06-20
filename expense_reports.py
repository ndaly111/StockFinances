# expense_reports.py
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
#  Module constants & setup                                                  #
# --------------------------------------------------------------------------- #
DB_PATH     = "Stock Data.db"   # only used by __main__ example
OUTPUT_DIR  = "charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

__all__ = [
    "ensure_tables",
    "store_data",
    "fetch_yearly_data",
    "fetch_ttm_data",
    "plot_expense_charts",
    "plot_expense_percent_chart",
    "generate_expense_reports",
]


# --------------------------------------------------------------------------- #
#  Utility helpers                                                            #
# --------------------------------------------------------------------------- #
def clean_value(val):
    """Convert NaN → None, datetime → ISO string."""
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
def ensure_tables(conn: sqlite3.Connection, *, drop: bool = False) -> None:
    """
    Make sure both tables exist (and optionally drop/recreate them).
    """
    cur = conn.cursor()
    for tbl in TABLES:
        if drop:
            cur.execute(f"DROP TABLE IF EXISTS {tbl};")
        cur.execute(TABLE_SCHEMA.format(name=tbl))
    conn.commit()
    if drop:
        print("✅ Tables dropped & recreated with fresh schema")
    else:
        print("✅ Tables ensured (created if missing)")


# --------------------------------------------------------------------------- #
#  Storage helpers                                                            #
# --------------------------------------------------------------------------- #
def store_data(conn: sqlite3.Connection, ticker: str, *, mode: str = "annual") -> None:
    """
    Fetch annual or quarterly financials via yfinance and upsert into SQLite.
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
    rows: list[tuple] = []

    for idx, row in raw_df.iterrows():
        pe         = idx.to_pydatetime() if isinstance(idx, pd.Timestamp) else idx
        total_rev  = row.get("Total Revenue")
        cost_rev, rnd, mkt, adm, sga, fda, ppl, ins, oth = extract_expenses(row)
        rows.append((
            ticker,
            clean_value(pe),
            clean_value(total_rev),
            clean_value(cost_rev),
            clean_value(rnd),
            clean_value(mkt),
            clean_value(adm),
            clean_value(sga),
            clean_value(fda),
            clean_value(ppl),
            clean_value(ins),
            clean_value(oth),
        ))

    conn.executemany(
        f"INSERT OR REPLACE INTO {table} VALUES (?,?,?,?,?,?,?,?,?,?,?,?);",
        rows
    )
    conn.commit()
    print(f"✅ {mode.capitalize()} data stored for {ticker}")


# --------------------------------------------------------------------------- #
#  Data fetchers                                                              #
# --------------------------------------------------------------------------- #
def fetch_yearly_data(conn: sqlite3.Connection, ticker: str) -> pd.DataFrame:
    """
    Read IncomeStatement, group by calendar year, sum everything.
    """
    df = pd.read_sql_query(
        "SELECT * FROM IncomeStatement WHERE ticker = ?",
        conn,
        params=(ticker,)
    )
    if df.empty:
        return df

    df["period_ending"] = pd.to_datetime(df["period_ending"])
    df["year"]          = df["period_ending"].dt.year
    agg_cols            = df.columns.difference(["ticker", "period_ending", "year"])

    return df.groupby("year", as_index=False)[agg_cols].sum()


def fetch_ttm_data(conn: sqlite3.Connection, ticker: str) -> pd.DataFrame:
    """
    Sum the most recent four quarters and label as "TTM".
    """
    df = pd.read_sql_query(
        "SELECT * FROM QuarterlyIncomeStatement WHERE ticker = ? ORDER BY period_ending DESC",
        conn,
        params=(ticker,)
    )
    if df.empty or len(df) < 4:
        return pd.DataFrame()

    df["period_ending"] = pd.to_datetime(df["period_ending"])
    recent = df.head(4).sort_values("period_ending")

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
    if absx >= 1e12: return f"${x/1e12:.{dec}f} T"
    if absx >= 1e9:  return f"${x/1e9:.{dec}f} B"
    if absx >= 1e6:  return f"${x/1e6:.{dec}f} M"
    if absx >= 1e3:  return f"${x/1e3:.{dec}f} K"
    return f"${x:.{dec}f}"


def plot_expense_charts(df: pd.DataFrame, ticker: str):
    """
    Create a stacked bar chart of absolute expenses vs. revenue.
    """
    numeric_cols = df.columns.difference(["year"])
    df_plot      = df.copy()
    df_plot[numeric_cols] = df_plot[numeric_cols].fillna(0)

    # Remove all-zero rows
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
        ("cost_of_revenue",         "Cost of Revenue"),
        ("research_and_development","R&D"),
        ("selling_and_marketing",   "Sales & Marketing"),
        ("general_and_admin",       "G&A"),
        ("sga_combined",            "SG&A"),
        ("facilities_da",           "Facilities / D&A"),
        ("personnel_costs",         "Personnel"),
        ("insurance_claims",        "Insurance"),
        ("other_operating",         "Other Op."),
    ]

    for col, lbl in categories:
        if col in df_plot.columns and df_plot[col].sum() > 0:
            vals = df_plot[col].to_numpy()
            bars = ax.bar(
                pos - width/2,
                vals,
                width,
                bottom=bottom,
                label=lbl
            )
            for i, b in enumerate(bars):
                h = vals[i]
                if h > 0:
                    ax.text(
                        b.get_x() + b.get_width()/2,
                        bottom[i] + h/2,
                        format_short(h, 0),
                        ha="center", va="center",
                        color="white", fontsize=7
                    )
            bottom += vals

    # Revenue on top
    revs = df_plot["total_revenue"].to_numpy()
    bars = ax.bar(
        pos + width/2,
        revs,
        width,
        label="Revenue"
    )
    for b in bars:
        ax.text(
            b.get_x() + b.get_width()/2,
            b.get_height(),
            format_short(b.get_height(), 0),
            ha="center", va="bottom",
            fontsize=8, weight="bold"
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


def plot_expense_percent_chart(df: pd.DataFrame, ticker: str):
    """
    Create a stacked bar chart of expenses as % of revenue.
    """
    df_plot = df.copy()
    df_plot["total_revenue"] = pd.to_numeric(df_plot["total_revenue"], errors="coerce").fillna(0)
    numeric_cols = df_plot.columns.difference(["year", "total_revenue"])
    df_plot[numeric_cols] = df_plot[numeric_cols].apply(
        lambda s: pd.to_numeric(s, errors="coerce").fillna(0)
    )

    df_plot = df_plot.loc[df_plot["total_revenue"] != 0]
    if df_plot.empty:
        print("⛔ No valid revenue years — skipping percent-of-revenue chart")
        return

    revenue_nonzero = df_plot["total_revenue"].mask(df_plot["total_revenue"] == 0, np.nan)
    pct_df = pd.DataFrame({"year": df_plot["year"]})
    for col in numeric_cols:
        pct_df[col] = df_plot[col].divide(revenue_nonzero) * 100

    percent_cols = pct_df.columns.difference(["year"])
    nonzero_cats = [col for col in percent_cols if pct_df[col].sum() > 0]
    if not nonzero_cats:
        print("⛔ All expense categories are zero percent — skipping chart")
        return

    years = pct_df["year"].astype(str)
    pos   = np.arange(len(pct_df))
    width = 0.6
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bottom = np.zeros(len(pct_df))

    category_info = [
        ("cost_of_revenue",         "Cost of Revenue"),
        ("research_and_development","R&D"),
        ("selling_and_marketing",   "Sales & Marketing"),
        ("general_and_admin",       "G&A"),
        ("sga_combined",            "SG&A"),
        ("facilities_da",           "Facilities / D&A"),
        ("personnel_costs",         "Personnel"),
        ("insurance_claims",        "Insurance"),
        ("other_operating",         "Other Op."),
    ]

    for col, lbl in category_info:
        if col in nonzero_cats:
            vals = pct_df[col].to_numpy()
            bars = ax.bar(
                pos,
                vals,
                width,
                bottom=bottom,
                label=lbl
            )
            for i, b in enumerate(bars):
                p = vals[i]
                if p > 2:
                    ax.text(
                        b.get_x() + width/2,
                        bottom[i] + p/2,
                        f"{p:.1f}%",
                        ha="center", va="center",
                        color="white", fontsize=7
                    )
            bottom += vals

    ax.set_xticks(pos)
    ax.set_xticklabels(years)
    ax.set_ylabel("Percent of Revenue")
    ax.set_title(f"Expenses as % of Revenue — {ticker}")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    plt.tight_layout()

    outfile = os.path.join(OUTPUT_DIR, f"{ticker}_expense_percent_chart.png")
    plt.savefig(outfile, dpi=100)
    plt.close()
    print(f"✅ Saved percent-of-revenue chart → {outfile}")


# --------------------------------------------------------------------------- #
#  High‐level orchestrator                                                    #
# --------------------------------------------------------------------------- #
def generate_expense_reports(
    conn: sqlite3.Connection,
    ticker: str,
    *,
    rebuild_schema: bool = False
) -> None:
    """
    Orchestrate a full expense‐report run for one ticker.
    """
    ensure_tables(conn, drop=rebuild_schema)
    store_data(conn, ticker, mode="annual")
    store_data(conn, ticker, mode="quarterly")

    yearly = fetch_yearly_data(conn, ticker)
    if yearly.empty:
        print("⛔ No data found — skipping charts")
        return

    ttm  = fetch_ttm_data(conn, ticker)
    full = pd.concat([yearly, ttm], ignore_index=True)

    plot_expense_charts(full, ticker)
    plot_expense_percent_chart(full, ticker)


# --------------------------------------------------------------------------- #
#  Example CLI                                                               #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH, timeout=10)
    generate_expense_reports(conn, "AAPL")
    conn.close()

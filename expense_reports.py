"""
expense_reports.py
-------------------------------------------------------------------------------
Builds annual / quarterly operating-expense tables, stores them in SQLite,
and generates for each ticker:

  1) Revenue vs stacked expenses chart   (absolute $)
  2) Expenses as % of revenue chart
  3) YoY expense-change table (HTML)
"""

from __future__ import annotations

import os, sqlite3, math
from datetime import datetime

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

# ───────────────────────────────────
# Configuration
# ───────────────────────────────────
DB_PATH        = "Stock Data.db"
OUTPUT_DIR     = "charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

__all__ = ["generate_expense_reports"]

# ───────────────────────────────────
# Helpers
# ───────────────────────────────────
def clean_value(val):
    if pd.isna(val):
        return None
    if isinstance(val, (pd.Timestamp, datetime)):
        return val.isoformat()
    return val

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

# ───────────────────────────────────
# SQLite schema
# ───────────────────────────────────
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

def ensure_tables(*, drop: bool = False, conn: sqlite3.Connection | None = None):
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

# ───────────────────────────────────
# Data ingestion
# ───────────────────────────────────
def store_data(ticker: str, *, mode: str = "annual", conn: sqlite3.Connection | None = None):
    yf_tkr = yf.Ticker(ticker)
    raw_df = yf_tkr.financials.transpose() if mode == "annual" else yf_tkr.quarterly_financials.transpose()

    if raw_df.empty:
        print(f"⚠️  No {mode} financials found for {ticker}")
        return

    new_conn = conn is None
    if new_conn:
        conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
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
            clean_value(adm), clean_value(sga), clean_value(fda),
            clean_value(ppl), clean_value(ins), clean_value(oth)
        ))

    conn.commit()
    if new_conn:
        conn.close()

# ───────────────────────────────────
# Helpers to fetch & aggregate data
# ───────────────────────────────────
def fetch_yearly_data(ticker: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM IncomeStatement WHERE ticker = ?", conn, params=(ticker,))
    conn.close()

    if df.empty:
        return pd.DataFrame()

    df["period_ending"] = pd.to_datetime(df["period_ending"])
    df["year_int"]      = df["period_ending"].dt.year

    agg_cols = df.columns.difference(["ticker", "period_ending", "year_int"])
    grouped  = df.groupby("year_int", as_index=False)[agg_cols].sum()
    grouped["year_label"] = grouped["year_int"].astype(int).astype(str)
    return grouped

def fetch_ttm_data(ticker: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT * FROM QuarterlyIncomeStatement
        WHERE ticker = ?
        ORDER BY period_ending DESC
    """, conn, params=(ticker,))
    conn.close()

    if df.empty():
        return pd.DataFrame()

    df["period_ending"] = pd.to_datetime(df["period_ending"])
    recent = df.head(4).sort_values("period_ending")
    if len(recent) < 4:
        return pd.DataFrame()

    expected = pd.date_range(end=recent["period_ending"].max(), periods=4, freq="Q")
    if list(expected.to_period("Q")) != list(recent["period_ending"].dt.to_period("Q")):
        return pd.DataFrame()

    ttm = recent.drop(columns=["ticker", "period_ending"]).sum().to_frame().T
    ttm.insert(0, "year_label", "TTM")
    ttm["year_int"] = np.nan
    return ttm

# ───────────────────────────────────
# Chart helpers
# ───────────────────────────────────
def _format_short(x, _pos=None, dec=1):
    if pd.isna(x): return "$0"
    absx = abs(x)
    if absx >= 1e12: return f"${x/1e12:.{dec}f} T"
    if absx >= 1e9:  return f"${x/1e9:.{dec}f} B"
    if absx >= 1e6:  return f"${x/1e6:.{dec}f} M"
    if absx >= 1e3:  return f"${x/1e3:.{dec}f} K"
    return f"${x:.{dec}f}"

def plot_expense_charts(full: pd.DataFrame, ticker: str):
    full = full.copy().sort_values("year_int")
    x_labels = full["year_label"].tolist()

    use_combined = full["sga_combined"].notna().any()
    categories = [
        ("Cost of Revenue",     "cost_of_revenue",          "#6d6d6d"),
        ("R&D",                 "research_and_development", "blue"),
        ("G&A",                 "general_and_admin",        "#ffb3c6"),
        ("Selling & Marketing", "selling_and_marketing",    "#ffc6e2"),
        ("SG&A",                "sga_combined",             "#c2a5ff"),
        ("Facilities / D&A",    "facilities_da",            "orange"),
    ]
    if use_combined:
        categories = [(lbl, col, c) for lbl, col, c in categories if col not in ("general_and_admin", "selling_and_marketing")]

    fig, ax = plt.subplots(figsize=(11, 6))
    bottoms = np.zeros(len(full))
    for label, col, color in categories:
        vals = full[col].fillna(0).values
        ax.bar(x_labels, vals, bottom=bottoms, label=label, color=color, width=0.6)
        bottoms += vals

    ax.plot(x_labels, full["total_revenue"].values, color="black", linewidth=2, marker="o", label="Revenue")
    ax.set_ylim(0, max(bottoms.max(), full["total_revenue"].max()) * 1.10)
    ax.set_title(f"Revenue vs Operating Expenses — {ticker}")
    ax.yaxis.set_major_formatter(FuncFormatter(_format_short))
    ax.set_ylabel("USD")
    ax.legend(frameon=False, ncol=2)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"{ticker}_expenses_vs_revenue.png"))
    plt.close(fig)

def plot_expense_percent_chart(full: pd.DataFrame, ticker: str):
    full = full.copy().sort_values("year_int")
    full = full.loc[full["total_revenue"] != 0].reset_index(drop=True)
    x_labels = full["year_label"].tolist()

    use_combined = full["sga_combined"].notna().any()
    categories = [
        ("Cost of Revenue",     "cost_of_revenue",          "#6d6d6d"),
        ("R&D",                 "research_and_development", "blue"),
        ("G&A",                 "general_and_admin",        "#ffb3c6"),
        ("Selling & Marketing", "selling_and_marketing",    "#ffc6e2"),
        ("SG&A",                "sga_combined",             "#c2a5ff"),
        ("Facilities / D&A",    "facilities_da",            "orange"),
    ]
    if use_combined:
        categories = [(lbl, col, c) for lbl, col, c in categories if col not in ("general_and_admin", "selling_and_marketing")]

    for _lbl, col, _c in categories:
        pct_col = col + "_pct"
        full[pct_col] = (full[col] / full["total_revenue"] * 100).where(full["total_revenue"] != 0)

    fig, ax = plt.subplots(figsize=(11, 6))
    bottoms = np.zeros(len(full))
    for label, col, color in categories:
        vals = full[col + "_pct"].fillna(0).values
        ax.bar(x_labels, vals, bottom=bottoms, label=label, color=color, width=0.6)
        for x, y0, v in zip(x_labels, bottoms, vals):
            if v > 4:
                ax.text(x, y0 + v/2, f"{v:.1f}%", ha="center", va="center", fontsize=8, color="white")
        bottoms += vals

    ax.axhline(100, linestyle="--", linewidth=1, color="black", label="100% of Revenue", zorder=5)
    ylim_max = max(110, (int(bottoms.max()/10) + 2) * 10)
    ax.set_ylim(0, ylim_max)
    ax.set_yticks(np.arange(0, ylim_max + 1, 10))
    ax.set_ylabel("Percent of Revenue")
    ax.set_title(f"Expenses as % of Revenue — {ticker}")
    ax.legend(frameon=False, ncol=2)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"{ticker}_expenses_pct_of_rev.png"))
    plt.close(fig)

# ───────────────────────────────────
# Main mini-entry-point
# ───────────────────────────────────
def generate_expense_reports(ticker: str, *, rebuild_schema: bool = False, conn: sqlite3.Connection | None = None):
    ensure_tables(drop=rebuild_schema, conn=conn)
    store_data(ticker, mode="annual", conn=conn)
    store_data(ticker, mode="quarterly", conn=conn)

    yearly = fetch_yearly_data(ticker)
    if yearly.empty:
        print(f"⛔ No data found for {ticker} — skipping charts")
        return

    full = pd.concat([yearly, fetch_ttm_data(ticker)], ignore_index=True)
    plot_expense_charts(full, ticker)
    plot_expense_percent_chart(full, ticker)

    # ── YoY Expense-change table ───────────────────────────────────────────────
    yoy_df = full[[
        "year_label", "total_revenue", "cost_of_revenue",
        "research_and_development", "selling_and_marketing", "general_and_admin"
    ]].sort_values("year_label")

    # Drop years with missing / zero revenue
    yoy_df = yoy_df[yoy_df["total_revenue"].notna() & (yoy_df["total_revenue"] != 0)]

    for col in yoy_df.columns[1:]:
        yoy_df[col] = (
            yoy_df[col].pct_change()
            .replace([np.inf, -np.inf], np.nan)
            .round(4) * 100
        )

    yoy_df = yoy_df.rename(columns={
        "year_label": "Year",
        "total_revenue": "Revenue Change (%)",
        "cost_of_revenue": "Cost of Revenue Change (%)",
        "research_and_development": "R&D Change (%)",
        "selling_and_marketing": "Sales & Marketing Change (%)",
        "general_and_admin": "G&A Change (%)"
    })

    html_path = os.path.join(OUTPUT_DIR, f"{ticker}_yoy_expense_change.html")
    html_content = (
        '<div class="scroll-table-wrapper">' +
        yoy_df.to_html(index=False, classes="expense-table", border=0) +
        '</div>'
    )
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[{ticker}] YoY expense-change table saved ➜ {html_path}")

# ───────────────────────────────────
# Stand-alone execution
# ───────────────────────────────────
if __name__ == "__main__":
    generate_expense_reports("AAPL")

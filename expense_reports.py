# expense_reports.py
# -----------------------------------------------------------------------------
# Builds annual / quarterly operating-expense tables, stores them in SQLite,
# and generates two charts per ticker:
#
#   1) Revenue vs. stacked expenses (absolute $)
#   2) Expenses as % of revenue
#   3) YoY expense-change HTML table
#   4) Absolute expense-dollar HTML table
# -----------------------------------------------------------------------------

from __future__ import annotations
import os, sqlite3
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import yfinance as yf

from expense_labels import (
    COST_OF_REVENUE, RESEARCH_AND_DEVELOPMENT, SELLING_AND_MARKETING,
    GENERAL_AND_ADMIN, SGA_COMBINED, FACILITIES_DA, PERSONNEL_COSTS,
    INSURANCE_CLAIMS, OTHER_OPERATING,
)

DB_PATH        = "Stock Data.db"
OUTPUT_DIR     = "charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)
__all__ = ["generate_expense_reports"]

def _fetch_financials(ticker: str, freq: str = "annual") -> pd.DataFrame:
    yf_ticker = yf.Ticker(ticker)
    if freq == "annual":
        df = yf_ticker.financials
    else:
        df = yf_ticker.quarterly_financials
    return df.T.reset_index().rename(columns={"index": "Date"})

def _is_empty(col: pd.Series) -> bool:
    return (col.replace(0, np.nan).notna().sum() == 0)

def _format_dollar(val):
    return f"${val/1e9:.1f}B" if val > 1e9 else f"${val/1e6:.1f}M"

def _cats(df: pd.DataFrame) -> dict[str, list[str]]:
    # Use SG&A fallback logic
    has_sga = not _is_empty(df.get("Selling General and Administrative", pd.Series()))
    has_sep = (
        not _is_empty(df.get("Selling And Marketing Expense", pd.Series())) or
        not _is_empty(df.get("General And Administrative Expense", pd.Series()))
    )

    return {
        "Cost of Revenue": COST_OF_REVENUE,
        "R&D": RESEARCH_AND_DEVELOPMENT,
        "Sales & Marketing": SELLING_AND_MARKETING if has_sep else [],
        "G&A": GENERAL_AND_ADMIN if has_sep else [],
        "SG&A": SGA_COMBINED if has_sga and not has_sep else [],
    }

def _prepare_expense_df(df: pd.DataFrame, cats: dict[str, list[str]]) -> pd.DataFrame:
    df = df.copy()
    df["Revenue"] = df["Total Revenue"]
    df = df.loc[df["Revenue"] != 0]
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    df["year_label"] = df["Date"].dt.year.astype(str)
    if df.iloc[-1]["Date"] > pd.Timestamp.now() - pd.DateOffset(months=15):
        df.iloc[-1, df.columns.get_loc("year_label")] = "TTM"

    for cat_name, fields in cats.items():
        df[cat_name] = df[fields].sum(axis=1, min_count=1)
    return df

def chart_abs(df: pd.DataFrame, ticker: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    fields = ["Cost of Revenue", "R&D", "Sales & Marketing", "G&A", "SG&A"]
    fields = [f for f in fields if f in df]
    bottoms = np.zeros(len(df))

    for field in fields:
        ax.bar(df["year_label"].astype(str), df[field] / 1e9, label=field, bottom=bottoms)
        bottoms += df[field].fillna(0).values

    ax.plot(df["year_label"].astype(str), df["Revenue"] / 1e9, label="Revenue", color="black", marker='o')
    ax.set_title(f"{ticker} Revenue vs. Operating Expenses")
    ax.set_ylabel("Amount ($B)")
    ax.legend(loc="upper left", fontsize="small")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/{ticker}_expense_stack.png")
    plt.close(fig)

def chart_pct(df: pd.DataFrame, ticker: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    fields = ["Cost of Revenue", "R&D", "Sales & Marketing", "G&A", "SG&A"]
    fields = [f for f in fields if f in df]

    bottoms = np.zeros(len(df))
    for field in fields:
        pct = df[field] / df["Revenue"]
        ax.bar(df["year_label"].astype(str), pct * 100, label=field, bottom=bottoms)
        bottoms += pct.fillna(0).values

    ax.set_title(f"{ticker} Expenses as % of Revenue")
    ax.set_ylabel("Percent of Revenue")
    ax.legend(loc="upper left", fontsize="small")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/{ticker}_expense_pct.png")
    plt.close(fig)

def write_html(df: pd.DataFrame, ticker: str):
    fields = ["Cost of Revenue", "R&D", "Sales & Marketing", "G&A", "SG&A"]
    fields = [f for f in fields if f in df]
    df_out = df[["year_label"] + fields].copy()
    df_out = df_out.rename(columns={"year_label": "Year"})
    df_out = df_out.set_index("Year")

    abs_df = df_out.applymap(_format_dollar)
    abs_df.to_html(f"{OUTPUT_DIR}/{ticker}_expense_table.html", escape=False)

    yoy = df_out.copy()
    for c in yoy.columns:
        yoy[c] = (
            yoy[c].astype(float)
                   .pct_change()
                   .replace([np.inf, -np.inf], np.nan)
                   .round(4) * 100
        )
    yoy = yoy.round(1).astype(str) + "%"
    yoy.to_html(f"{OUTPUT_DIR}/{ticker}_yoy_expense_change.html", escape=False)

def store(ticker: str, mode: str = "annual", conn=None):
    df = _fetch_financials(ticker, freq=mode)
    if df is None or df.empty or "Total Revenue" not in df:
        return

    cats = _cats(df)
    df = _prepare_expense_df(df, cats)
    chart_abs(df, ticker)
    chart_pct(df, ticker)
    write_html(df, ticker)

    if conn:
        df.to_sql(f"{ticker}_expense_{mode}", conn, if_exists="replace", index=False)

def ensure(drop=False, conn=None):
    if conn is None:
        conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    if drop:
        cur.execute("DROP TABLE IF EXISTS expense_data")
    cur.execute('''CREATE TABLE IF NOT EXISTS expense_data (
        Ticker TEXT,
        Date TEXT,
        Category TEXT,
        Amount REAL
    )''')
    conn.commit()

def generate_expense_reports(ticker: str, *, rebuild_schema=False, conn=None):
    ensure(drop=rebuild_schema, conn=conn)
    store(ticker, mode="annual", conn=conn)
    store(ticker, mode="quarterly", conn=conn)

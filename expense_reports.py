"""
expense_reports.py
-------------------------------------------------------------------------------
Per-ticker outputs
  1) Revenue-vs-stacked-expense chart        ($)
  2) Expenses-as-%-of-revenue chart          (%)
  3) YoY expense-change HTML table           (%)
  4) Absolute expense-dollar HTML table      ($)
"""

from __future__ import annotations
import os, sqlite3
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib import colors as mcolors
import numpy as np
import pandas as pd
import yfinance as yf

from expense_labels import (
    COST_OF_REVENUE, RESEARCH_AND_DEVELOPMENT, SELLING_AND_MARKETING,
    GENERAL_AND_ADMIN, SGA_COMBINED, FACILITIES_DA, PERSONNEL_COSTS,
    INSURANCE_CLAIMS, OTHER_OPERATING,
)

DB_PATH, OUTPUT_DIR = "Stock Data.db", "charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)
__all__ = ["generate_expense_reports"]

CATEGORY_MAP = {
    "Cost of Revenue ($)": COST_OF_REVENUE,
    "R&D ($)": RESEARCH_AND_DEVELOPMENT,
    "Sales & Marketing ($)": SELLING_AND_MARKETING,
    "G&A ($)": GENERAL_AND_ADMIN,
    "SG&A ($)": SGA_COMBINED,
    "Facilities / D&A ($)": FACILITIES_DA,
    "Personnel Costs ($)": PERSONNEL_COSTS,
    "Insurance / Claims ($)": INSURANCE_CLAIMS,
    "Other Operating ($)": OTHER_OPERATING,
}
CATEGORY_COLORS = [
    "#6d6d6d", "blue", "#ffc6e2", "#ffb3c6", "#c2a5ff", "orange", "green", "brown", "#999999"
]
CATEGORY_KEYS = list(CATEGORY_MAP.keys())

def _fmt_short(x: float, d: int = 1) -> str:
    if pd.isna(x): return ""
    for div, suf in [(1e12,"T"),(1e9,"B"),(1e6,"M"),(1e3,"K")]:
        if abs(x) >= div:
            return f"${x/div:.{d}f}{suf}"
    return f"${x:.{d}f}"

def _all_nan_or_zero(col: pd.Series) -> bool:
    return (col.replace(0, np.nan).notna().sum() == 0)

def pick_any(row, labels):
    for k in row.index:
        if pd.notna(row[k]) and any(lbl.lower() in k.lower() for lbl in labels):
            return row[k]
    return None

def extract_expenses(row):
    return {k: pick_any(row, CATEGORY_MAP[k]) for k in CATEGORY_KEYS}

def clean(v):
    if pd.isna(v): return None
    return v.isoformat() if isinstance(v, (pd.Timestamp, datetime)) else v

def ensure(drop=False, *, conn=None):
    own = conn is None
    if own: conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    base = "ticker TEXT, period_ending TEXT, total_revenue REAL"
    fields = ", ".join([f'"{k}" REAL' for k in CATEGORY_KEYS])
    schema = f"CREATE TABLE IF NOT EXISTS {{n}} ({base}, {fields}, PRIMARY KEY(ticker,period_ending));"
    for t in ("IncomeStatement", "QuarterlyIncomeStatement"):
        if drop: cur.execute(f"DROP TABLE IF EXISTS {t}")
        cur.execute(schema.format(n=t))
    conn.commit(); cur.close()
    if own: conn.close()

def store(tkr, *, mode="annual", conn=None):
    df = (yf.Ticker(tkr).financials.transpose()
          if mode == "annual"
          else yf.Ticker(tkr).quarterly_financials.transpose())
    if df.empty: return
    own = conn is None
    if own: conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    table = "IncomeStatement" if mode == "annual" else "QuarterlyIncomeStatement"
    for idx, row in df.iterrows():
        pe = idx.to_pydatetime() if isinstance(idx, pd.Timestamp) else idx
        expenses = extract_expenses(row)
        cur.execute(
            f"INSERT OR REPLACE INTO {table} VALUES ({','.join(['?']*(3 + len(CATEGORY_KEYS)))})",
            [tkr, clean(pe), clean(row.get("Total Revenue"))] + [clean(expenses[k]) for k in CATEGORY_KEYS]
        )
    conn.commit(); cur.close()
    if own: conn.close()

def yearly(tkr, conn=None):
    own = conn is None
    if own: conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM IncomeStatement WHERE ticker=?", conn, params=(tkr,))
    if own: conn.close()
    if df.empty: return df
    df["period_ending"] = pd.to_datetime(df["period_ending"])
    df["year_int"] = df["period_ending"].dt.year
    g = df.groupby("year_int", as_index=False).sum(numeric_only=True)
    g["year_label"] = g["year_int"].astype(str)
    return g

def ttm(tkr, conn=None):
    own = conn is None
    if own: conn = sqlite3.connect(DB_PATH)
    q = pd.read_sql_query(
        "SELECT * FROM QuarterlyIncomeStatement WHERE ticker=? ORDER BY period_ending DESC",
        conn, params=(tkr,))
    if own: conn.close()
    if q.empty: return q
    q["period_ending"] = pd.to_datetime(q["period_ending"])
    recent = q.head(4).sort_values("period_ending")
    if len(recent) < 4: return pd.DataFrame()
    expect = pd.date_range(end=recent["period_ending"].max(), periods=4, freq="Q")
    if list(expect.to_period("Q")) != list(recent["period_ending"].dt.to_period("Q")):
        return pd.DataFrame()
    ttm_df = recent.drop(columns=["ticker", "period_ending"]).sum().to_frame().T
    ttm_df.insert(0, "year_label", "TTM"); ttm_df["year_int"] = np.nan
    return ttm_df

def chart_abs(df, tkr):
    f = df.sort_values("year_int"); xl = f["year_label"].tolist()
    cats = [(k, k, c) for k, c in zip(CATEGORY_KEYS, CATEGORY_COLORS)
            if k in f.columns and not _all_nan_or_zero(f[k])]
    fig, ax = plt.subplots(figsize=(11, 6)); bot = np.zeros(len(f))
    for label, col, color in cats:
        v = f[col].fillna(0).values
        ax.bar(xl, v, bottom=bot, color=color, label=label)
        bot += v
    ax.plot(xl, f["total_revenue"], "k-o", lw=2, label="Revenue")
    ax.set_ylim(0, max(bot.max(), f["total_revenue"].max()) * 1.1)
    ax.set_title(f"Revenue vs Operating Expenses — {tkr}")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: _fmt_short(x)))
    ax.legend(frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{tkr}_expenses_vs_revenue.png"))
    plt.close()

def chart_pct(df, tkr):
    f = df.sort_values("year_int").loc[lambda d: d["total_revenue"] != 0]
    xl = f["year_label"].tolist()
    cats = [(k, k, c) for k, c in zip(CATEGORY_KEYS, CATEGORY_COLORS)
            if k in f.columns and not _all_nan_or_zero(f[k])]
    for _, col, _ in cats:
        f[col + "_pct"] = f[col] / f["total_revenue"] * 100
    fig, ax = plt.subplots(figsize=(11, 4))
    bot = np.zeros(len(f))
    for label, col, color in cats:
        vals = f[col + "_pct"].fillna(0).values
        ax.bar(xl, vals, bottom=bot, color=color, width=.6)
        bot += vals
    ax.axhline(100, ls="--", color="black")
    ax.set_ylim(0, np.ceil((bot.max() + 8) / 10) * 10)
    ax.set_yticks(np.arange(0, ax.get_ylim()[1]+1, 10))
    ax.set_title(f"Expenses as % of Revenue — {tkr}")
    ax.set_ylabel("Percent of Revenue")
    ax.legend([c[0] for c in cats], bbox_to_anchor=(1.01, 0.5), loc="center left", frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{tkr}_expenses_pct_of_rev.png"), dpi=120, bbox_inches="tight")
    plt.close()

def write_html(df: pd.DataFrame, out_path: str):
    df.to_html(out_path, index=False, border=0, justify="center")

def generate_expense_reports(tkr, *, rebuild_schema=False, conn=None):
    ensure(drop=rebuild_schema, conn=conn)
    store(tkr, mode="annual", conn=conn)
    store(tkr, mode="quarterly", conn=conn)
    yearly_df = yearly(tkr, conn=conn)
    if yearly_df.empty: return
    full = pd.concat([yearly_df, ttm(tkr, conn=conn)], ignore_index=True)
    chart_abs(full, tkr)
    chart_pct(full, tkr)

    display_cols = ["year_label", "total_revenue"] + [k for k in CATEGORY_KEYS if k in full.columns]
    df = full[display_cols].sort_values("year_label")
    df = df[df["total_revenue"].notna() & (df["total_revenue"] != 0)]
    df = df.drop(columns=[c for c in df.columns[2:] if _all_nan_or_zero(df[c])])

    fmt = df.copy()
    for c in fmt.columns[1:]:
        fmt[c] = fmt[c].apply(_fmt_short)
    fmt = fmt.rename(columns={"year_label": "Year", "total_revenue": "Revenue ($)"})
    write_html(fmt, os.path.join(OUTPUT_DIR, f"{tkr}_expense_absolute.html"))

    yoy = df.copy()
    for c in yoy.columns[2:]:
        yoy[c] = yoy[c].pct_change().replace([np.inf, -np.inf], np.nan) * 100
    yoy = yoy.rename(columns={"year_label": "Year", "total_revenue": "Revenue Change (%)"})
    yoy = yoy[yoy.iloc[:,2:].notna().any(axis=1)]
    write_html(yoy.round(1), os.path.join(OUTPUT_DIR, f"{tkr}_yoy_expense_change.html"))

if __name__ == "__main__":
    generate_expense_reports("AAPL")

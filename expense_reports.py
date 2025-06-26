"""
expense_reports.py
-------------------------------------------------------------------------------
Outputs per ticker
  1) Revenue-vs-stacked-expense chart        ($)
  2) Expenses-as-%-of-revenue chart          (%)
  3) YoY expense-change HTML table           (%)
  4) Absolute expense-dollar HTML table      ($)
"""

from __future__ import annotations
import os, sqlite3
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

DB_PATH, OUTPUT_DIR = "Stock Data.db", "charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)
__all__ = ["generate_expense_reports"]

# ───────────────── helper utils ─────────────────
_SUFFIXES = [(1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")]

def _fmt_short(x: float, d: int = 1) -> str:
    """Format large numbers with K/M/B/T suffix, one-decimal max."""
    if pd.isna(x):
        return ""
    for div, suf in _SUFFIXES:          # ← typo fixed here
        if abs(x) >= div:
            return f"${x/div:.{d}f}{suf}"
    return f"${x:.{d}f}"

def clean(v):
    if pd.isna(v):
        return None
    return v.isoformat() if isinstance(v, (pd.Timestamp, datetime)) else v

def pick_any(row: pd.Series, labels):
    for k in row.index:
        if pd.notna(row[k]) and any(lbl.lower() in k.lower() for lbl in labels):
            return row[k]
    return None

def extract_expenses(r: pd.Series):
    return (
        pick_any(r, COST_OF_REVENUE),
        pick_any(r, RESEARCH_AND_DEVELOPMENT),
        pick_any(r, SELLING_AND_MARKETING),
        pick_any(r, GENERAL_AND_ADMIN),
        pick_any(r, SGA_COMBINED),
        pick_any(r, FACILITIES_DA),
        pick_any(r, PERSONNEL_COSTS),
        pick_any(r, INSURANCE_CLAIMS),
        pick_any(r, OTHER_OPERATING),
    )

# ───────────────── DB schema / ingest ─────────────────
TABLES = ("IncomeStatement", "QuarterlyIncomeStatement")
SCHEMA = """
CREATE TABLE IF NOT EXISTS {n}(
  ticker TEXT, period_ending TEXT,
  total_revenue REAL, cost_of_revenue REAL, research_and_development REAL,
  selling_and_marketing REAL, general_and_admin REAL, sga_combined REAL,
  facilities_da REAL, personnel_costs REAL, insurance_claims REAL,
  other_operating REAL, PRIMARY KEY(ticker,period_ending));
"""

def ensure(drop=False, *, conn=None):
    own = conn is None
    if own:
        conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    for t in TABLES:
        if drop:
            cur.execute(f"DROP TABLE IF EXISTS {t}")
        cur.execute(SCHEMA.format(n=t))
    conn.commit(); cur.close()
    if own:
        conn.close()

def store(tkr, *, mode="annual", conn=None):
    df = (yf.Ticker(tkr).financials.transpose()
          if mode == "annual"
          else yf.Ticker(tkr).quarterly_financials.transpose())
    if df.empty:
        return
    own = conn is None
    if own:
        conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    tbl = "IncomeStatement" if mode == "annual" else "QuarterlyIncomeStatement"
    for idx, row in df.iterrows():
        pe = idx.to_pydatetime() if isinstance(idx, pd.Timestamp) else idx
        c, r, m, a, s, f, p, i, o = extract_expenses(row)
        cur.execute(
            f"INSERT OR REPLACE INTO {tbl} VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (tkr, clean(pe), clean(row.get("Total Revenue")), clean(c), clean(r),
             clean(m), clean(a), clean(s), clean(f), clean(p), clean(i), clean(o)),
        )
    conn.commit(); cur.close()
    if own:
        conn.close()

# ───────────────── fetch helpers ─────────────────
def yearly(tkr):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM IncomeStatement WHERE ticker=?",
                           conn, params=(tkr,))
    conn.close()
    if df.empty:
        return df
    df["period_ending"] = pd.to_datetime(df["period_ending"])
    df["year_int"] = df["period_ending"].dt.year
    g = df.groupby("year_int", as_index=False).sum(numeric_only=True)
    g["year_label"] = g["year_int"].astype(str)
    return g

def ttm(tkr):
    conn = sqlite3.connect(DB_PATH)
    q = pd.read_sql_query(
        "SELECT * FROM QuarterlyIncomeStatement WHERE ticker=? ORDER BY period_ending DESC",
        conn, params=(tkr,))
    conn.close()
    if q.empty:
        return q
    q["period_ending"] = pd.to_datetime(q["period_ending"])
    rec = q.head(4).sort_values("period_ending")
    if len(rec) < 4:
        return pd.DataFrame()
    exp = pd.date_range(end=rec["period_ending"].max(), periods=4, freq="Q")
    if list(exp.to_period("Q")) != list(rec["period_ending"].dt.to_period("Q")):
        return pd.DataFrame()
    out = rec.drop(columns=["ticker", "period_ending"]).sum().to_frame().T
    out.insert(0, "year_label", "TTM"); out["year_int"] = np.nan
    return out

# ───────────────── chart helpers ─────────────────
def _cats(df, combo):
    base = [
        ("Cost of Revenue", "cost_of_revenue", "#6d6d6d"),
        ("R&D", "research_and_development", "blue"),
        ("G&A", "general_and_admin", "#ffb3c6"),
        ("Selling & Marketing", "selling_and_marketing", "#ffc6e2"),
        ("SG&A", "sga_combined", "#c2a5ff"),
        ("Facilities / D&A", "facilities_da", "orange"),
    ]
    if combo:
        base = [c for c in base if c[1] not in ("general_and_admin",
                                                "selling_and_marketing")]
    return [c for c in base if c[1] in df.columns]

def chart_abs(df, tkr):
    f = df.sort_values("year_int"); xl = f["year_label"].tolist()
    cats = _cats(f, f["sga_combined"].notna().any())
    fig, ax = plt.subplots(figsize=(11, 6)); bot = np.zeros(len(f))
    for lbl, col, clr in cats:
        v = f[col].fillna(0).values
        ax.bar(xl, v, bottom=bot, color=clr, width=.6, label=lbl); bot += v
    ax.plot(xl, f["total_revenue"], "k-o", lw=2, label="Revenue")
    ax.set_ylim(0, max(bot.max(), f["total_revenue"].max()) * 1.1)
    ax.set_title(f"Revenue vs Operating Expenses — {tkr}")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: _fmt_short(x)))
    ax.legend(frameon=False, ncol=2); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,
                             f"{tkr}_expenses_vs_revenue.png")); plt.close()

def chart_pct(df, tkr):
    f = df.sort_values("year_int"); f = f[f["total_revenue"] != 0]
    xl = f["year_label"].tolist(); cats = _cats(f, f["sga_combined"].notna().any())
    for _, c, _ in cats:
        f[c + "_pct"] = f[c] / f["total_revenue"] * 100
    fig, ax = plt.subplots(figsize=(11, 6)); bot = np.zeros(len(f))
    for lbl, c, clr in cats:
        v = f[c + "_pct"].fillna(0).values
        ax.bar(xl, v, bottom=bot, color=clr, width=.6, label=lbl)
        for x, y0, val in zip(xl, bot, v):
            if val > 4:
                ax.text(x, y0 + val / 2, f"{val:.1f}%", ha="center",
                        va="center", color="white", fontsize=8)
        bot += v
    ax.axhline(100, ls="--", lw=1, c="black")
    ymax = max(110, (int(bot.max() / 10) + 2) * 10)
    ax.set_ylim(0, ymax); ax.set_yticks(np.arange(0, ymax + 1, 10))
    ax.set_title(f"Expenses as % of Revenue — {tkr}")
    ax.legend(frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,
                             f"{tkr}_expenses

# expense_reports.py
# ────────────────────────────────────────────────────────────────
# Per-ticker outputs
#   1) Revenue-vs-stacked-expense chart        ($)
#   2) Expenses-as-%-of-revenue chart          (%)
#   3) YoY expense-change HTML table           (%)
#   4) Absolute expense-dollar HTML table      ($)
# ────────────────────────────────────────────────────────────────

from __future__ import annotations
import os, sqlite3
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib import colors as mcolors
import numpy as np
import pandas as pd
import yfinance as yf

# ──────────────────────────────────────────────────────────────
# labels defined in a separate file you already have
# ──────────────────────────────────────────────────────────────
from expense_labels import (
    COST_OF_REVENUE, RESEARCH_AND_DEVELOPMENT, SELLING_AND_MARKETING,
    GENERAL_AND_ADMIN, SGA_COMBINED, FACILITIES_DA, PERSONNEL_COSTS,
    INSURANCE_CLAIMS, OTHER_OPERATING,
)

DB_PATH, OUTPUT_DIR = "Stock Data.db", "charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)
__all__ = ["generate_expense_reports"]

# ─────────────────────────── helpers ──────────────────────────
_SUFFIXES = [(1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")]

def _fmt_short(x: float, d: int = 1) -> str:
    """Pretty-print numbers with K/M/B/T suffix."""
    if pd.isna(x):
        return ""
    for div, suf in _SUFFIXES:
        if abs(x) >= div:
            return f"${x/div:.{d}f}{suf}"
    return f"${x:.{d}f}"

def _all_nan_or_zero(col: pd.Series) -> bool:
    """True if entire column is NaN or zero."""
    return (col.replace(0, np.nan).notna().sum() == 0)

def clean(v):
    if pd.isna(v):
        return None
    return v.isoformat() if isinstance(v, (pd.Timestamp, datetime)) else v

def pick_any(row: pd.Series, labels: list[str]):
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

# ────────────────────── database schema / IO ─────────────────────
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
    table = "IncomeStatement" if mode == "annual" else "QuarterlyIncomeStatement"
    for idx, row in df.iterrows():
        pe = idx.to_pydatetime() if isinstance(idx, pd.Timestamp) else idx
        c, r, m, a, s, f, p, i, o = extract_expenses(row)
        cur.execute(
            f"INSERT OR REPLACE INTO {table} VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                tkr, clean(pe), clean(row.get("Total Revenue")),
                clean(c), clean(r), clean(m), clean(a),
                clean(s), clean(f), clean(p), clean(i), clean(o),
            ),
        )
    conn.commit(); cur.close()
    if own:
        conn.close()

# ─────────────────────── pull yearly / TTM ──────────────────────
def yearly(tkr):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM IncomeStatement WHERE ticker=?", conn, params=(tkr,))
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
    recent = q.head(4).sort_values("period_ending")
    if len(recent) < 4:
        return pd.DataFrame()
    expect = pd.date_range(end=recent["period_ending"].max(), periods=4, freq="Q")
    if list(expect.to_period("Q")) != list(recent["period_ending"].dt.to_period("Q")):
        return pd.DataFrame()
    ttm_df = recent.drop(columns=["ticker", "period_ending"]).sum().to_frame().T
    ttm_df.insert(0, "year_label", "TTM"); ttm_df["year_int"] = np.nan
    return ttm_df

# ───────────────────────── chart helpers ─────────────────────────
def _cats(df, combo):
    base = [
        ("Cost of Revenue",     "cost_of_revenue",          "#6d6d6d"),
        ("R&D",                 "research_and_development", "blue"),
        ("G&A",                 "general_and_admin",        "#ffb3c6"),
        ("Selling & Marketing", "selling_and_marketing",    "#ffc6e2"),
        ("SG&A",                "sga_combined",             "#c2a5ff"),
        ("Facilities / D&A",    "facilities_da",            "orange"),
    ]
    if combo:
        base = [c for c in base if c[1] not in ("general_and_admin","selling_and_marketing")]
    return [c for c in base if c[1] in df.columns]

# ──────────────────────────── CHART 1 ───────────────────────────
def chart_abs(df, tkr):
    f = df.sort_values("year_int").copy()

    # ensure numeric
    f["total_revenue"] = pd.to_numeric(f["total_revenue"], errors="coerce").fillna(0.0).astype(float)
    for _, col, _ in _cats(f, f["sga_combined"].notna().any()):
        f[col] = pd.to_numeric(f[col], errors="coerce").fillna(0.0).astype(float)

    xl   = f["year_label"].tolist()
    cats = _cats(f, f["sga_combined"].notna().any())

    fig, ax = plt.subplots(figsize=(11, 6))
    bottom = np.zeros(len(f), dtype=float)

    for label, col, colour in cats:
        vals = f[col].values.astype(float)
        ax.bar(xl, vals, bottom=bottom, color=colour, width=.6, label=label)
        bottom += vals

    ax.plot(xl, f["total_revenue"], "k-o", lw=2, label="Revenue")
    ax.set_ylim(0, max(bottom.max(), f["total_revenue"].max()) * 1.1)
    ax.set_title(f"Revenue vs Operating Expenses — {tkr}")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: _fmt_short(x)))
    ax.legend(frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{tkr}_expenses_vs_revenue.png"))
    plt.close()

# ──────────────────────────── CHART 2 ───────────────────────────
def chart_pct(df: pd.DataFrame, ticker: str):
    f = df.sort_values("year_int").loc[lambda d: d["total_revenue"] != 0].copy()
    xlab = f["year_label"].tolist()

    base = [
        ("Cost of Revenue",     "cost_of_revenue",          "#6d6d6d"),
        ("R&D",                 "research_and_development", "blue"),
        ("G&A",                 "general_and_admin",        "#ffb3c6"),
        ("Selling & Marketing", "selling_and_marketing",    "#ffc6e2"),
        ("SG&A",                "sga_combined",             "#c2a5ff"),
        ("Facilities / D&A",    "facilities_da",            "orange"),
    ]
    if f["sga_combined"].notna().any():
        base = [c for c in base if c[1] not in ("general_and_admin","selling_and_marketing")]
    cats = [c for c in base if c[1] in f.columns]

    # ensure revenue numeric
    f["total_revenue"] = pd.to_numeric(f["total_revenue"], errors="coerce").fillna(0.0).astype(float)

    for _, col, _ in cats:
        f[col] = pd.to_numeric(f[col], errors="coerce").fillna(0.0).astype(float)
        f[col + "_pct"] = f[col] / f["total_revenue"] * 100.0

    def _txt(clr: str) -> str:
        r, g, b = mcolors.to_rgb(clr)
        return "white" if (0.299*r + 0.587*g + 0.114*b) < 0.6 else "black"

    fig, ax = plt.subplots(figsize=(11, 4))
    bottom = np.zeros(len(f), dtype=float)

    for label, col, colour in cats:
        vals = f[col + "_pct"].values.astype(float)
        ax.bar(xlab, vals, bottom=bottom, color=colour, width=.6, zorder=2)
        for x, y0, v in zip(xlab, bottom, vals):
            if v > 4:
                ax.text(x, y0 + v/2, f"{v:.1f}%", ha="center", va="center",
                        fontsize=8, color=_txt(colour))
        bottom += vals

    ax.axhline(100, ls="--", lw=1, color="black", zorder=5)
    ylim = np.ceil((bottom.max() + max(8, bottom.max()*0.08)) / 10) * 10
    ax.set_ylim(0, ylim); ax.set_yticks(np.arange(0, ylim+1, 10))
    ax.set_ylabel("Percent of Revenue")
    ax.set_title(f"Expenses as % of Revenue — {ticker}")
    ax.legend([c[0] for c in cats], bbox_to_anchor=(1.01, 0.5),
              loc="center left", frameon=False)
    fig.subplots_adjust(right=0.78, top=0.88)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"{ticker}_expenses_pct_of_rev.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[{ticker}] expense-% chart saved → {out}")

# ───────────────────────── tables ─────────────────────────
def write_html(df: pd.DataFrame, out: str):
    df.to_html(out, index=False, border=0, justify="center")

def generate_expense_reports(tkr, *, rebuild_schema=False, conn=None):
    ensure(drop=rebuild_schema, conn=conn)
    store(tkr, mode="annual",   conn=conn)
    store(tkr, mode="quarterly", conn=conn)

    yearly_df = yearly(tkr)
    if yearly_df.empty:
        print(f"⛔ No data for {tkr}")
        return

    full = pd.concat([yearly_df, ttm(tkr)], ignore_index=True)
    full = full.loc[full["total_revenue"].notna() & (full["total_revenue"] != 0)]

    chart_abs(full, tkr)
    chart_pct(full, tkr)

    base = ["total_revenue","cost_of_revenue","research_and_development",
            "selling_and_marketing","general_and_admin","sga_combined"]
    cols = ["year_label"] + [c for c in base if c in full.columns]

    abs_df = full[cols].sort_values("year_label")
    abs_df = abs_df.drop(columns=[c for c in abs_df.columns[1:] if _all_nan_or_zero(abs_df[c])])
    abs_fmt = abs_df.copy()
    for c in abs_fmt.columns[1:]:
        abs_fmt[c] = abs_fmt[c].apply(_fmt_short)
    rename_abs = {
        "year_label":"Year","total_revenue":"Revenue ($)",
        "cost_of_revenue":"Cost of Revenue ($)","research_and_development":"R&D ($)",
        "selling_and_marketing":"Sales & Marketing ($)",
        "general_and_admin":"G&A ($)","sga_combined":"SG&A ($)"
    }
    write_html(abs_fmt.rename(columns={k:v for k,v in rename_abs.items() if k in abs_fmt}),
               os.path.join(OUTPUT_DIR,f"{tkr}_expense_absolute.html"))

    yoy = full[cols].sort_values("year_label")
    for c in cols[1:]:
        yoy[c] = (yoy[c].pct_change()
                  .replace([np.inf,-np.inf],np.nan)
                  .round(4)*100)
    yoy = yoy.drop(columns=[c for c in yoy.columns[1:] if yoy[c].notna().sum() == 0])
    yoy = yoy[yoy.iloc[:,1:].notna().any(axis=1)]
    rename_pct = {
        "year_label":"Year","total_revenue":"Revenue Change (%)",
        "cost_of_revenue":"Cost of Revenue Change (%)",
        "research_and_development":"R&D Change (%)",
        "selling_and_marketing":"Sales & Marketing Change (%)",
        "general_and_admin":"G&A Change (%)","sga_combined":"SG&A Change (%)"
    }
    write_html(yoy.rename(columns={k:v for k,v in rename_pct.items() if k in yoy}),
               os.path.join(OUTPUT_DIR,f"{tkr}_yoy_expense_change.html"))

    print(f"[{tkr}] ✔ charts & tables generated")

# ───────────────────────────── runner ─────────────────────────────
if __name__ == "__main__":
    generate_expense_reports("AAPL")

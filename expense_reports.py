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
    if pd.isna(x):
        return ""
    for div, suf in _SUFFIXES:
        if abs(x) >= div:
            return f"${x/div:.{d}f}{suf}"
    return f"${x:.{d}f}"

def _all_nan_or_zero(s: pd.Series) -> bool:
    return (s.replace(0, np.nan).notna().sum() == 0)

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
        base = [c for c in base if c[1] not in ("general_and_admin", "selling_and_marketing")]
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
    ax.legend(frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{tkr}_expenses_vs_revenue.png"))
    plt.close()

def chart_pct(df, tkr):
    """
    Stacked bars of expenses as % of revenue with:
      • dashed 100 % line
      • adaptive head-room
      • label color (white/black) chosen for contrast
    """
    f = df.sort_values("year_int")
    f = f[f["total_revenue"] != 0]
    xl = f["year_label"].tolist()

    cats = _cats(f, f["sga_combined"].notna().any())

    # % columns
    for _, col, _ in cats:
        f[col + "_pct"] = f[col] / f["total_revenue"] * 100

    def _text_color(hex_color: str) -> str:
        r, g, b = (int(hex_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
        return "white" if (0.299*r + 0.587*g + 0.114*b) < 140 else "black"

    fig, ax = plt.subplots(figsize=(11, 6))
    bottoms = np.zeros(len(f))

    for lbl, col, clr in cats:
        vals = f[col + "_pct"].fillna(0).values
        ax.bar(xl, vals, bottom=bottoms, color=clr, width=.6, label=lbl, zorder=2)
        for x, y0, v in zip(xl, bottoms, vals):
            if v > 4:
                ax.text(x, y0 + v/2, f"{v:.1f}%", ha="center", va="center",
                        fontsize=8, color=_text_color(clr))
        bottoms += vals

    ax.axhline(100, ls="--", lw=1, color="black", zorder=5)

    max_total = bottoms.max()
    ylim_max = 100 if max_total <= 100 else ((int(max_total/10) + 1) * 10) + 10
    ax.set_ylim(0, ylim_max)
    ax.set_yticks(np.arange(0, ylim_max+1, 10))

    ax.set_ylabel("Percent of Revenue")
    ax.set_title(f"Expenses as % of Revenue — {tkr}")
    ax.legend(frameon=False, ncol=2)
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, f"{tkr}_expenses_pct_of_rev.png")
    fig.savefig(out); plt.close()
    print(f"[{tkr}] expense-% chart saved → {out}")

# ───────────────── HTML helper ─────────────────
def write_html(df: pd.DataFrame, path: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            '<div class="scroll-table-wrapper">' +
            df.to_html(index=False, classes="expense-table", border=0, na_rep="") +
            '</div>'
        )

# ───────────────── main entry ─────────────────
def generate_expense_reports(tkr, *, rebuild_schema=False, conn=None):
    ensure(drop=rebuild_schema, conn=conn)
    store(tkr, mode="annual", conn=conn)
    store(tkr, mode="quarterly", conn=conn)

    yr = yearly(tkr)
    if yr.empty:
        print(f"⛔ No data for {tkr}")
        return
    full = pd.concat([yr, ttm(tkr)], ignore_index=True)
    chart_abs(full, tkr); chart_pct(full, tkr)

    base = ["total_revenue", "cost_of_revenue", "research_and_development",
            "selling_and_marketing", "general_and_admin", "sga_combined"]
    cols = ["year_label"] + [c for c in base if c in full.columns]

    # -------- absolute-$ table --------
    abs_df = full[cols].sort_values("year_label")
    abs_df = abs_df[abs_df["total_revenue"].notna() & (abs_df["total_revenue"] != 0)]
    abs_df = abs_df.drop(columns=[c for c in abs_df.columns[1:] if _all_nan_or_zero(abs_df[c])])

    abs_fmt = abs_df.copy()
    for c in abs_fmt.columns[1:]:
        abs_fmt[c] = abs_fmt[c].apply(_fmt_short)

    rename_abs = {
        "year_label": "Year", "total_revenue": "Revenue ($)",
        "cost_of_revenue": "Cost of Revenue ($)", "research_and_development": "R&D ($)",
        "selling_and_marketing": "Sales & Marketing ($)", "general_and_admin": "G&A ($)",
        "sga_combined": "SG&A ($)"
    }
    abs_fmt = abs_fmt.rename(columns={k:v for k,v in rename_abs.items() if k in abs_fmt.columns})
    write_html(abs_fmt, os.path.join(OUTPUT_DIR, f"{tkr}_expense_absolute.html"))

    # -------- YoY-% table --------
    yoy = full[cols].sort_values("year_label")
    yoy = yoy[yoy["total_revenue"].notna() & (yoy["total_revenue"] != 0)]
    for c in cols[1:]:
        yoy[c] = (yoy[c].pct_change().replace([np.inf,-np.inf], np.nan).round(4)*100)

    yoy = yoy.drop(columns=[c for c in yoy.columns[1:] if yoy[c].notna().sum() == 0])
    yoy = yoy[yoy.iloc[:,1:].notna().any(axis=1)]

    rename_pct = {
        "year_label": "Year", "total_revenue": "Revenue Change (%)",
        "cost_of_revenue": "Cost of Revenue Change (%)", "research_and_development": "R&D Change (%)",
        "selling_and_marketing": "Sales & Marketing Change (%)", "general_and_admin": "G&A Change (%)",
        "sga_combined": "SG&A Change (%)"
    }
    yoy = yoy.rename(columns={k:v for k,v in rename_pct.items() if k in yoy.columns})
    write_html(yoy, os.path.join(OUTPUT_DIR, f"{tkr}_yoy_expense_change.html"))

    print(f"[{tkr}] ✔ charts & tables generated")

if __name__ == "__main__":
    generate_expense_reports("AAPL")

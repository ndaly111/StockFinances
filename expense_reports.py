# ─────────────────────────────────────────────────────────────
#  expense_reports.py   ←  FULL REPLACEMENT
# ─────────────────────────────────────────────────────────────
from __future__ import annotations
import os, sqlite3, warnings
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib import colors as mcolors
import numpy as np, pandas as pd, yfinance as yf

from expense_labels import (
    COST_OF_REVENUE, RESEARCH_AND_DEVELOPMENT, SELLING_AND_MARKETING,
    GENERAL_AND_ADMIN, SGA_COMBINED, FACILITIES_DA, PERSONNEL_COSTS,
    INSURANCE_CLAIMS, OTHER_OPERATING,
)

DB_PATH   = "Stock Data.db"
OUT_DIR   = "charts"
os.makedirs(OUT_DIR, exist_ok=True)
__all__ = ["generate_expense_reports"]

warnings.filterwarnings("ignore", module="matplotlib.category")  # silence cat-unit spam

# ─────────────────────────── utils ───────────────────────────
_SUFFIXES = [(1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")]
def _fmt_short(x: float, d: int = 1) -> str:
    if pd.isna(x): return ""
    for div, suf in _SUFFIXES:
        if abs(x) >= div: return f"${x/div:.{d}f}{suf}"
    return f"${x:.{d}f}"

def _all_zero(col) -> bool:
    return (pd.to_numeric(col, errors="coerce").fillna(0) == 0).all()

def _rgb_or_hex(c):
    """Return r,g,b floats ∈ [0,1] for any Matplotlib-acceptable colour."""
    try:
        return mcolors.to_rgb(c)
    except ValueError:
        return (0,0,0)

# ────────────────────── DB schema / IO ───────────────────────
TABLES = ("IncomeStatement", "QuarterlyIncomeStatement")
SCHEMA = """
CREATE TABLE IF NOT EXISTS {n}(
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
    PRIMARY KEY(ticker, period_ending)
);"""

def ensure(drop=False, conn=None):
    own = conn is None
    if own: conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    for t in TABLES:
        if drop: cur.execute(f"DROP TABLE IF EXISTS {t}")
        cur.execute(SCHEMA.format(n=t))
    conn.commit()
    if own: conn.close()

def store(tkr, mode="annual", conn=None):
    own = conn is None
    if own: conn = sqlite3.connect(DB_PATH)
    fin = (
        yf.Ticker(tkr).financials.T if mode=="annual"
        else yf.Ticker(tkr).quarterly_financials.T
    )
    if fin.empty: return
    cur = conn.cursor()
    tbl = "IncomeStatement" if mode=="annual" else "QuarterlyIncomeStatement"
    for idx,row in fin.iterrows():
        pe = idx.to_pydatetime() if hasattr(idx,"to_pydatetime") else idx
        vals = (
            row.get("Total Revenue"),
            *[row.get(c) for c in (
                COST_OF_REVENUE, RESEARCH_AND_DEVELOPMENT, SELLING_AND_MARKETING,
                GENERAL_AND_ADMIN, SGA_COMBINED, FACILITIES_DA,
                PERSONNEL_COSTS, INSURANCE_CLAIMS, OTHER_OPERATING
            )]
        )
        cur.execute(
            f"INSERT OR REPLACE INTO {tbl} VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (tkr, pe, *[None if pd.isna(v) else float(v) for v in vals])
        )
    conn.commit()
    if own: conn.close()

# ───────────────────── pull yearly / TTM ─────────────────────
def yearly(tkr):
    df = pd.read_sql(f"SELECT * FROM IncomeStatement WHERE ticker='{tkr}'", sqlite3.connect(DB_PATH))
    if df.empty: return df
    df["period_ending"] = pd.to_datetime(df["period_ending"])
    df["year_int"] = df["period_ending"].dt.year
    g = df.groupby("year_int", as_index=False).sum(numeric_only=True)
    g["year_label"] = g["year_int"].astype(str)
    return g

def ttm(tkr):
    q = pd.read_sql(
        f"""SELECT * FROM QuarterlyIncomeStatement
            WHERE ticker='{tkr}' ORDER BY period_ending DESC""",
        sqlite3.connect(DB_PATH))
    if q.empty or len(q) < 4: return pd.DataFrame()
    q["period_ending"] = pd.to_datetime(q["period_ending"])
    recent = q.head(4).sort_values("period_ending")
    expect = pd.date_range(end=recent["period_ending"].max(), periods=4, freq="QE")
    if list(expect.to_period("Q")) != list(recent["period_ending"].dt.to_period("Q")):
        return pd.DataFrame()
    t = recent.drop(columns=["ticker","period_ending"]).sum().to_frame().T
    t.insert(0,"year_label","TTM"); t["year_int"]=np.nan
    return t

# ───────────────────────── charts ────────────────────────────
CATS = [
    ("Cost of Revenue","cost_of_revenue","#6d6d6d"),
    ("R&D","research_and_development","tab:blue"),
    ("G&A","general_and_admin","#ffb3c6"),
    ("Selling & Marketing","selling_and_marketing","#ffc6e2"),
    ("SG&A","sga_combined","#c2a5ff"),
    ("Facilities / D&A","facilities_da","tab:orange"),
]
def _pick_cats(df):
    cats = CATS.copy()
    if df["sga_combined"].notna().any():
        cats = [c for c in cats if c[1] not in ("general_and_admin","selling_and_marketing")]
    return [c for c in cats if c[1] in df.columns]

def chart_abs(df, tkr):
    f = df.sort_values("year_int")
    x = f["year_label"].tolist()
    cats = _pick_cats(f)
    fig, ax = plt.subplots(figsize=(11,6))
    bot = np.zeros(len(f))
    for lbl,col,clr in cats:
        vals = pd.to_numeric(f[col], errors="coerce").fillna(0).to_numpy(float)
        ax.bar(x, vals, bottom=bot, color=clr, width=.6, label=lbl)
        bot += vals
    ax.plot(x, pd.to_numeric(f["total_revenue"], errors="coerce").to_numpy(float),
            "k-o", lw=2, label="Revenue")
    ax.set_ylim(0, max(bot.max(), f["total_revenue"].max())*1.1)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_: _fmt_short(v)))
    ax.set_title(f"Revenue vs Operating Expenses — {tkr}")
    ax.legend(frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR,f"{tkr}_expenses_vs_revenue.png"))
    plt.close()

def chart_pct(df, tkr):
    f = df.sort_values("year_int")
    f = f.loc[f["total_revenue"].ne(0)]
    x = f["year_label"].tolist()
    cats = _pick_cats(f)
    for _,col,_ in cats:
        f[col+"_pct"] = pd.to_numeric(f[col], errors="coerce")/f["total_revenue"]*100
    fig, ax = plt.subplots(figsize=(11,4))
    bot = np.zeros(len(f))
    for lbl,col,clr in cats:
        vals = f[col+"_pct"].fillna(0).to_numpy(float)
        ax.bar(x, vals, bottom=bot, color=clr, width=.6, zorder=2)
        for xi,y0,v in zip(x, bot, vals):
            if v>4:
                r,g,b = _rgb_or_hex(clr)
                txt_col = "white" if (0.299*r+0.587*g+0.114*b)<0.6 else "black"
                ax.text(xi, y0+v/2, f"{v:.1f}%", ha="center", va="center",
                        fontsize=8, color=txt_col)
        bot += vals
    ax.axhline(100, ls="--", lw=1, color="black")
    ylim = np.ceil((bot.max()*1.08)/10)*10
    ax.set_ylim(0, ylim)
    ax.set_yticks(np.arange(0, ylim+1, 10))
    ax.set_ylabel("% of Revenue")
    ax.set_title(f"Expenses as % of Revenue — {tkr}")
    ax.legend([c[0] for c in cats], bbox_to_anchor=(1.01,0.5),
              loc="center left", frameon=False)
    plt.tight_layout()
    out = os.path.join(OUT_DIR,f"{tkr}_expenses_pct_of_rev.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()

# ───────────────────────── tables ────────────────────────────
def write_html(df, path): df.to_html(path, index=False, border=0, justify="center")

def generate_expense_reports(tkr, rebuild_schema=False, conn=None):
    ensure(drop=rebuild_schema, conn=conn)
    store(tkr, "annual", conn=conn)
    store(tkr, "quarterly", conn=conn)

    df = pd.concat([yearly(tkr), ttm(tkr)], ignore_index=True)
    if df.empty: return
    df = df.loc[df["total_revenue"].notna() & df["total_revenue"].ne(0)]

    chart_abs(df, tkr)
    chart_pct(df, tkr)

    base = ["total_revenue","cost_of_revenue","research_and_development",
            "selling_and_marketing","general_and_admin","sga_combined"]
    cols = ["year_label"] + [c for c in base if c in df.columns]

    abs_tbl = df[cols].sort_values("year_label").copy()
    abs_tbl.drop(columns=[c for c in abs_tbl.columns[1:] if _all_zero(abs_tbl[c])], inplace=True)
    for c in abs_tbl.columns[1:]:
        abs_tbl[c]=abs_tbl[c].apply(_fmt_short)
    rename = {
        "year_label":"Year",
        "total_revenue":"Revenue ($)",
        "cost_of_revenue":"Cost of Revenue ($)",
        "research_and_development":"R&D ($)",
        "selling_and_marketing":"Sales & Marketing ($)",
        "general_and_admin":"G&A ($)",
        "sga_combined":"SG&A ($)"
    }
    abs_tbl.rename(columns=rename, inplace=True)
    write_html(abs_tbl, os.path.join(OUT_DIR,f"{tkr}_expense_absolute.html"))

    pct_tbl = df[cols].sort_values("year_label").copy()
    for c in cols[1:]:
        pct_tbl[c] = pd.to_numeric(pct_tbl[c], errors="coerce").pct_change()*100
    pct_tbl.replace([np.inf,-np.inf], np.nan, inplace=True)
    pct_tbl.drop(columns=[c for c in pct_tbl.columns[1:] if pct_tbl[c].notna().sum()==0], inplace=True)
    pct_tbl = pct_tbl[pct_tbl.iloc[:,1:].notna().any(axis=1)]
    rename_pct = {
        "year_label":"Year","total_revenue":"Revenue Δ (%)",
        "cost_of_revenue":"Cost of Revenue Δ (%)",
        "research_and_development":"R&D Δ (%)",
        "selling_and_marketing":"Sales & Marketing Δ (%)",
        "general_and_admin":"G&A Δ (%)",
        "sga_combined":"SG&A Δ (%)"
    }
    pct_tbl.rename(columns=rename_pct, inplace=True)
    write_html(pct_tbl, os.path.join(OUT_DIR,f"{tkr}_yoy_expense_change.html"))

    print(f"[{tkr}] ✔ charts & tables generated")

# stand-alone
if __name__=="__main__":
    generate_expense_reports("AAPL")

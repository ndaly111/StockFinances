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

# ─────────────────────────────────────────────────────────────
#  expense label aliases (keep this file next to expense_reports.py)
# ─────────────────────────────────────────────────────────────
from expense_labels import (
    COST_OF_REVENUE, RESEARCH_AND_DEVELOPMENT, SELLING_AND_MARKETING,
    GENERAL_AND_ADMIN, SGA_COMBINED, FACILITIES_DA, PERSONNEL_COSTS,
    INSURANCE_CLAIMS, OTHER_OPERATING,
)

DB_PATH, OUTPUT_DIR = "Stock Data.db", "charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)
__all__ = ["generate_expense_reports"]

# ─────────────── category meta  (db name, aliases, colour, pretty) ─────────────
CATS = [
    ("cost_of_revenue",          COST_OF_REVENUE,          "#6d6d6d", "Cost of Revenue"),
    ("research_and_development", RESEARCH_AND_DEVELOPMENT, "blue",    "R&D"),
    ("selling_and_marketing",    SELLING_AND_MARKETING,    "#ffc6e2", "Selling & Marketing"),
    ("general_and_admin",        GENERAL_AND_ADMIN,        "#ffb3c6", "G&A"),
    ("sga_combined",             SGA_COMBINED,             "#c2a5ff", "SG&A"),
    ("facilities_da",            FACILITIES_DA,            "orange",  "Facilities / D&A"),
    ("personnel_costs",          PERSONNEL_COSTS,          "green",   "Personnel Costs"),
    ("insurance_claims",         INSURANCE_CLAIMS,         "brown",   "Insurance / Claims"),
    ("other_operating",          OTHER_OPERATING,          "#999999", "Other Operating"),
]
CAT_NAMES   = [c[0] for c in CATS]          # db / column keys
CAT_COLOURS = {c[0]: c[2] for c in CATS}
CAT_PRETTY  = {c[0]: c[3] for c in CATS}

# ─────────────── helpers ───────────────
_SUFFIXES = [(1e12,"T"),(1e9,"B"),(1e6,"M"),(1e3,"K")]
def _fmt_short(x:float,d:int=1)->str:
    if pd.isna(x): return ""
    for div,suf in _SUFFIXES:
        if abs(x)>=div: return f"${x/div:.{d}f}{suf}"
    return f"${x:.{d}f}"

def _all_nan_or_zero(col:pd.Series)->bool:
    return (col.replace(0,np.nan).notna().sum()==0)

def _clean(v):
    if pd.isna(v): return None
    return v.isoformat() if isinstance(v,(pd.Timestamp,datetime)) else v

def _pick_any(row:pd.Series, aliases:list[str]):
    for k in row.index:
        if pd.notna(row[k]) and any(alias.lower() in k.lower() for alias in aliases):
            return row[k]
    return None

# ─────────────── DB schema / IO ───────────────
TABLES = ("IncomeStatement","QuarterlyIncomeStatement")
_BASE_SCHEMA = "ticker TEXT, period_ending TEXT, total_revenue REAL"
_MORE = ", ".join([f"{c} REAL" for c in CAT_NAMES])
SCHEMA = f"({ _BASE_SCHEMA }, {_MORE}, PRIMARY KEY(ticker,period_ending))"

def ensure(drop=False,*,conn=None):
    own = conn is None
    if own: conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    for t in TABLES:
        if drop: cur.execute(f"DROP TABLE IF EXISTS {t}")
        cur.execute(f"CREATE TABLE IF NOT EXISTS {t} {SCHEMA};")
    conn.commit(); cur.close()
    if own: conn.close()

def _row_to_values(row:pd.Series):
    vals = []
    for _, aliases, _, _ in CATS:
        vals.append(_pick_any(row, aliases))
    return vals

def store(tkr,*,mode="annual",conn=None):
    raw = (yf.Ticker(tkr).financials.transpose() if mode=="annual"
           else yf.Ticker(tkr).quarterly_financials.transpose())
    if raw.empty: return
    own = conn is None
    if own: conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    table = "IncomeStatement" if mode=="annual" else "QuarterlyIncomeStatement"
    for idx,row in raw.iterrows():
        pe  = idx.to_pydatetime() if isinstance(idx,pd.Timestamp) else idx
        cur.execute(f"INSERT OR REPLACE INTO {table} VALUES ({','.join('?'*(3+len(CAT_NAMES)))})",
            [tkr,_clean(pe),_clean(row.get("Total Revenue"))]+[_clean(v) for v in _row_to_values(row)])
    conn.commit(); cur.close()
    if own: conn.close()

# ─────────────── pull annual & TTM ───────────────
def _annual(tkr):
    conn=sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM IncomeStatement WHERE ticker=?",
                           conn,params=(tkr,))
    conn.close()
    if df.empty: return df
    df["period_ending"] = pd.to_datetime(df["period_ending"])
    df["year_int"] = df["period_ending"].dt.year
    g = df.groupby("year_int",as_index=False).sum(numeric_only=True)
    g["year_label"] = g["year_int"].astype(str)
    return g

def _ttm(tkr):
    conn=sqlite3.connect(DB_PATH)
    q = pd.read_sql_query(
        "SELECT * FROM QuarterlyIncomeStatement WHERE ticker=? ORDER BY period_ending DESC",
        conn,params=(tkr,))
    conn.close()
    if q.empty: return q
    q["period_ending"]=pd.to_datetime(q["period_ending"])
    recent=q.head(4).sort_values("period_ending")
    if len(recent)<4: return pd.DataFrame()
    expect=pd.date_range(end=recent["period_ending"].max(),periods=4,freq="Q")
    if list(expect.to_period("Q"))!=list(recent["period_ending"].dt.to_period("Q")):
        return pd.DataFrame()
    ttm=recent.drop(columns=["ticker","period_ending"]).sum().to_frame().T
    ttm.insert(0,"year_label","TTM"); ttm["year_int"]=np.nan
    return ttm

# ─────────────── chart helpers ───────────────
def _cat_order(df:pd.DataFrame):
    """Return (label,col,color) tuples, filtering unusable cols & SG&A logic."""
    has_sga = df["sga_combined"].notna().any()
    ordered=[]
    for col,_,color,pretty in CATS:
        if has_sga and col in ("general_and_admin","selling_and_marketing"):
            continue
        if col in df.columns and not _all_nan_or_zero(df[col]):
            ordered.append((pretty,col,color))
    return ordered

def _chart_abs(df,tkr):
    f=df.sort_values("year_int"); xl=f["year_label"].tolist()
    cats=_cat_order(f)
    fig,ax=plt.subplots(figsize=(11,6)); bot=np.zeros(len(f))
    for lbl,col,clr in cats:
        v=f[col].fillna(0).values
        ax.bar(xl,v,bottom=bot,color=clr,width=.6,label=lbl)
        bot+=v
    ax.plot(xl,f["total_revenue"],"k-o",lw=2,label="Revenue")
    ax.set_ylim(0,max(bot.max(),f["total_revenue"].max())*1.1)
    ax.set_title(f"Revenue vs Operating Expenses — {tkr}")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x,p:_fmt_short(x)))
    ax.legend(frameon=False,ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,f"{tkr}_expenses_vs_revenue.png"))
    plt.close()

def _chart_pct(df,tkr):
    f=(df.sort_values("year_int").loc[lambda d:d["total_revenue"]!=0])
    xl=f["year_label"].tolist()
    cats=_cat_order(f)
    for _,col,_ in cats:
        f[col+"_pct"]=f[col]/f["total_revenue"]*100
    fig,ax=plt.subplots(figsize=(11,4)); bot=np.zeros(len(f))
    for lbl,col,clr in cats:
        vals=f[col+"_pct"].fillna(0).values
        ax.bar(xl,vals,bottom=bot,color=clr,width=.6)
        bot+=vals
    ax.axhline(100,ls="--",color="black")
    ax.set_ylim(0,np.ceil((bot.max()+8)/10)*10)
    ax.set_yticks(np.arange(0,ax.get_ylim()[1]+1,10))
    ax.set_ylabel("Percent of Revenue")
    ax.set_title(f"Expenses as % of Revenue — {tkr}")
    ax.legend([c[0] for c in cats],bbox_to_anchor=(1.01,0.5),
              loc="center left",frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,f"{tkr}_expenses_pct_of_rev.png"),
                dpi=120,bbox_inches="tight")
    plt.close()

def _write_html(df:pd.DataFrame,path:str): df.to_html(path,index=False,border=0,justify="center")

# ─────────────── main entry ───────────────
def generate_expense_reports(tkr,*,rebuild_schema=False,conn=None):
    ensure(drop=rebuild_schema,conn=conn)
    store(tkr,mode="annual",conn=conn)
    store(tkr,mode="quarterly",conn=conn)

    annual=_annual(tkr)
    if annual.empty:
        print(f"⛔ No data for {tkr}"); return
    full=pd.concat([annual,_ttm(tkr)],ignore_index=True)

    # ----- charts -----
    _chart_abs(full,tkr)
    _chart_pct(full,tkr)

    # ----- tables -----
    cols=["year_label","total_revenue"]+[c for c in CAT_NAMES if c in full.columns]
    df=(full[cols].sort_values("year_label")
          .loc[lambda d:(d["total_revenue"].notna())&(d["total_revenue"]!=0)])
    # drop useless categories
    df=df.drop(columns=[c for c in df.columns[2:] if _all_nan_or_zero(df[c])])

    # absolute $
    fmt=df.copy()
    for c in fmt.columns[1:]: fmt[c]=fmt[c].apply(_fmt_short)
    rename={"year_label":"Year","total_revenue":"Revenue ($)"}
    rename.update({k:CAT_PRETTY[k]+" ($)" for k in df.columns if k in CAT_PRETTY})
    fmt=fmt.rename(columns=rename)
    _write_html(fmt,os.path.join(OUTPUT_DIR,f"{tkr}_expense_absolute.html"))

    # YoY %
    yoy=df.copy()
    for c in yoy.columns[2:]:
        yoy[c]=(yoy[c].pct_change().replace([np.inf,-np.inf],np.nan)*100)
    yoy=yoy.loc[yoy.iloc[:,2:].notna().any(axis=1)]
    rename_pct={"year_label":"Year","total_revenue":"Revenue Change (%)"}
    rename_pct.update({k:CAT_PRETTY[k]+" Change (%)" for k in yoy.columns if k in CAT_PRETTY})
    yoy=yoy.rename(columns=rename_pct).round(1)
    _write_html(yoy,os.path.join(OUTPUT_DIR,f"{tkr}_yoy_expense_change.html"))

    print(f"[{tkr}] ✔ charts & tables generated")

# ──────────────────────────────────────────
if __name__ == "__main__":
    generate_expense_reports("AAPL")

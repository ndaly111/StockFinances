"""
expense_reports.py
-------------------------------------------------------------------------------
Per-ticker outputs:
  1) Revenue-vs-stacked-expense chart ($)
  2) Expenses as %-of-revenue chart
  3) YoY expense-change HTML table
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

# ── helpers ─────────────────────────────────────────────────────────────
def clean_value(v):
    if pd.isna(v): return None
    return v.isoformat() if isinstance(v,(pd.Timestamp,datetime)) else v

def extract_expenses(row: pd.Series):
    def pick(labels):
        for k in row.index:
            if pd.notna(row[k]) and any(lbl.lower() in k.lower() for lbl in labels):
                return row[k]
        return None
    return (
        pick(COST_OF_REVENUE), pick(RESEARCH_AND_DEVELOPMENT),
        pick(SELLING_AND_MARKETING), pick(GENERAL_AND_ADMIN),
        pick(SGA_COMBINED), pick(FACILITIES_DA),
        pick(PERSONNEL_COSTS), pick(INSURANCE_CLAIMS), pick(OTHER_OPERATING),
    )

# ── DB schema helpers ───────────────────────────────────────────────────
TABLES = ("IncomeStatement","QuarterlyIncomeStatement")
SCHEMA = """
CREATE TABLE IF NOT EXISTS {n}(
  ticker TEXT, period_ending TEXT, total_revenue REAL,
  cost_of_revenue REAL, research_and_development REAL,
  selling_and_marketing REAL, general_and_admin REAL, sga_combined REAL,
  facilities_da REAL, personnel_costs REAL, insurance_claims REAL,
  other_operating REAL, PRIMARY KEY(ticker,period_ending));
"""
def ensure_tables(drop=False,*,conn=None):
    own = conn is None
    if own: conn=sqlite3.connect(DB_PATH)
    cur=conn.cursor()
    for t in TABLES:
        if drop: cur.execute(f"DROP TABLE IF EXISTS {t}")
        cur.execute(SCHEMA.format(n=t))
    conn.commit(); cur.close()
    if own: conn.close()

# ── ingest ──────────────────────────────────────────────────────────────
def store_data(ticker,*,mode="annual",conn=None):
    df = (yf.Ticker(ticker).financials.transpose() if mode=="annual"
          else yf.Ticker(ticker).quarterly_financials.transpose())
    if df.empty: return
    own=conn is None
    if own: conn=sqlite3.connect(DB_PATH)
    cur=conn.cursor(); tbl="IncomeStatement" if mode=="annual" else "QuarterlyIncomeStatement"
    for idx,row in df.iterrows():
        pe=idx.to_pydatetime() if isinstance(idx,pd.Timestamp) else idx
        cost,rnd,mkt,adm,sga,fda,ppl,ins,oth=extract_expenses(row)
        cur.execute(f"INSERT OR REPLACE INTO {tbl} VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (ticker,clean_value(pe),clean_value(row.get("Total Revenue")),
             clean_value(cost),clean_value(rnd),clean_value(mkt),clean_value(adm),
             clean_value(sga),clean_value(fda),clean_value(ppl),
             clean_value(ins),clean_value(oth)))
    conn.commit(); cur.close()
    if own: conn.close()

# ── fetch helpers ────────────────────────────────────────────────────────
def yearly_df(tkr):
    conn=sqlite3.connect(DB_PATH)
    df=pd.read_sql_query("SELECT * FROM IncomeStatement WHERE ticker=?",
                         conn,params=(tkr,))
    conn.close()
    if df.empty: return df
    df["period_ending"]=pd.to_datetime(df["period_ending"])
    df["year_int"]=df["period_ending"].dt.year
    g=df.groupby("year_int",as_index=False).sum(numeric_only=True)
    g["year_label"]=g["year_int"].astype(str)
    return g

def ttm_df(tkr):
    conn=sqlite3.connect(DB_PATH)
    q=pd.read_sql_query(
        "SELECT * FROM QuarterlyIncomeStatement WHERE ticker=? ORDER BY period_ending DESC",
        conn,params=(tkr,))
    conn.close()
    if q.empty: return q
    q["period_ending"]=pd.to_datetime(q["period_ending"])
    recent=q.head(4).sort_values("period_ending")
    if len(recent)<4: return pd.DataFrame()
    exp=pd.date_range(end=recent["period_ending"].max(),periods=4,freq="Q")
    if list(exp.to_period("Q"))!=list(recent["period_ending"].dt.to_period("Q")):
        return pd.DataFrame()
    ttm=recent.drop(columns=["ticker","period_ending"]).sum().to_frame().T
    ttm.insert(0,"year_label","TTM"); ttm["year_int"]=np.nan
    return ttm

# ── chart helpers ───────────────────────────────────────────────────────
def _fmt(x,_p=None,d=1):
    if pd.isna(x): return "$0"
    n=abs(x)
    return f"${x/1e12:.{d}f} T" if n>=1e12 else f"${x/1e9:.{d}f} B" if n>=1e9 \
        else f"${x/1e6:.{d}f} M" if n>=1e6 else f"${x/1e3:.{d}f} K" if n>=1e3 \
        else f"${x:.{d}f}"

def _available_cats(df,combo):
    cand=[("Cost of Revenue","cost_of_revenue","#6d6d6d"),
          ("R&D","research_and_development","blue"),
          ("G&A","general_and_admin","#ffb3c6"),
          ("Selling & Marketing","selling_and_marketing","#ffc6e2"),
          ("SG&A","sga_combined","#c2a5ff"),
          ("Facilities / D&A","facilities_da","orange")]
    if combo:                                         # company has sga_combined
        cand=[c for c in cand if c[1] not in ("general_and_admin","selling_and_marketing")]
    return [c for c in cand if c[1] in df.columns]

def chart_abs(full,tkr):
    f=full.sort_values("year_int").copy(); xl=f["year_label"].tolist()
    cats=_available_cats(f,f["sga_combined"].notna().any())
    fig,ax=plt.subplots(figsize=(11,6)); bottoms=np.zeros(len(f))
    for lbl,col,clr in cats:
        vals=f[col].fillna(0).values
        ax.bar(xl,vals,bottom=bottoms,color=clr,label=lbl,width=.6)
        bottoms+=vals
    ax.plot(xl,f["total_revenue"],"k-o",lw=2,label="Revenue")
    ax.set_ylim(0,max(bottoms.max(),f["total_revenue"].max())*1.1)
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt))
    ax.set_title(f"Revenue vs Operating Expenses — {tkr}")
    ax.legend(frameon=False,ncol=2); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,f"{tkr}_expenses_vs_revenue.png")); plt.close()

def chart_pct(full,tkr):
    f=full.sort_values("year_int").copy()
    f=f[f["total_revenue"]!=0]; xl=f["year_label"].tolist()
    cats=_available_cats(f,f["sga_combined"].notna().any())
    for _,c,_ in cats: f[c+"_pct"]=(f[c]/f["total_revenue"]*100)
    fig,ax=plt.subplots(figsize=(11,6)); bottoms=np.zeros(len(f))
    for lbl,c,clr in cats:
        v=f[c+"_pct"].fillna(0).values
        ax.bar(xl,v,bottom=bottoms,color=clr,label=lbl,width=.6)
        for x,y0,val in zip(xl,bottoms,v):
            if val>4: ax.text(x,y0+val/2,f"{val:.1f}%",ha="center",va="center",color="white",fontsize=8)
        bottoms+=v
    ax.axhline(100,ls="--",lw=1,c="black"); ylim=max(110,(int(bottoms.max()/10)+2)*10)
    ax.set_ylim(0,ylim); ax.set_yticks(np.arange(0,ylim+1,10))
    ax.set_title(f"Expenses as % of Revenue — {tkr}"); ax.legend(frameon=False,ncol=2)
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR,f"{tkr}_expenses_pct_of_rev.png")); plt.close()

# ── main ────────────────────────────────────────────────────────────────
def generate_expense_reports(tkr,*,rebuild_schema=False,conn=None):
    ensure_tables(drop=rebuild_schema,conn=conn)
    store_data(tkr,mode="annual",conn=conn); store_data(tkr,mode="quarterly",conn=conn)
    yr=yearly_df(tkr)
    if yr.empty: print(f"⛔ No data for {tkr}"); return
    full=pd.concat([yr,ttm_df(tkr)],ignore_index=True)
    chart_abs(full,tkr); chart_pct(full,tkr)

    # YoY table
    base=["total_revenue","cost_of_revenue","research_and_development",
          "selling_and_marketing","general_and_admin","sga_combined"]
    cols=["year_label"]+[c for c in base if c in full.columns]
    yoy=full[cols].sort_values("year_label")
    yoy=yoy[yoy["total_revenue"].notna()&(yoy["total_revenue"]!=0)]
    for c in cols[1:]:
        yoy[c]=(yoy[c].pct_change().replace([np.inf,-np.inf],np.nan).round(4)*100)
    rename={"year_label":"Year","total_revenue":"Revenue Change (%)",
            "cost_of_revenue":"Cost of Revenue Change (%)",
            "research_and_development":"R&D Change (%)",
            "selling_and_marketing":"Sales & Marketing Change (%)",
            "general_and_admin":"G&A Change (%)",
            "sga_combined":"SG&A Change (%)"}
    yoy=yoy.rename(columns={k:v for k,v in rename.items() if k in yoy.columns})
    with open(os.path.join(OUTPUT_DIR,f"{tkr}_yoy_expense_change.html"),"w",encoding="utf-8") as f:
        f.write('<div class="scroll-table-wrapper">'+yoy.to_html(index=False,classes="expense-table",border=0)+'</div>')
    print(f"[{tkr}] charts & YoY table generated")

if __name__=="__main__":
    generate_expense_reports("AAPL")

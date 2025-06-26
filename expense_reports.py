"""
expense_reports.py
-------------------------------------------------------------------------------
Generates for each ticker:
  1) Revenue vs stacked-expense chart (absolute $)
  2) Expenses as % of revenue chart
  3) YoY expense-change HTML table
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

DB_PATH, OUTPUT_DIR = "Stock Data.db", "charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)
__all__ = ["generate_expense_reports"]

# ── helpers ─────────────────────────────────────────────────────────────
def clean_value(val):
    if pd.isna(val):                           return None
    if isinstance(val, (pd.Timestamp, datetime)): return val.isoformat()
    return val

def extract_expenses(row: pd.Series):
    def match_any(labels):
        for key in row.index:
            for lbl in labels:
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

# ── schema helpers ──────────────────────────────────────────────────────
TABLES = ("IncomeStatement", "QuarterlyIncomeStatement")
TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS {name} (
  ticker TEXT, period_ending TEXT,
  total_revenue REAL, cost_of_revenue REAL, research_and_development REAL,
  selling_and_marketing REAL, general_and_admin REAL, sga_combined REAL,
  facilities_da REAL, personnel_costs REAL, insurance_claims REAL,
  other_operating REAL, PRIMARY KEY (ticker, period_ending)
);
"""

def ensure_tables(*, drop=False, conn=None):
    own = conn is None
    if own: conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    for tbl in TABLES:
        if drop: cur.execute(f"DROP TABLE IF EXISTS {tbl}")
        cur.execute(TABLE_SCHEMA.format(name=tbl))
    conn.commit();  cur.close()
    if own: conn.close()

# ── ingest ──────────────────────────────────────────────────────────────
def store_data(ticker, *, mode="annual", conn=None):
    raw = yf.Ticker(ticker).financials.transpose() if mode=="annual" \
          else yf.Ticker(ticker).quarterly_financials.transpose()
    if raw.empty: return

    own = conn is None
    if own: conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    tbl = "IncomeStatement" if mode=="annual" else "QuarterlyIncomeStatement"
    for idx,row in raw.iterrows():
        pe = idx.to_pydatetime() if isinstance(idx,pd.Timestamp) else idx
        cost,rnd,mkt,adm,sga,fda,ppl,ins,oth = extract_expenses(row)
        cur.execute(f"""INSERT OR REPLACE INTO {tbl} VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (ticker,clean_value(pe),clean_value(row.get("Total Revenue")),
             clean_value(cost),clean_value(rnd),clean_value(mkt),
             clean_value(adm),clean_value(sga),clean_value(fda),
             clean_value(ppl),clean_value(ins),clean_value(oth)))
    conn.commit();  cur.close()
    if own: conn.close()

# ── fetch helpers ────────────────────────────────────────────────────────
def fetch_yearly_data(ticker):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM IncomeStatement WHERE ticker=?", conn, params=(ticker,))
    conn.close()
    if df.empty: return pd.DataFrame()
    df["period_ending"] = pd.to_datetime(df["period_ending"])
    df["year_int"]      = df["period_ending"].dt.year
    grouped = df.groupby("year_int",as_index=False).sum(numeric_only=True)
    grouped["year_label"] = grouped["year_int"].astype(str)
    return grouped

def fetch_ttm_data(ticker):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""SELECT * FROM QuarterlyIncomeStatement
                              WHERE ticker=? ORDER BY period_ending DESC""",
                           conn,params=(ticker,))
    conn.close()
    if df.empty: return pd.DataFrame()
    df["period_ending"]=pd.to_datetime(df["period_ending"])
    recent=df.head(4).sort_values("period_ending")
    if len(recent)<4: return pd.DataFrame()
    exp=pd.date_range(end=recent["period_ending"].max(),periods=4,freq="Q")
    if list(exp.to_period("Q"))!=list(recent["period_ending"].dt.to_period("Q")): return pd.DataFrame()
    ttm=recent.drop(columns=["ticker","period_ending"]).sum().to_frame().T
    ttm.insert(0,"year_label","TTM"); ttm["year_int"]=np.nan
    return ttm

# ── plotting ────────────────────────────────────────────────────────────
def _fmt_short(x, _p=None, d=1):
    if pd.isna(x): return "$0"
    n=abs(x)
    return f"${x/1e12:.{d}f} T" if n>=1e12 else f"${x/1e9:.{d}f} B" if n>=1e9 \
        else f"${x/1e6:.{d}f} M" if n>=1e6 else f"${x/1e3:.{d}f} K" if n>=1e3 \
        else f"${x:.{d}f}"

def plot_expense_charts(full,ticker):
    full=full.sort_values("year_int")
    xl=full["year_label"].tolist()
    use_comb=full["sga_combined"].notna().any()
    cats=[("Cost of Revenue","cost_of_revenue","#6d6d6d"),
          ("R&D","research_and_development","blue"),
          ("G&A","general_and_admin","#ffb3c6"),
          ("Selling & Marketing","selling_and_marketing","#ffc6e2"),
          ("SG&A","sga_combined","#c2a5ff"),
          ("Facilities / D&A","facilities_da","orange")]
    if use_comb: cats=[c for c in cats if c[1] not in ("general_and_admin","selling_and_marketing")]
    fig,ax=plt.subplots(figsize=(11,6)); bottoms=np.zeros(len(full))
    for lbl,col,clr in cats:
        v=full[col].fillna(0).values; ax.bar(xl,v,bottom=bottoms,label=lbl,color=clr,width=.6); bottoms+=v
    ax.plot(xl,full["total_revenue"].values,"k-o",lw=2,label="Revenue")
    ax.set_ylim(0,max(bottoms.max(),full["total_revenue"].max())*1.1)
    ax.set_title(f"Revenue vs Operating Expenses — {ticker}")
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_short))
    ax.legend(frameon=False,ncol=2); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,f"{ticker}_expenses_vs_revenue.png")); plt.close()

def plot_expense_percent_chart(full,ticker):
    full=full.sort_values("year_int"); full=full[full["total_revenue"]!=0]
    xl=full["year_label"].tolist()
    use_comb=full["sga_combined"].notna().any()
    cats=[("Cost of Revenue","cost_of_revenue","#6d6d6d"),
          ("R&D","research_and_development","blue"),
          ("G&A","general_and_admin","#ffb3c6"),
          ("Selling & Marketing","selling_and_marketing","#ffc6e2"),
          ("SG&A","sga_combined","#c2a5ff"),
          ("Facilities / D&A","facilities_da","orange")]
    if use_comb: cats=[c for c in cats if c[1] not in ("general_and_admin","selling_and_marketing")]
    for _,col,_ in cats:
        full[col+"_pct"]=(full[col]/full["total_revenue"]*100).where(full["total_revenue"]!=0)
    fig,ax=plt.subplots(figsize=(11,6)); bottoms=np.zeros(len(full))
    for lbl,col,clr in cats:
        v=full[col+"_pct"].fillna(0).values
        ax.bar(xl,v,bottom=bottoms,label=lbl,color=clr,width=.6)
        for x,y0,val in zip(xl,bottoms,v):
            if val>4: ax.text(x,y0+val/2,f"{val:.1f}%",ha="center",va="center",fontsize=8,color="white")
        bottoms+=v
    ax.axhline(100,ls="--",lw=1,c="black",zorder=5)
    ylim=max(110,(int(bottoms.max()/10)+2)*10); ax.set_ylim(0,ylim); ax.set_yticks(np.arange(0,ylim+1,10))
    ax.set_title(f"Expenses as % of Revenue — {ticker}"); ax.legend(frameon=False,ncol=2)
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR,f"{ticker}_expenses_pct_of_rev.png")); plt.close()

# ── main entry ───────────────────────────────────────────────────────────
def generate_expense_reports(ticker,*,rebuild_schema=False,conn=None):
    ensure_tables(drop=rebuild_schema,conn=conn)
    store_data(ticker,mode="annual",conn=conn); store_data(ticker,mode="quarterly",conn=conn)
    yearly=fetch_yearly_data(ticker)
    if yearly.empty: print(f"⛔ No data for {ticker}"); return
    full=pd.concat([yearly,fetch_ttm_data(ticker)],ignore_index=True)
    plot_expense_charts(full,ticker); plot_expense_percent_chart(full,ticker)

    # dynamic column set for YoY table
    base_cols=["total_revenue","cost_of_revenue","research_and_development",
               "selling_and_marketing","general_and_admin","sga_combined"]
    cols=["year_label"]+[c for c in base_cols if c in full.columns]
    yoy=full[cols].sort_values("year_label")
    yoy=yoy[yoy["total_revenue"].notna()&(yoy["total_revenue"]!=0)]
    for col in cols[1:]:
        yoy[col]=(yoy[col].pct_change().replace([np.inf,-np.inf],np.nan).round(4)*100)
    rename={
        "year_label":"Year","total_revenue":"Revenue Change (%)",
        "cost_of_revenue":"Cost of Revenue Change (%)",
        "research_and_development":"R&D Change (%)",
        "selling_and_marketing":"Sales & Marketing Change (%)",
        "general_and_admin":"G&A Change (%)",
        "sga_combined":"SG&A Change (%)"
    }
    yoy=yoy.rename(columns={k:v for k,v in rename.items() if k in yoy.columns})
    html=os.path.join(OUTPUT_DIR,f"{ticker}_yoy_expense_change.html")
    with open(html,"w",encoding="utf-8") as f:
        f.write('<div class="scroll-table-wrapper">'+
                yoy.to_html(index=False,classes="expense-table",border=0)+
                '</div>')
    print(f"[{ticker}] YoY expense-table → {html}")

if __name__=="__main__":
    generate_expense_reports("AAPL")

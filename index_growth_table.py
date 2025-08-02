#!/usr/bin/env python3
# index_growth_table.py  –  Plotly panels w/ tables  (2025-08-02 rev M)
# ────────────────────────────────────────────────────────────────────
# • Logs implied-growth, P/E, EPS and 10-yr yield
# • Creates three interactive Plotly-JS panels per index
#     1) Implied Growth      2) P/E ratio
#     3) EPS (left) vs 10-yr yield % (right)
#   Each panel HTML already contains the same blue-framed summary table.
# • Writes a static overview table (unchanged)
# ────────────────────────────────────────────────────────────────────

import os, sqlite3, numpy as np, pandas as pd
from datetime import datetime
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─── Config ───────────────────────────────────────────────
DB_PATH, CHART_DIR = "Stock Data.db", "charts"
IDXES = ["SPY", "QQQ"]
FALLBACK_YIELD = 0.045  # 4.5 %
os.makedirs(CHART_DIR, exist_ok=True)

# ─── CSS + helper for blue-framed summary tables ──────────
SUMMARY_CSS = """
<style>
.summary-table{width:100%;border-collapse:collapse;
  font-family:Verdana,Arial,sans-serif;font-size:12px;
  border:3px solid #003366;}
.summary-table th{background:#f2f2f2;padding:4px 6px;
  border:1px solid #B0B0B0;text-align:center;}
.summary-table td{padding:4px 6px;border:1px solid #B0B0B0;text-align:center;}
</style>
"""
def _pct_color(v):
    try:
        v=float(v)
        if v<=30:return"color:#008800;font-weight:bold"
        if v>=70:return"color:#CC0000;font-weight:bold"
    except:pass
    return""
def _build_html(df:pd.DataFrame)->str:
    sty=(df.style.hide(axis="index")
               .map(lambda v:_pct_color(v),subset="%ctile")
               .set_table_attributes('class="summary-table"'))
    return SUMMARY_CSS+sty.to_html()

# ─── Yield normaliser ─────────────────────────────────────
def _norm_yld(v):
    try:
        if v is None:return FALLBACK_YIELD
        v=float(v)
        if v<0.5:return v
        if v<20:return v/100
        return v/1000
    except:return FALLBACK_YIELD

# ─── yfinance helpers ────────────────────────────────────
def _fetch_pe_eps(tk):
    info=yf.Ticker(tk).info
    ttm_pe=info.get("trailingPE")
    fwd_pe=info.get("forwardPE")
    ttm_eps=info.get("trailingEps")
    if fwd_pe is None:
        px,eps_f=info.get("regularMarketPrice"),info.get("forwardEps")
        if px and eps_f:
            try:fwd_pe=px/eps_f
            except ZeroDivisionError:pass
    return ttm_pe,fwd_pe,ttm_eps

def _growth(ttm_pe,fwd_pe,y):
    return (y*ttm_pe-1 if ttm_pe else None,
            y*fwd_pe-1 if fwd_pe else None)

# ─── DB helpers ───────────────────────────────────────────
def _ensure_tables(conn):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS Index_Growth_History (
      Date TEXT, Ticker TEXT, Growth_Type TEXT, Implied_Growth REAL,
      PRIMARY KEY (Date,Ticker,Growth_Type));
    CREATE TABLE IF NOT EXISTS Index_PE_History (
      Date TEXT, Ticker TEXT, PE_Type TEXT, PE_Ratio REAL,
      PRIMARY KEY (Date,Ticker,PE_Type));
    CREATE TABLE IF NOT EXISTS Index_EPS_History (
      Date TEXT, Ticker TEXT, EPS_Type TEXT, EPS REAL,
      PRIMARY KEY (Date,Ticker,EPS_Type));
    CREATE TABLE IF NOT EXISTS Treasury_Yield_History (
      Date TEXT PRIMARY KEY, TenYr REAL);
    """)

def _log_today(y):
    today=datetime.today().strftime("%Y-%m-%d")
    with sqlite3.connect(DB_PATH) as conn:
        _ensure_tables(conn);cur=conn.cursor()
        cur.execute("INSERT OR REPLACE INTO Treasury_Yield_History VALUES (?,?)",
                    (today,y))
        for tk in IDXES:
            ttm_pe,fwd_pe,ttm_eps=_fetch_pe_eps(tk)
            ttm_g,fwd_g=_growth(ttm_pe,fwd_pe,y)
            if ttm_g is not None:
                cur.execute("INSERT OR REPLACE INTO Index_Growth_History VALUES (?,?, 'TTM', ?)",
                            (today,tk,ttm_g))
            if fwd_g is not None:
                cur.execute("INSERT OR REPLACE INTO Index_Growth_History VALUES (?,?, 'Forward', ?)",
                            (today,tk,fwd_g))
            if ttm_pe is not None:
                cur.execute("INSERT OR REPLACE INTO Index_PE_History VALUES (?,?, 'TTM', ?)",
                            (today,tk,ttm_pe))
            if fwd_pe is not None:
                cur.execute("INSERT OR REPLACE INTO Index_PE_History VALUES (?,?, 'Forward', ?)",
                            (today,tk,fwd_pe))
            if ttm_eps is not None:
                cur.execute("INSERT OR REPLACE INTO Index_EPS_History VALUES (?,?, 'TTM', ?)",
                            (today,tk,ttm_eps))
        conn.commit()

# ─── Helpers: fetch series ───────────────────────────────
def _pivot(tk,tbl,typ_col,val_col):
    with sqlite3.connect(DB_PATH) as conn:
        df=pd.read_sql_query(
            f"SELECT Date,{typ_col},{val_col} AS v FROM {tbl} "
            "WHERE Ticker=? ORDER BY Date ASC",conn,params=(tk,))
    if df.empty:return None
    df["Date"]=pd.to_datetime(df["Date"])
    return df.pivot(index="Date",columns=typ_col,values="v")

def _yield_series():
    with sqlite3.connect(DB_PATH) as conn:
        df=pd.read_sql_query("SELECT Date,TenYr FROM Treasury_Yield_History "
                             "ORDER BY Date ASC",conn)
    if df.empty:return None
    df["Date"]=pd.to_datetime(df["Date"])
    return df.set_index("Date")["TenYr"]

# ─── Stats summary ───────────────────────────────────────
def _percentile(s,v):
    s=pd.to_numeric(s,errors="coerce").dropna().sort_values()
    if s.empty or v is None or np.isnan(v):return None
    rank=np.searchsorted(s.values,float(v),side="right")
    return max(1,min(99,round(rank/len(s)*100)))

def _summary(df):
    if df is None or df.empty:return {}
    out={}
    for col in df.columns:
        s=df[col].dropna()
        if s.empty:continue
        latest=s.iloc[-1]
        out[col]=dict(Latest=latest,Avg=s.mean(),Med=s.median(),
                      Min=s.min(),Max=s.max(),
                      **{"%ctile":_percentile(s,latest) or "—"})
    return out

# ─── Plotly panel creator (chart + table) ────────────────
def _panel(df,title,ytitle,table_stats,filename,y2=None):
    """Write self-contained HTML panel with chart + summary table."""
    if df is None or df.empty:
        open(filename,"w").write("<p>No data yet.</p>")
        return
    fig=make_subplots(specs=[[{"secondary_y":y2 is not None}]])
    for col in df.columns:
        fig.add_trace(go.Scatter(x=df.index,y=df[col],mode="lines",name=col),
                      secondary_y=False)
    if y2 is not None:
        fig.add_trace(go.Scatter(x=y2.index,y=y2*100,mode="lines",
                                 line=dict(dash="dash"),name="10-yr Yield (%)"),
                      secondary_y=True)
        fig.update_yaxes(title_text="Yield (%)",secondary_y=True)
    fig.update_layout(
        title=title,yaxis_title=ytitle,template="plotly_white",height=500,
        xaxis=dict(rangeselector=dict(buttons=[
            dict(count=7,label="1 w",step="day",stepmode="backward"),
            dict(count=1,label="1 m",step="month",stepmode="backward"),
            dict(count=6,label="6 m",step="month",stepmode="backward"),
            dict(step="year",stepmode="todate",label="YTD"),
            dict(count=1,label="1 y",step="year",stepmode="backward"),
            dict(count=5,label="5 y",step="year",stepmode="backward"),
            dict(count=10,label="10 y",step="year",stepmode="backward"),
            dict(step="all",label="All")]),
            rangeslider=dict(visible=True),type="date"),
        legend=dict(orientation="h",yanchor="bottom",y=1.02,
                    xanchor="right",x=1))
    chart_html=fig.to_html(include_plotlyjs="cdn",full_html=False)
    table_html=_build_html(pd.DataFrame(table_stats).T) if table_stats else ""
    full=(
        "<html><head>"+SUMMARY_CSS+
        "<script src='https://cdn.plot.ly/plotly-latest.min.js'></script></head>"
        "<body>"+chart_html+table_html+"</body></html>"
    )
    open(filename,"w",encoding="utf-8").write(full)

# ─── Build assets per index ──────────────────────────────
def _build_assets(tk):
    # Implied Growth panel
    gdf=_pivot(tk,"Index_Growth_History","Growth_Type","Implied_Growth")
    _panel(gdf,f"{tk} Implied Growth","Implied Growth",
           _summary(gdf),os.path.join(CHART_DIR,f"{tk.lower()}_growth_panel.html"))

    # P/E panel
    pdf=_pivot(tk,"Index_PE_History","PE_Type","PE_Ratio")
    _panel(pdf,f"{tk} P/E Ratio","P/E",
           _summary(pdf),os.path.join(CHART_DIR,f"{tk.lower()}_pe_panel.html"))

    # EPS vs Yield panel
    with sqlite3.connect(DB_PATH) as conn:
        eps_s=(pd.read_sql_query(
            "SELECT Date,EPS FROM Index_EPS_History "
            "WHERE Ticker=? AND EPS_Type='TTM' ORDER BY Date ASC",conn,params=(tk,))
            .assign(Date=lambda d:pd.to_datetime(d["Date"]))
            .set_index("Date")["EPS"].dropna())
    yld=_yield_series()
    stats_eps=_summary(pd.DataFrame({"EPS":eps_s})) if not eps_s.empty else {}
    _panel(pd.DataFrame({"EPS":eps_s}),f"{tk} EPS vs 10-yr Yield","EPS (US$)",
           stats_eps,os.path.join(CHART_DIR,f"{tk.lower()}_eps_yield_panel.html"),
           y2=yld)

def _refresh_assets():
    for tk in IDXES:_build_assets(tk)

# ─── Overview (unchanged) ─────────────────────────────────
def _latest(series,tk,table,val_col,typ):
    with sqlite3.connect(DB_PATH) as conn:
        r=conn.execute(
            f"SELECT {val_col} FROM {table} "
            f"WHERE Ticker=? AND {typ}='TTM' ORDER BY Date DESC LIMIT 1",
            (tk,)).fetchone()
    return r[0] if r else None

def _percentile_wrap(series,val):
    p=_percentile(series,val)
    return f"{p}<sup>th</sup>" if p is not None else "–"

def _overview():
    with sqlite3.connect(DB_PATH) as conn:
        rows=[]
        for tk in IDXES:
            pe=_latest(None,tk,"Index_PE_History","PE_Ratio","PE_Type")
            gr=_latest(None,tk,"Index_Growth_History","Implied_Growth","Growth_Type")
            pe_hist=_history_series(conn,"Index_PE_History",tk,"PE_Ratio","PE_Type='TTM'")
            gr_hist=_history_series(conn,"Index_Growth_History",tk,"Implied_Growth","Growth_Type='TTM'")
            link=f'<a href="{tk.lower()}_growth_panel.html">{tk}</a>'
            if pe is None or gr is None:
                rows.append(f"<tr><td>{tk}</td><td colspan='4'>No data yet.</td></tr>")
            else:
                rows.append(
                    "<tr><td>"+link+"</td>"
                    f"<td>{pe:.1f}</td><td>{gr:.1%}</td>"
                    f"<td>{_percentile_wrap(pe_hist,pe)}</td>"
                    f"<td>{_percentile_wrap(gr_hist,gr)}</td></tr>")
    return ("<table border='1' style='border-collapse:collapse;'>"
            "<thead><tr><th>Ticker</th><th>P/E</th><th>Implied Growth</th>"
            "<th>P/E percentile</th><th>Implied Growth Percentile</th></tr></thead>"
            "<tbody>"+ "".join(rows)+"</tbody></table>")

# ─── Mini-main ───────────────────────────────────────────
def index_growth(treasury_yield:float|None=None)->str:
    y=_norm_yld(treasury_yield)
    print(f"[index_growth] Using 10-yr yield = {y:.4f}")
    _log_today(y);_refresh_assets()
    return _overview()

if __name__=="__main__":
    html=index_growth()  # fallback yield
    open(os.path.join(CHART_DIR,"spy_qqq_overview.html"),"w").write(html)
    print("Wrote spy_qqq_overview.html")

#!/usr/bin/env python3
# index_growth_charts.py  –  FULL FILE  (v2025-07-13 final)
# -----------------------------------------------------------
# • Builds Implied-Growth & P/E charts + matching summary
#   tables for one index ticker (default = SPY)
# • Uniform CSS styling, headless-safe backend, robust
#   column detection, empty-series guards, numeric coercion
# -----------------------------------------------------------

import os, sqlite3, pandas as pd, matplotlib
matplotlib.use("Agg")                     # headless-safe
import matplotlib.pyplot as plt

DB_PATH = "Stock Data.db"
OUT_DIR = "charts"
os.makedirs(OUT_DIR, exist_ok=True)

# ───────── shared CSS ──────────────────────────────────────
SUMMARY_CSS = """
<style>
.summary-table{
  width:100%;border-collapse:collapse;
  font-family:Verdana,Arial,sans-serif;font-size:12px}
.summary-table th{
  background:#003366;color:#fff;padding:4px 6px;
  border:1px solid #ddd;text-align:center}
.summary-table td{
  padding:4px 6px;border:1px solid #ddd;text-align:center}
</style>
"""

# ───────── helpers ─────────────────────────────────────────
def _columns(conn) -> list[str]:
    return [r[1] for r in conn.execute("PRAGMA table_info(Index_Growth_History)")]

def _pe_column(conn) -> str:
    cols, low = _columns(conn), {}
    for c in cols: low[c.lower()] = c
    pref = ["PE_Ratio","PE","P_E","PERatio","PE_ratio",
            "TTM_PE","PE_TTM","TTM_PE_Ratio",
            "PriceEarnings","Price_Earnings"]
    for name in pref:
        if name.lower() in low: return low[name.lower()]
    for c in cols:
        cln = c.replace("_","").lower()
        if "pe" in cln and "pct" not in cln and "percent" not in cln:
            return c
    raise RuntimeError("P/E column not found in Index_Growth_History")

def _series(conn, col:str, ticker:str="SPY") -> pd.Series:
    q = f"""SELECT Date,{col}
            FROM   Index_Growth_History
            WHERE  Ticker=? AND Growth_Type='TTM'
            ORDER  BY Date"""
    df = pd.read_sql(q, conn, params=(ticker,))
    df["Date"] = pd.to_datetime(df["Date"])
    s = pd.to_numeric(df.set_index("Date")[col], errors="coerce").dropna()
    return s

def _pctile(s:pd.Series): return "—" if s.empty else round(s.rank(pct=True).iloc[-1]*100,2)

def _row(label:str, s:pd.Series)->dict:
    if s.empty:
        na = "N/A"; return {"Metric":label,"Latest":na,"Avg":na,"Med":na,
                            "Min":na,"Max":na,"%ctile":"—"}
    r = lambda f: round(f(s),2)
    return {"Metric":label,"Latest":r(lambda x:x.iloc[-1]),
            "Avg":r(pd.Series.mean),"Med":r(pd.Series.median),
            "Min":r(pd.Series.min),"Max":r(pd.Series.max),
            "%ctile":_pctile(s)}

def _chart(s, title, ylab, fname):
    plt.figure(); plt.plot(s.index, s.values)
    plt.title(title); plt.ylabel(ylab)
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    path = os.path.join(OUT_DIR,fname); plt.savefig(path); plt.close(); return path

def _html(df, fname, pct_cols=None):
    fmt = {c:"{:,.2%}".format for c in (pct_cols or [])}
    html = SUMMARY_CSS + (
        df.style.format(fmt).hide(axis="index")
          .set_table_attributes('class="summary-table"').to_html())
    path = os.path.join(OUT_DIR,fname)
    open(path,"w",encoding="utf-8").write(html); return path

# ───────── entrypoint ─────────────────────────────────────
def render_index_growth_charts(ticker:str="SPY"):
    with sqlite3.connect(DB_PATH) as conn:
        ig_s = _series(conn,"Implied_Growth",ticker)
        pe_s = _series(conn,_pe_column(conn),ticker)
    ig_png = _chart(ig_s,f"{ticker} Implied Growth (TTM)",
                    "Implied Growth Rate",f"{ticker}_implied_growth.png")
    pe_png = _chart(pe_s,f"{ticker} P/E Ratio","P/E",
                    f"{ticker}_pe_ratio.png")
    ig_html = _html(pd.DataFrame([_row("Implied Growth (TTM)",ig_s)]),
                    f"{ticker}_implied_growth_summary.html",
                    pct_cols=["Latest","Avg","Med","Min","Max"])
    pe_html = _html(pd.DataFrame([_row("P/E Ratio (TTM)",pe_s)]),
                    f"{ticker}_pe_ratio_summary.html")
    return {"implied_chart":ig_png,"pe_chart":pe_png,
            "implied_table":ig_html,"pe_table":pe_html}

if __name__ == "__main__":
    import sys ; tk = sys.argv[1] if len(sys.argv)>1 else "SPY"
    for k,v in render_index_growth_charts(tk).items(): print(f"{k}: {v}")

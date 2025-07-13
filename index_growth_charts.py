#!/usr/bin/env python3
# index_growth_charts.py  –  FULL FILE  (v2025-07-14 a)
# -----------------------------------------------------------
# • Builds Implied-Growth & P/E charts
# • Saves two separate, blue-framed summary tables:
#      <T>_implied_growth_summary.html   /  <t>_growth_summary.html
#      <T>_pe_ratio_summary.html         /  <t>_pe_summary.html
# • Implied-growth values rendered as percentages
# -----------------------------------------------------------

import os, sqlite3, pandas as pd, matplotlib
matplotlib.use("Agg")                      # headless backend
import matplotlib.pyplot as plt

DB_PATH, OUT_DIR = "Stock Data.db", "charts"
os.makedirs(OUT_DIR, exist_ok=True)

# ───────── shared CSS (blue frame + grey header/grid) ──────
SUMMARY_CSS = """
<style>
.summary-table{
  width:100%;border-collapse:collapse;
  font-family:Verdana,Arial,sans-serif;font-size:12px;
  border:3px solid #003366;
}
.summary-table th{
  background:#f2f2f2;padding:4px 6px;border:1px solid #B0B0B0;text-align:center}
.summary-table td{
  padding:4px 6px;border:1px solid #B0B0B0;text-align:center}
</style>
"""

# ───────── helpers ─────────────────────────────────────────
def _columns(conn):          # all cols in Index_Growth_History
    return [r[1] for r in conn.execute("PRAGMA table_info(Index_Growth_History)")]

def _pe_col(conn):           # robust P/E column resolver
    cols, low = _columns(conn), {}
    for c in cols: low[c.lower()] = c
    pref = ["PE_Ratio","PE","P_E","PERatio","PE_ratio",
            "TTM_PE","PE_TTM","TTM_PE_Ratio","PriceEarnings","Price_Earnings"]
    for p in pref:
        if p.lower() in low:
            return low[p.lower()]
    for c in cols:
        cln=c.replace("_","").lower()
        if "pe" in cln and "pct" not in cln and "percent" not in cln:
            return c
    raise RuntimeError("No P/E column in Index_Growth_History")

def _series(conn, col, tk):
    df = pd.read_sql(
        f"""SELECT Date,{col}
              FROM Index_Growth_History
             WHERE Ticker=? AND Growth_Type='TTM'
          ORDER BY Date""", conn, params=(tk,))
    df["Date"] = pd.to_datetime(df["Date"])
    return pd.to_numeric(df.set_index("Date")[col], errors="coerce").dropna()

def _pctile(s): return "—" if s.empty else round(s.rank(pct=True).iloc[-1]*100,2)

def _row(label,s):   # numerical values kept raw; % formatting applied later
    if s.empty:
        return dict(Metric=label,Latest="N/A",Avg="N/A",Med="N/A",
                    Min="N/A",Max="N/A",**{"%ctile":"—"})
    r=lambda f: f(s)
    return dict(Metric=label,
                Latest=r(lambda x:x.iloc[-1]),
                Avg=r(pd.Series.mean),Med=r(pd.Series.median),
                Min=r(pd.Series.min),Max=r(pd.Series.max),
                **{"%ctile":_pctile(s)})

def _chart(s,title,ylab,fname):
    plt.figure(); plt.plot(s.index,s.values)
    plt.title(title); plt.ylabel(ylab)
    plt.xticks(rotation=45,ha="right"); plt.tight_layout()
    path=os.path.join(OUT_DIR,fname); plt.savefig(path); plt.close(); return path

def _pct_color(val):      # green ≤30, red ≥70
    try:
        v=float(val)
        if v<=30: return "color:#008800;font-weight:bold"
        if v>=70: return "color:#CC0000;font-weight:bold"
    except: pass
    return ""

def _style_table(df, pct=False):
    """Return HTML for df with correct formatting (pct→% with 2-dp)."""
    def fmt_num(v):
        if pd.isna(v):              return ""
        if isinstance(v,(int,float)):
            return f"{v*100:.2f}%" if pct else f"{v:.2f}"
        return v
    styled = (df.style
                .format({c:fmt_num for c in df.columns if c!="Metric"})
                .hide(axis="index")
                .map(_pct_color,subset="%ctile")
                .set_table_attributes('class="summary-table"')
                .to_html())
    return SUMMARY_CSS + styled

def _save_tables(tk, ig_df, pe_df):
    """Write four HTML fragments: growth & PE, both upper/lower-case names."""
    files = {
        f"{tk}_implied_growth_summary.html": _style_table(ig_df, pct=True),
        f"{tk}_pe_ratio_summary.html":       _style_table(pe_df, pct=False),
        f"{tk.lower()}_growth_summary.html": _style_table(ig_df, pct=True),
        f"{tk.lower()}_pe_summary.html":     _style_table(pe_df, pct=False),
    }
    # for very old pages still reading *_growth_tbl.html
    files[f"{tk}_growth_tbl.html"] = files[f"{tk}_implied_growth_summary.html"]

    for name, html in files.items():
        with open(os.path.join(OUT_DIR,name), "w", encoding="utf-8") as f:
            f.write(html)
    return files[f"{tk}_implied_growth_summary.html"]   # return one canonical path

# ───────── callable entry-point / mini-main ───────────────
def render_index_growth_charts(tk="SPY"):
    with sqlite3.connect(DB_PATH) as conn:
        ig_s=_series(conn,"Implied_Growth",tk)
        pe_s=_series(conn,_pe_col(conn),   tk)

    ig_png=_chart(ig_s,f"{tk} Implied Growth (TTM)",
                  "Implied Growth Rate",f"{tk}_implied_growth.png")
    pe_png=_chart(pe_s,f"{tk} P/E Ratio","P/E",
                  f"{tk}_pe_ratio.png")

    tables=_save_tables(
        tk,
        pd.DataFrame([_row("Implied Growth (TTM)",ig_s)]),
        pd.DataFrame([_row("PE Ratio (TTM)",pe_s)]))

    return {"implied_chart":ig_png,"pe_chart":pe_png,"growth_table":tables}

# legacy alias used elsewhere
mini_main = render_index_growth_charts

if __name__=="__main__":
    import sys
    ticker=sys.argv[1] if len(sys.argv)>1 else "SPY"
    for k,v in render_index_growth_charts(ticker).items():
        print(f"{k}: {v}")

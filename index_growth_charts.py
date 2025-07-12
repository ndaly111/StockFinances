#!/usr/bin/env python3
# index_growth_charts.py  –  FULL FILE  (v2025-07-13 i)
# -----------------------------------------------------------
# • Builds Implied-Growth & P/E charts
# • Writes ONE summary table styled exactly like the
#   reference screenshot (blue outer frame, grey header, grid)
# -----------------------------------------------------------

import os, sqlite3, pandas as pd, matplotlib
matplotlib.use("Agg")                  # CI / headless backend
import matplotlib.pyplot as plt

DB_PATH, OUT_DIR = "Stock Data.db", "charts"
os.makedirs(OUT_DIR, exist_ok=True)

# ───────── table CSS (pixel-perfect match) ─────────────────
SUMMARY_CSS = """
<style>
.summary-table{
  width:100%;
  border-collapse:collapse;
  font-family:Verdana,Arial,sans-serif;
  font-size:12px;
  border:3px solid #003366;               /* blue outer frame */
}
.summary-table th{
  background:#f2f2f2;                     /* light-grey header */
  padding:4px 6px;
  border:1px solid #B0B0B0;               /* mid-grey grid */
  text-align:center;
}
.summary-table td{
  padding:4px 6px;
  border:1px solid #B0B0B0;               /* mid-grey grid */
  text-align:center;
}
</style>
"""

# ───────── helpers ─────────────────────────────────────────
def _columns(conn):
    return [r[1] for r in conn.execute(
            "PRAGMA table_info(Index_Growth_History)")]

def _pe_col(conn):
    cols, low = _columns(conn), {}
    for c in cols: low[c.lower()] = c
    pref = ["PE_Ratio","PE","P_E","PERatio","PE_ratio",
            "TTM_PE","PE_TTM","TTM_PE_Ratio",
            "PriceEarnings","Price_Earnings"]
    for p in pref:
        if p.lower() in low: return low[p.lower()]
    for c in cols:                                  # fuzzy fallback
        if "pe" in c.lower().replace("_","") and "pct" not in c.lower():
            return c
    raise RuntimeError("No P/E column in Index_Growth_History")

def _series(conn, col, tk="SPY"):
    df = pd.read_sql(
        f"""SELECT Date,{col}
              FROM Index_Growth_History
             WHERE Ticker=? AND Growth_Type='TTM'
          ORDER BY Date""", conn, params=(tk,))
    df["Date"] = pd.to_datetime(df["Date"])
    return pd.to_numeric(df.set_index("Date")[col], errors="coerce").dropna()

def _pctile(s): return "—" if s.empty else round(s.rank(pct=True).iloc[-1]*100, 2)

def _row(label, s):
    if s.empty:
        return {"Metric":label,"Latest":"N/A","Avg":"N/A","Med":"N/A",
                "Min":"N/A","Max":"N/A","%ctile":"—"}
    r = lambda f: round(f(s),2)
    return {"Metric":label,"Latest":r(lambda x:x.iloc[-1]),"Avg":r(pd.Series.mean),
            "Med":r(pd.Series.median),"Min":r(pd.Series.min),
            "Max":r(pd.Series.max),"%ctile":_pctile(s)}

def _chart(s, title, ylab, fname):
    plt.figure()
    plt.plot(s.index, s.values)
    plt.title(title); plt.ylabel(ylab)
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    path = os.path.join(OUT_DIR, fname)
    plt.savefig(path); plt.close(); return path

def _pct_color(val):        # green ≤30, red ≥70
    try:
        v = float(val)
        if v <= 30: return "color:#008800;font-weight:bold"
        if v >= 70: return "color:#CC0000;font-weight:bold"
    except: pass
    return ""

def _write_summary(df, tk):
    html = (
        df.style
          .format({"Latest":"{:.2f}".format,"Avg":"{:.2f}".format,
                   "Med":"{:.2f}".format,"Min":"{:.2f}".format,
                   "Max":"{:.2f}".format,"%ctile":"{:.2f}".format})
          .hide(axis="index")
          .map(_pct_color, subset="%ctile")
          .set_table_attributes('class="summary-table"')
          .to_html()
    )
    final_html = SUMMARY_CSS + html
    # main + legacy filenames (so templates don’t break)
    for name in [f"{tk}_summary.html",
                 f"{tk}_implied_growth_summary.html",
                 f"{tk}_pe_ratio_summary.html"]:
        with open(os.path.join(OUT_DIR, name), "w", encoding="utf-8") as f:
            f.write(final_html)
    return os.path.join(OUT_DIR, f"{tk}_summary.html")

# ───────── callable entry-point / mini-main ────────────────
def render_index_growth_charts(tk="SPY"):
    with sqlite3.connect(DB_PATH) as conn:
        ig_s = _series(conn, "Implied_Growth", tk)
        pe_s = _series(conn, _pe_col(conn), tk)

    ig_png = _chart(ig_s, f"{tk} Implied Growth (TTM)",
                    "Implied Growth Rate", f"{tk}_implied_growth.png")
    pe_png = _chart(pe_s, f"{tk} P/E Ratio", "P/E",
                    f"{tk}_pe_ratio.png")

    summary_html = _write_summary(
        pd.DataFrame([_row("Implied Growth (TTM)", ig_s),
                      _row("PE Ratio (TTM)",       pe_s)]),
        tk)

    return {"implied_chart": ig_png,
            "pe_chart":      pe_png,
            "summary_table": summary_html}

# legacy alias
mini_main = render_index_growth_charts

if __name__ == "__main__":
    import sys
    tk = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    for k, v in render_index_growth_charts(tk).items():
        print(f"{k}: {v}")

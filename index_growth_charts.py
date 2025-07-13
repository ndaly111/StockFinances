#!/usr/bin/env python3
# index_growth_charts.py  –  FULL FILE  (v2025-07-14 c)
# -----------------------------------------------------------
# Generates:
#   • <T>_implied_growth.png / <T>_pe_ratio.png
#   • Growth table  →  <T>_implied_growth_summary.html
#                     <t>_growth_summary.html
#                     <T>_growth_tbl.html      (very old pages)
#   • P/E   table  →  <T>_pe_ratio_summary.html
#                     <t>_pe_summary.html
# -----------------------------------------------------------

import os, sqlite3, pandas as pd, matplotlib
matplotlib.use("Agg")                      # headless / CI backend
import matplotlib.pyplot as plt

DB_PATH, OUT_DIR = "Stock Data.db", "charts"
os.makedirs(OUT_DIR, exist_ok=True)

# ───────── shared CSS (blue frame + grey header/grid) ──────
SUMMARY_CSS = """
<style>
.summary-table{
  width:100%;border-collapse:collapse;
  font-family:Verdana,Arial,sans-serif;font-size:12px;
  border:3px solid #003366;}
.summary-table th{
  background:#f2f2f2;padding:4px 6px;border:1px solid #B0B0B0;text-align:center;}
.summary-table td{
  padding:4px 6px;border:1px solid #B0B0B0;text-align:center;}
</style>
"""

# ───────── helpers ─────────────────────────────────────────
def _columns(conn):
    return [r[1] for r in conn.execute("PRAGMA table_info(Index_Growth_History)")]

def _pe_col(conn):
    cols, low = _columns(conn), {}
    for c in cols: low[c.lower()] = c
    pref = ["PE_Ratio","PE","P_E","PERatio","PE_ratio",
            "TTM_PE","PE_TTM","TTM_PE_Ratio","PriceEarnings","Price_Earnings"]
    for p in pref:
        if p.lower() in low: return low[p.lower()]
    for c in cols:
        cln = c.replace("_","").lower()
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

# ── percentage formatter (non-breaking space before %) ─────
def _pct_fmt(x: float) -> str:
    return f"{x * 100:.2f} %"           # NBSP keeps % attached

def _row(label: str, s: pd.Series, as_pct: bool = False) -> dict:
    """Return one summary row; numeric stats pre-formatted if as_pct."""
    if s.empty:
        return {"Metric": label, "Latest": "N/A", "Avg": "N/A", "Med": "N/A",
                "Min": "N/A", "Max": "N/A", "%ctile": "—"}

    stats = {
        "Metric": label,
        "Latest": s.iloc[-1],
        "Avg":    s.mean(),
        "Med":    s.median(),
        "Min":    s.min(),
        "Max":    s.max(),
        "%ctile": round(s.rank(pct=True).iloc[-1] * 100, 2)   # keep 0-100
    }

    if as_pct:
        for k in ("Latest", "Avg", "Med", "Min", "Max"):
            stats[k] = _pct_fmt(stats[k])

    return stats

def _chart(series, title, ylab, fname):
    plt.figure()
    plt.plot(series.index, series.values)
    plt.title(title); plt.ylabel(ylab)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, fname)
    plt.savefig(path); plt.close(); return path

def _pct_color(val):                  # green ≤30, red ≥70
    try:
        v = float(val)
        if v <= 30: return "color:#008800;font-weight:bold"
        if v >= 70: return "color:#CC0000;font-weight:bold"
    except: pass
    return ""

def _format_table(df, percent_cols):
    """Return styled HTML; columns in percent_cols already pre-formatted."""
    def fmt_cell(x, col):
        if isinstance(x,

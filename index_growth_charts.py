#!/usr/bin/env python3
# index_growth_charts.py  –  FULL FILE  (v2025-07-14 e)
# -----------------------------------------------------------
# Outputs
#   • <T>_implied_growth.png | <T>_pe_ratio.png
#   • Growth tables:
#       <T>_implied_growth_summary.html
#       <t>_growth_summary.html
#       <T>_growth_tbl.html              (very old pages)
#   • P/E tables:
#       <T>_pe_ratio_summary.html
#       <t>_pe_summary.html
# -----------------------------------------------------------

import os, sqlite3, pandas as pd, matplotlib
matplotlib.use("Agg")                     # headless / CI backend
import matplotlib.pyplot as plt

DB_PATH, OUT_DIR = "Stock Data.db", "charts"
os.makedirs(OUT_DIR, exist_ok=True)

# ───────── CSS (blue frame, grey grid) ─────────────────────
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

# ───────── helpers ─────────────────────────────────────────
def _columns(conn):
    return [row[1] for row in conn.execute("PRAGMA table_info(Index_Growth_History)")]

def _pe_col(conn):
    cols = _columns(conn); low = {c.lower(): c for c in cols}
    pref = ["PE_Ratio","PE","P_E","PERatio","PE_ratio",
            "TTM_PE","PE_TTM","TTM_PE_Ratio",
            "PriceEarnings","Price_Earnings"]
    for p in pref:
        if p.lower() in low: return low[p.lower()]
    for c in cols:
        cln = c.replace("_","").lower()
        if "pe" in cln and "pct" not in cln and "percent" not in cln:
            return c
    raise RuntimeError("No P/E column found in Index_Growth_History")

def _series(conn, col, tk):
    df = pd.read_sql(
        f"""SELECT Date,{col}
              FROM Index_Growth_History
             WHERE Ticker=? AND Growth_Type='TTM'
          ORDER  BY Date""", conn, params=(tk,))
    df["Date"] = pd.to_datetime(df["Date"])
    return pd.to_numeric(df.set_index("Date")[col], errors="coerce").dropna()

# ── percentile helper: whole number, no decimals ───────────
def _pctile(s) -> str:
    """Return whole-number percentile (e.g. 87.44 → '87')."""
    return "—" if s.empty else str(int(round(s.rank(pct=True).iloc[-1] * 100)))

def _pct_fmt(x: float) -> str:                 # 0.1923 → '19.23 %'
    return f"{x * 100:.2f} %"                 # NBSP keeps % with the number

def _row(label, s, pct=False):
    if s.empty:
        return dict(Metric=label, Latest="N/A", Avg="N/A", Med="N/A",
                    Min="N/A", Max="N/A", **{"%ctile":"—"})
    stats = dict(
        Metric = label,
        Latest = s.iloc[-1],
        Avg    = s.mean(),
        Med    = s.median(),
        Min    = s.min(),
        Max    = s.max(),
        **{"%ctile": _pctile(s)}
    )
    if pct:                                     # convert core stats to XX.XX %
        for k in ("Latest","Avg","Med","Min","Max"):
            stats[k] = _pct_fmt(stats[k])
    return stats

def _chart(s, title, ylab, fname):
    plt.figure()
    plt.plot(s.index, s.values)
    plt.title(title); plt.ylabel(ylab)
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    path = os.path.join(OUT_DIR, fname)
    plt.savefig(path); plt.close(); return path

def _pct_color(v):                              # green ≤30, red ≥70
    try:
        v = float(v)
        if v <= 30: return "color:#008800;font-weight:bold"
        if v >= 70: return "color:#CC0000;font-weight:bold"
    except: pass
    return ""

def _build_html(df):
    sty = (df.style
             .hide(axis="index")
             .map(_pct_color, subset="%ctile")
             .set_table_attributes('class="summary-table"'))
    return SUMMARY_CSS + sty.to_html()

def _save_tables(tk, ig_df, pe_df):
    files = {
        f"{tk}_implied_growth_summary.html": _build_html(ig_df),
        f"{tk.lower()}_growth_summary.html": _build_html(ig_df),
        f"{tk}_growth_tbl.html":             _build_html(ig_df),   # legacy
        f"{tk}_pe_ratio_summary.html":       _build_html(pe_df),
        f"{tk.lower()}_pe_summary.html":     _build_html(pe_df)
    }
    for name, html in files.items():
        with open(os.path.join(OUT_DIR, name), "w", encoding="utf-8") as f:
            f.write(html)

# ───────── callable entry-point / mini-main ────────────────
def render_index_growth_charts(tk="SPY"):
    with sqlite3.connect(DB_PATH) as conn:
        ig_s = _series(conn, "Implied_Growth", tk)
        pe_s = _series(conn, _pe_col(conn),   tk)

    _chart(ig_s, f"{tk} Implied Growth (TTM)",
           "Implied Growth Rate", f"{tk}_implied_growth.png")
    _chart(pe_s, f"{tk} P/E Ratio", "P/E",
           f"{tk}_pe_ratio.png")

    _save_tables(
        tk,
        pd.DataFrame([_row("Implied Growth (TTM)", ig_s, pct=True)]),
        pd.DataFrame([_row("PE Ratio (TTM)",       pe_s, pct=False)])
    )

# legacy alias
mini_main = render_index_growth_charts

if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    render_index_growth_charts(ticker)
    print("Tables & charts generated for", ticker)

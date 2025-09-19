#!/usr/bin/env python3
# index_growth_charts.py  –  FULL FILE  (v2025-07-14 g)
# -----------------------------------------------------------
# • Reads Implied_Growth from   Index_Growth_History
# • Reads P/E          from     Index_PE_History
# • Generates charts + tables under all legacy filenames
# -----------------------------------------------------------

import os, sqlite3, pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")                     # headless / CI backend
import matplotlib.pyplot as plt

DB_PATH, OUT_DIR = "Stock Data.db", "charts"
os.makedirs(OUT_DIR, exist_ok=True)

# ───────── uniform CSS (blue frame + grey grid) ────────────
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
def _series_growth(conn, tk):
    """Return Implied_Growth (TTM) series for ticker tk."""
    df = pd.read_sql(
        """SELECT Date, Implied_Growth AS val
             FROM Index_Growth_History
            WHERE Ticker=? AND Growth_Type='TTM'
         ORDER BY Date""", conn, params=(tk,))
    df["Date"] = pd.to_datetime(df["Date"])
    return pd.to_numeric(df.set_index("Date")["val"], errors="coerce").dropna()

def _series_pe(conn, tk):
    """Return PE_Ratio (TTM) series for ticker tk."""
    df = pd.read_sql(
        """SELECT Date, PE_Ratio AS val
             FROM Index_PE_History
            WHERE Ticker=? AND PE_Type='TTM'
         ORDER BY Date""", conn, params=(tk,))
    df["Date"] = pd.to_datetime(df["Date"])
    return pd.to_numeric(df.set_index("Date")["val"], errors="coerce").dropna()

def _pctile(s) -> str:                      # whole-number percentile
    """Return percentile rank of the latest value in *s* (1-99)."""
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return "—"
    val = s.iloc[-1]
    s_sorted = s.sort_values()
    rank = np.searchsorted(s_sorted.values, float(val), side="right")
    pct  = (rank / len(s_sorted)) * 100
    return str(int(round(max(1, min(99, pct)))))

def _pct_fmt(x: float) -> str:              # 0.1923 → '19.23 %'
    return f"{x * 100:.2f} %"

def _row(label, s, pct=False):
    if s.empty:
        return dict(Metric=label, Latest="N/A", Avg="N/A", Med="N/A",
                    Min="N/A", Max="N/A", **{"%ctile": "—"})
    stats = dict(
        Metric = label,
        Latest = s.iloc[-1],
        Avg    = s.mean(),
        Med    = s.median(),
        Min    = s.min(),
        Max    = s.max(),
        **{"%ctile": _pctile(s)}
    )
    if pct:                                 # convert to XX.XX %
        for k in ("Latest","Avg","Med","Min","Max"):
            stats[k] = _pct_fmt(stats[k])
    else:                                   # numeric table → two decimals
        for k in ("Latest","Avg","Med","Min","Max"):
            stats[k] = f"{stats[k]:.2f}"
    return stats

def _chart(series, title, ylab, fname):
    plt.figure()
    plt.plot(series.index, series.values)
    plt.title(title); plt.ylabel(ylab)
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    path = os.path.join(OUT_DIR, fname)
    plt.savefig(path); plt.close(); return path

def _pct_color(v):                          # green ≤30, red ≥70
    try:
        v=float(v)
        if v<=30: return "color:#008800;font-weight:bold"
        if v>=70: return "color:#CC0000;font-weight:bold"
    except: pass
    return ""

def _build_html(df):
    sty = (df.style
             .hide(axis="index")
             .map(_pct_color, subset="%ctile")
             .set_table_attributes('class="summary-table"'))
    return SUMMARY_CSS + sty.to_html()

def _save_tables(tk, ig_df, pe_df):
    tk_lower = tk.lower()
    files = {
        f"{tk_lower}_growth_summary.html": _build_html(ig_df),
        f"{tk_lower}_pe_summary.html":     _build_html(pe_df),
    }
    for name, html in files.items():
        with open(os.path.join(OUT_DIR, name), "w", encoding="utf-8") as f:
            f.write(html)

# ───────── callable entry-point / mini-main ────────────────
def render_index_growth_charts(tk="SPY"):
    with sqlite3.connect(DB_PATH) as conn:
        ig_s = _series_growth(conn, tk)
        pe_s = _series_pe(conn, tk)

    ig_plot = ig_s
    ig_ylabel = "Implied Growth Rate"
    if not ig_s.empty:
        med = ig_s.median(skipna=True)
        max_abs = ig_s.abs().max()
        if (
            pd.notna(med) and np.isfinite(med)
            and abs(med) < 1
            and pd.notna(max_abs) and np.isfinite(max_abs)
            and max_abs <= 2
        ):
            # Stored as decimals (e.g., 0.18 for 18%) → scale a copy for plotting.
            ig_plot = ig_s * 100
            ig_ylabel = "Implied Growth Rate (%)"

    _chart(ig_plot, f"{tk} Implied Growth (TTM)",
           ig_ylabel, f"{tk.lower()}_growth_chart.png")
    _chart(pe_s, f"{tk} P/E Ratio", "P/E",
           f"{tk.lower()}_pe_chart.png")

    _save_tables(
        tk,
        pd.DataFrame([_row("Implied Growth (TTM)", ig_s, pct=True)]),
        pd.DataFrame([_row("P/E Ratio (TTM)",       pe_s, pct=False)])
    )

# legacy alias
mini_main = render_index_growth_charts

if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    render_index_growth_charts(ticker)
    print("Tables & charts generated for", ticker)

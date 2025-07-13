#!/usr/bin/env python3
# index_growth_charts.py  –  FULL FILE  (v2025-07-13 l)   LAST UPDATED 2025-07-13
# ---------------------------------------------------------------------------------
# • Generates Implied-Growth & P/E charts
# • Writes ONE blue-framed summary table with BOTH metrics
# • Saves that table under every legacy filename so no template edits are needed
# ---------------------------------------------------------------------------------

import os, sqlite3, pandas as pd, matplotlib
matplotlib.use("Agg")                    # headless / CI-safe backend
import matplotlib.pyplot as plt

DB_PATH, OUT_DIR = "Stock Data.db", "charts"
os.makedirs(OUT_DIR, exist_ok=True)

# ───────── CSS (matches your screenshot) ──────────────────
SUMMARY_CSS = """
<style>
.summary-table{
  width:100%;border-collapse:collapse;
  font-family:Verdana,Arial,sans-serif;font-size:12px;
  border:3px solid #003366;
}
.summary-table th{
  background:#f2f2f2;
  padding:4px 6px;
  border:1px solid #B0B0B0;
  text-align:center;
}
.summary-table td{
  padding:4px 6px;
  border:1px solid #B0B0B0;
  text-align:center;
}
</style>
"""

# ───────── helpers ─────────────────────────────────────────
def _columns(conn):
    return [r[1] for r in conn.execute("PRAGMA table_info(Index_Growth_History)")]

def _pe_col(conn):
    cols = _columns(conn)
    low  = {c.lower(): c for c in cols}
    pref = ["PE_Ratio","PE","P_E","PERatio","PE_ratio",
            "TTM_PE","PE_TTM","TTM_PE_Ratio","PriceEarnings","Price_Earnings"]
    for p in pref:
        if p.lower() in low:
            return low[p.lower()]
    for c in cols:                       # fuzzy fallback
        cln = c.replace("_","").lower()
        if "pe" in cln and "pct" not in cln and "percent" not in cln:
            return c
    raise RuntimeError("No P/E column in Index_Growth_History")

def _series(conn, col, tk):
    df = pd.read_sql(
        f"""SELECT Date,{col}
              FROM Index_Growth_History
             WHERE Ticker=? AND Growth_Type='TTM'
          ORDER BY Date""",
        conn, params=(tk,))
    df["Date"] = pd.to_datetime(df["Date"])
    return pd.to_numeric(df.set_index("Date")[col], errors="coerce").dropna()

def _pctile(s):
    return "—" if s.empty else round(s.rank(pct=True).iloc[-1]*100, 2)

def _row(label, s):
    if s.empty:
        return {"Metric": label, "Latest": "N/A", "Avg": "N/A", "Med": "N/A",
                "Min": "N/A", "Max": "N/A", "%ctile": "—"}
    r = lambda f: round(f(s), 2)
    return {"Metric": label,
            "Latest": r(lambda x: x.iloc[-1]),
            "Avg":    r(pd.Series.mean),
            "Med":    r(pd.Series.median),
            "Min":    r(pd.Series.min),
            "Max":    r(pd.Series.max),
            "%ctile": _pctile(s)}

def _chart(s, title, ylab, fname):
    plt.figure()
    plt.plot(s.index, s.values)
    plt.title(title)
    plt.ylabel(ylab)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, fname)
    plt.savefig(path)
    plt.close()
    return path

def _pct_color(v):
    try:
        v = float(v)
        if v <= 30:
            return "color:#008800;font-weight:bold"
        if v >= 70:
            return "color:#CC0000;font-weight:bold"
    except Exception:
        pass
    return ""

_fmt = lambda v: "" if pd.isna(v) else (f"{v:.2f}" if isinstance(v, (int, float)) else v)

def _write_summary(df, tk):
    """Render the HTML table and save it under ALL expected filenames."""
    styled = (
        df.style
          .format({c: _fmt for c in df.columns if c != "Metric"})
          .hide(axis="index")
          .map(_pct_color, subset="%ctile")
          .set_table_attributes('class="summary-table"')
          .to_html()
    )
    final_html = SUMMARY_CSS + styled

    legacy_names = [
        f"{tk}_summary.html",                       # new canonical
        f"{tk}_implied_growth_summary.html",        # ticker pages
        f"{tk}_pe_ratio_summary.html",              # ticker pages
        f"{tk}_growth_tbl.html",                    # VERY old ticker pages
        f"{tk.lower()}_growth_summary.html",        # mini-page (spy)
        f"{tk.lower()}_pe_summary.html"             # mini-page (spy)
    ]

    for name in legacy_names:
        with open(os.path.join(OUT_DIR, name), "w", encoding="utf-8") as f:
            f.write(final_html)

    # return path to canonical file (not used by caller but handy)
    return os.path.join(OUT_DIR, legacy_names[0])

# ───────── main callable / mini-main ───────────────────────
def render_index_growth_charts(tk="SPY"):
    with sqlite3.connect(DB_PATH) as conn:
        ig_s = _series(conn, "Implied_Growth", tk)
        pe_s = _series(conn, _pe_col(conn),    tk)

    ig_png = _chart(ig_s, f"{tk} Implied Growth (TTM)",
                    "Implied Growth Rate", f"{tk}_implied_growth.png")
    pe_png = _chart(pe_s, f"{tk} P/E Ratio", "P/E",
                    f"{tk}_pe_ratio.png")

    summary_path = _write_summary(
        pd.DataFrame([
            _row("Implied Growth (TTM)", ig_s),
            _row("PE Ratio (TTM)",       pe_s)
        ]),
        tk
    )

    return {
        "implied_chart": ig_png,
        "pe_chart":      pe_png,
        "summary_table": summary_path
    }

# legacy alias expected elsewhere
mini_main = render_index_growth_charts

if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    for k, v in render_index_growth_charts(ticker).items():
        print(f"{k}: {v}")

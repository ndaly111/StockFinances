#!/usr/bin/env python3
# index_growth_charts.py  –  FULL FILE  (v2025-07-13 g)
# -----------------------------------------------------------
# • Generates Implied-Growth & P/E charts + matching summary
#   tables.  Styling now defers to Pandas’ default so the
#   P/E table looks exactly like the original top table.
# • Robust P/E column resolver, numeric coercion, empty-series
#   guards, CI-safe matplotlib backend, mini-main alias.
# -----------------------------------------------------------

import os, sqlite3, pandas as pd, matplotlib
matplotlib.use("Agg")                   # headless / CI backend
import matplotlib.pyplot as plt

DB_PATH, OUT_DIR = "Stock Data.db", "charts"
os.makedirs(OUT_DIR, exist_ok=True)

# ───────── helpers ─────────────────────────────────────────
def _columns(conn):             # existing cols in Index_Growth_History
    return [r[1] for r in conn.execute("PRAGMA table_info(Index_Growth_History)")]

def _pe_col(conn):              # robust P/E column resolver
    cols, low = _columns(conn), {}
    for c in cols: low[c.lower()] = c
    pref = ["PE_Ratio","PE","P_E","PERatio","PE_ratio",
            "TTM_PE","PE_TTM","TTM_PE_Ratio","PriceEarnings","Price_Earnings"]
    for p in pref:               # exact/near-exact first
        if p.lower() in low: return low[p.lower()]
    for c in cols:               # fuzzy “pe”
        cln = c.replace("_","").lower()
        if "pe" in cln and "pct" not in cln and "percent" not in cln:
            return c
    raise RuntimeError("No P/E column in Index_Growth_History")

def _series(conn, col, tk="SPY"):
    df = pd.read_sql(
        f"""SELECT Date,{col}
            FROM   Index_Growth_History
            WHERE  Ticker=? AND Growth_Type='TTM'
            ORDER  BY Date""", conn, params=(tk,))
    df["Date"] = pd.to_datetime(df["Date"])
    return pd.to_numeric(df.set_index("Date")[col], errors="coerce").dropna()

def _pctile(s): return "—" if s.empty else round(s.rank(pct=True).iloc[-1]*100, 2)

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
    plt.title(title); plt.ylabel(ylab)
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    path = os.path.join(OUT_DIR, fname)
    plt.savefig(path); plt.close(); return path

# simple green/red text colour for percentile column
def _color_pct(val):
    try:
        v = float(val)
        if v <= 30: return "color:#008800;font-weight:bold"
        if v >= 70: return "color:#CC0000;font-weight:bold"
    except: pass
    return ""

def _html(df, fname, pct_cols=None):
    pct_cols = pct_cols or []
    fmt = {c: "{:,.2%}".format for c in pct_cols}

    html = (df.style
              .format(fmt)
              .hide(axis="index")
              .map(_color_pct, subset="%ctile")   # Styler.map → no warning
              .to_html())                         # ← default Pandas styling
    path = os.path.join(OUT_DIR, fname)
    open(path, "w", encoding="utf-8").write(html)
    return path

# ───────── callable entry-point / mini-main ─────────────────
def render_index_growth_charts(tk="SPY"):
    with sqlite3.connect(DB_PATH) as conn:
        ig_s = _series(conn, "Implied_Growth", tk)
        pe_s = _series(conn, _pe_col(conn),   tk)

    ig_png = _chart(ig_s, f"{tk} Implied Growth (TTM)",
                    "Implied Growth Rate", f"{tk}_implied_growth.png")
    pe_png = _chart(pe_s, f"{tk} P/E Ratio", "P/E",
                    f"{tk}_pe_ratio.png")

    ig_html = _html(
        pd.DataFrame([_row("Implied Growth (TTM)", ig_s)]),
        f"{tk}_implied_growth_summary.html",
        pct_cols=["Latest", "Avg", "Med", "Min", "Max"]
    )
    pe_html = _html(
        pd.DataFrame([_row("P/E Ratio (TTM)", pe_s)]),
        f"{tk}_pe_ratio_summary.html"
    )

    return {"implied_chart": ig_png, "pe_chart": pe_png,
            "implied_table": ig_html, "pe_table": pe_html}

# legacy one-liner alias
mini_main = render_index_growth_charts

if __name__ == "__main__":
    import sys
    tk = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    for k, v in render_index_growth_charts(tk).items():
        print(f"{k}: {v}")

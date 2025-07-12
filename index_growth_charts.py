#!/usr/bin/env python3
# index_growth_charts.py  –  FULL FILE  (v2025-07-13 c)
# -----------------------------------------------------------
# • Builds Implied-Growth & P/E charts + matching summary
#   tables for a single index ticker (default = SPY)
# • Summary tables share identical CSS (.summary-table)
# • Ultra-flexible P/E column resolver + automatic numeric
#   conversion to avoid string-type errors                     ← NEW
# -----------------------------------------------------------

import os, sqlite3, pandas as pd, matplotlib.pyplot as plt

DB_PATH = "Stock Data.db"
OUT_DIR = "charts"
os.makedirs(OUT_DIR, exist_ok=True)

# ───────── shared CSS ──────────────────────────────────────
SUMMARY_CSS = """
<style>
.summary-table{
    width:100%;
    border-collapse:collapse;
    font-family:Verdana,Arial,sans-serif;
    font-size:12px;
}
.summary-table th{
    background:#003366;
    color:#fff;
    padding:4px 6px;
    border:1px solid #ddd;
    text-align:center;
}
.summary-table td{
    padding:4px 6px;
    border:1px solid #ddd;
    text-align:center;
}
</style>
"""

# ───────── helpers ─────────────────────────────────────────
def _get_columns(conn) -> list[str]:
    return [row[1] for row in conn.execute("PRAGMA table_info(Index_Growth_History)")]

def _resolve_pe_column(conn) -> str:
    cols = _get_columns(conn)
    lower = {c.lower(): c for c in cols}
    preferred = [
        "PE_Ratio","PE","P_E","PERatio","PE_ratio",
        "TTM_PE","PE_TTM","TTM_PE_Ratio","PriceEarnings","Price_Earnings"
    ]
    for name in preferred:
        if name.lower() in lower:
            return lower[name.lower()]
    # fuzzy contains "pe"
    for c in cols:
        clean = c.replace("_","").lower()
        if "pe" in clean and "pct" not in clean and "percent" not in clean:
            return c
    raise RuntimeError("No P/E column found in Index_Growth_History.")

def _fetch_series(conn, col:str, ticker:str="SPY") -> pd.Series:
    q = f"""
        SELECT Date, {col}
        FROM   Index_Growth_History
        WHERE  Ticker=? AND Growth_Type='TTM'
        ORDER  BY Date
    """
    df = pd.read_sql(q, conn, params=(ticker,))
    df["Date"] = pd.to_datetime(df["Date"])
    s = df.set_index("Date")[col]
    # ── NEW: force numeric ─────────────────────────────────
    s = pd.to_numeric(s, errors="coerce").dropna()
    return s

def _percentile(s:pd.Series) -> float:
    return round(s.rank(pct=True).iloc[-1] * 100, 2)

def _summary_row(label:str, s:pd.Series) -> dict:
    return {
        "Metric": label,
        "Latest": round(s.iloc[-1], 2),
        "Avg":    round(s.mean(),   2),
        "Med":    round(s.median(), 2),
        "Min":    round(s.min(),    2),
        "Max":    round(s.max(),    2),
        "%ctile": _percentile(s)
    }

def _save_chart(s:pd.Series, title:str, ylab:str, fname:str) -> str:
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

def _save_summary_html(df:pd.DataFrame, fname:str, pct_cols=None) -> str:
    pct_cols = pct_cols or []
    fmt = {c: "{:,.2%}".format for c in pct_cols}
    html = (
        SUMMARY_CSS +
        df.style.format(fmt).hide(axis="index")
          .set_table_attributes('class="summary-table"').to_html()
    )
    path = os.path.join(OUT_DIR, fname)
    with open(path, "w", encoding="utf-8") as f: f.write(html)
    return path

# ───────── main entrypoint ─────────────────────────────────
def render_index_growth_charts(ticker:str="SPY") -> dict:
    with sqlite3.connect(DB_PATH) as conn:
        ig_col = "Implied_Growth"
        pe_col = _resolve_pe_column(conn)

        ig_s = _fetch_series(conn, ig_col, ticker)
        pe_s = _fetch_series(conn, pe_col, ticker)

    ig_chart = _save_chart(ig_s, f"{ticker} Implied Growth (TTM)",
                           "Implied Growth Rate", f"{ticker}_implied_growth.png")
    pe_chart = _save_chart(pe_s, f"{ticker} P/E Ratio", "P/E",
                           f"{ticker}_pe_ratio.png")

    ig_html = _save_summary_html(
        pd.DataFrame([_summary_row("Implied Growth (TTM)", ig_s)]),
        f"{ticker}_implied_growth_summary.html",
        pct_cols=["Latest","Avg","Med","Min","Max"]
    )
    pe_html = _save_summary_html(
        pd.DataFrame([_summary_row("P/E Ratio (TTM)", pe_s)]),
        f"{ticker}_pe_ratio_summary.html"
    )

    return {
        "implied_chart": ig_chart,
        "pe_chart":      pe_chart,
        "implied_table": ig_html,
        "pe_table":      pe_html,
    }

if __name__ == "__main__":
    import sys
    tk = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    for k,v in render_index_growth_charts(tk).items():
        print(f"{k}: {v}")

#!/usr/bin/env python3
# index_growth_charts.py  –  FULL FILE  (v2025-07-13)
# -----------------------------------------------------------
# • Builds Implied-Growth & P/E charts + matching summary
#   tables for an index ticker (default = SPY)
# • Tables share identical CSS (.summary-table)
# • NEW: auto-detect correct P/E column (PE_Ratio ▸ PE)
# -----------------------------------------------------------

import os, sqlite3, pandas as pd, matplotlib.pyplot as plt
from datetime import datetime

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
def _resolve_column(conn, preferred:str, alternates:list[str]) -> str:
    """
    Return the first column that actually exists in Index_Growth_History.
    Raises if none are present.
    """
    cur = conn.execute("PRAGMA table_info(Index_Growth_History)")
    cols = {row[1].lower() for row in cur.fetchall()}
    for col in [preferred, *alternates]:
        if col.lower() in cols:
            return col
    raise RuntimeError(
        f"None of these columns exist in Index_Growth_History: "
        f"{[preferred, *alternates]}"
    )

def _fetch_series(conn, field:str, ticker:str="SPY") -> pd.Series:
    """Return a Date-indexed Series for the requested field."""
    # Auto-resolve if preferred column is missing (e.g., PE_Ratio → PE)
    col = _resolve_column(conn, field, ["PE", "P_E", "PERatio"]) if field == "PE_Ratio" else field
    q = f"""
        SELECT Date, {col}
        FROM   Index_Growth_History
        WHERE  Ticker=? AND Growth_Type='TTM'
        ORDER  BY Date
    """
    df = pd.read_sql(q, conn, params=(ticker,))
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date")[col].dropna()

def _percentile(series:pd.Series) -> float:
    return round(series.rank(pct=True).iloc[-1] * 100, 2)

def _summary_row(metric_name:str, series:pd.Series) -> dict:
    return {
        "Metric": metric_name,
        "Latest": round(series.iloc[-1], 2),
        "Avg":    round(series.mean(), 2),
        "Med":    round(series.median(), 2),
        "Min":    round(series.min(), 2),
        "Max":    round(series.max(), 2),
        "%ctile": _percentile(series)
    }

def _save_chart(series:pd.Series, title:str, ylab:str, fname:str) -> str:
    plt.figure()
    plt.plot(series.index, series.values)
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
    fmt = {c:"{:,.2%}".format for c in pct_cols}
    html = (
        SUMMARY_CSS
        + df.style
            .format(fmt)
            .hide(axis="index")
            .set_table_attributes('class="summary-table"')
            .to_html()
    )
    path = os.path.join(OUT_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path

# ───────── main entrypoint ─────────────────────────────────
def render_index_growth_charts(ticker:str="SPY") -> dict:
    """
    Generates:
       • <ticker>_implied_growth.png
       • <ticker>_pe_ratio.png
       • <ticker>_implied_growth_summary.html
       • <ticker>_pe_ratio_summary.html
    Returns a dict of paths (for downstream use).
    """
    with sqlite3.connect(DB_PATH) as conn:
        ig_series = _fetch_series(conn, "Implied_Growth", ticker)
        pe_series = _fetch_series(conn, "PE_Ratio",       ticker)

    ig_chart = _save_chart(
        ig_series, f"{ticker} Implied Growth (TTM)", "Implied Growth Rate",
        f"{ticker}_implied_growth.png"
    )
    pe_chart = _save_chart(
        pe_series, f"{ticker} P/E Ratio", "P/E",
        f"{ticker}_pe_ratio.png"
    )

    ig_df = pd.DataFrame([_summary_row("Implied Growth (TTM)", ig_series)])
    pe_df = pd.DataFrame([_summary_row("PE Ratio (TTM)",        pe_series)])

    ig_html = _save_summary_html(
        ig_df, f"{ticker}_implied_growth_summary.html",
        pct_cols=["Latest","Avg","Med","Min","Max"]
    )
    pe_html = _save_summary_html(
        pe_df, f"{ticker}_pe_ratio_summary.html"
    )

    return {
        "implied_chart":  ig_chart,
        "pe_chart":       pe_chart,
        "implied_table":  ig_html,
        "pe_table":       pe_html,
    }

# ───────── CLI helper ─────────────────────────────────────
if __name__ == "__main__":
    import sys
    tk = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    out_paths = render_index_growth_charts(tk)
    print("\n".join(f"{k}: {v}" for k,v in out_paths.items()))

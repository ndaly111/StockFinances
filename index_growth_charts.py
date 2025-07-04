# index_growth_charts.py
# --------------------------------------------------------------------
# Creates / refreshes implied-growth charts + summary tables for
# SPY and QQQ from the Index_Growth_History table in Stock Data.db
# --------------------------------------------------------------------

import os, sqlite3, pandas as pd, matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

DB_PATH     = "Stock Data.db"
TABLE       = "Index_Growth_History"
OUTPUT_DIR  = "charts"
INDEXES     = ["SPY", "QQQ"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Helpers ─────────────────────────────────────────────────────────
def _fetch_history(ticker):
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            f"SELECT Date, Growth_Type, Implied_Growth "
            f"FROM {TABLE} WHERE Ticker = ? ORDER BY Date ASC",
            conn, params=(ticker,)
        )
    if df.empty:
        return None
    df["Date"] = pd.to_datetime(df["Date"])
    return df.pivot(index="Date", columns="Growth_Type", values="Implied_Growth")

def _compute_summary(df):
    summary = {}
    for col in ["TTM", "Forward"]:
        if col in df:
            summary[col] = {
                "Average": df[col].mean(),
                "Median" : df[col].median(),
                "Min"    : df[col].min(),
                "Max"    : df[col].max()
            }
    return summary

def _build_chart(df, summary, tk):
    out_png = os.path.join(OUTPUT_DIR, f"{tk.lower()}_growth_chart.png")
    # Guard: if df is None or empty, write a 1×1 transparent PNG placeholder
    if df is None or df.empty:
        plt.figure(figsize=(0.01,0.01))
        plt.axis("off")
        plt.savefig(out_png, transparent=True, dpi=10)
        plt.close()
        return

    fig, ax = plt.subplots(figsize=(10,6))
    if "TTM" in df:
        ax.plot(df.index, df["TTM"], label="TTM", color="blue")
        for lbl, ls in [("Average",":"),("Median","--"),("Min","-."),("Max","-.")]:
            ax.axhline(summary["TTM"][lbl], color="blue", linestyle=ls, linewidth=1)
    if "Forward" in df:
        ax.plot(df.index, df["Forward"], label="Forward", color="green")
        for lbl, ls in [("Average",":"),("Median","--"),("Min","-."),("Max","-.")]:
            ax.axhline(summary["Forward"][lbl], color="green", linestyle=ls, linewidth=1)

    ax.set_title(f"{tk} Implied Growth Rates Over Time")
    ax.set_ylabel("Implied Growth Rate")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def _build_summary_html(summary, tk):
    out_html = os.path.join(OUTPUT_DIR, f"{tk.lower()}_growth_summary.html")
    if not summary:                 # no data yet
        open(out_html, "w", encoding="utf-8").write("<p>No implied-growth data yet.</p>")
        return

    rows = []
    link = f'<a href="{tk.lower()}_growth.html">{tk}</a>'
    for gtype, stats in summary.items():
        for stat, val in stats.items():
            rows.append({
                "Ticker"     : link,
                "Growth Type": gtype,
                "Statistic"  : stat,
                "Value"      : f"{val:.2%}"
            })
    pd.DataFrame(rows).to_html(out_html, index=False, escape=False)

# ─── Public mini-main ────────────────────────────────────────────────
def render_index_growth_charts():
    """
    Generates or refreshes:
        charts/spy_growth_chart.png
        charts/spy_growth_summary.html
        charts/qqq_growth_chart.png
        charts/qqq_growth_summary.html
    """
    print("[index_growth_charts] Building SPY & QQQ growth assets …")
    for tk in INDEXES:
        df       = _fetch_history(tk)
        summary  = _compute_summary(df) if df is not None else {}
        _build_summary_html(summary, tk)
        _build_chart(df, summary, tk)
    print("[index_growth_charts] Done.")

# ─── Stand-alone entry point ─────────────────────────────────────────
if __name__ == "__main__":
    render_index_growth_charts()

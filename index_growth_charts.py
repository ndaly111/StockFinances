# index_growth_charts.py
# --------------------------------------------------------------------
# Creates / refreshes implied-growth charts + summary tables for
# SPY and QQQ from the Index_Growth_History table in Stock Data.db
# --------------------------------------------------------------------

import os, sqlite3, pandas as pd, matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy.stats import percentileofscore

DB_PATH     = "Stock Data.db"
OUTPUT_DIR  = "charts"
TABLE       = "Index_Growth_History"
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

def get_percentile(value, data_list):
    try:
        return percentileofscore(data_list, value) / 100
    except:
        return None

def _get_price(conn, ticker):
    try:
        df = pd.read_sql_query("SELECT * FROM MarketData WHERE Ticker = ?", conn, params=(ticker,))
        if df.empty:
            return None
        return df["Price"].iloc[-1]
    except:
        return None

def _get_forward_eps_info(ticker):
    with sqlite3.connect(DB_PATH) as conn:
        price = _get_price(conn, ticker)
        print(f"[DEBUG] {ticker} current price: {price}")
        if price is None:
            return None, None

        pe_all = pd.read_sql_query("""
            SELECT Ticker, Date, PE_Ratio
            FROM Index_PE_History
            WHERE PE_Type = 'Forward'
        """, conn)

        print(f"[DEBUG] PE Forward Entries: {len(pe_all)}")

        latest_pe = (
            pe_all.sort_values(["Ticker", "Date"])
            .dropna()
            .drop_duplicates(subset=["Ticker"], keep="last")
        )

        latest_pe = latest_pe[latest_pe["PE_Ratio"] > 0]
        latest_pe["Forward_EPS"] = latest_pe["Ticker"].map(lambda t: _get_price(conn, t)) / latest_pe["PE_Ratio"]

        print(f"[DEBUG] Sample PE DF:\n{latest_pe.head()}")

        this_row = latest_pe[latest_pe["Ticker"].str.upper() == ticker.upper()]
        print(f"[DEBUG] Matching row for {ticker}:\n{this_row}")

        if this_row.empty:
            return None, None

        this_eps = price / this_row["PE_Ratio"].values[0]
        percentile = get_percentile(this_eps, latest_pe["Forward_EPS"].tolist())

        return this_eps, percentile

def _build_chart(df, summary, tk):
    out_png = os.path.join(OUTPUT_DIR, f"{tk.lower()}_growth_chart.png")
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
    if not summary:
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

    # Add forward EPS and percentile
    forward_eps, forward_eps_pct = _get_forward_eps_info(tk)
    if forward_eps is not None:
        rows.append({
            "Ticker"     : link,
            "Growth Type": "Forward EPS",
            "Statistic"  : "Latest",
            "Value"      : f"{forward_eps:.2f}"
        })
    if forward_eps_pct is not None:
        rows.append({
            "Ticker"     : link,
            "Growth Type": "Forward EPS",
            "Statistic"  : "Percentile",
            "Value"      : f"{forward_eps_pct:.2%}"
        })

    pd.DataFrame(rows).to_html(out_html, index=False, escape=False)

# ─── Public mini-main ────────────────────────────────────────────────
def render_index_growth_charts():
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

# index_growth_charts.py
# -----------------------------------------------------------
# Builds SPY & QQQ implied-growth charts + summary tables
# -----------------------------------------------------------

import os, sqlite3, pandas as pd, matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy.stats import percentileofscore   # now available

DB_PATH     = "Stock Data.db"
TABLE       = "Index_Growth_History"
OUTPUT_DIR  = "charts"
INDEXES     = ["SPY", "QQQ"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Helpers ────────────────────────────────────────────────
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
    if df is None or df.empty:
        return {}
    out = {}
    for col in ["TTM", "Forward"]:
        if col in df:
            out[col] = {
                "Average": df[col].mean(),
                "Median" : df[col].median(),
                "Min"    : df[col].min(),
                "Max"    : df[col].max()
            }
    return out

def _build_chart(df, summary, tk):
    out_png = os.path.join(OUTPUT_DIR, f"{tk.lower()}_growth_chart.png")
    if df is None or df.empty:
        plt.figure(figsize=(0.01, 0.01))
        plt.axis("off")
        plt.savefig(out_png, transparent=True, dpi=10)
        plt.close()
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    if "TTM" in df:
        ax.plot(df.index, df["TTM"], label="TTM", color="blue")
        for lbl, ls in [("Average", ":"), ("Median", "--"), ("Min", "-."), ("Max", "-.")]:
            ax.axhline(summary["TTM"][lbl], color="blue", linestyle=ls, linewidth=1)
    if "Forward" in df:
        ax.plot(df.index, df["Forward"], label="Forward", color="green")
        for lbl, ls in [("Average", ":"), ("Median", "--"), ("Min", "-."), ("Max", "-.")]:
            ax.axhline(summary["Forward"][lbl], color="green", linestyle=ls, linewidth=1)

    ax.set_title(f"{tk} Implied Growth Rates Over Time")
    ax.set_ylabel("Implied Growth Rate")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

# ─── Forward-EPS helpers ───────────────────────────────────
def _get_price(conn, ticker):
    row = pd.read_sql_query(
        "SELECT last_price FROM MarketData WHERE ticker = ?", conn, params=(ticker,)
    )
    return None if row.empty else row["last_price"].iloc[0]

def _get_forward_eps_info(ticker):
    with sqlite3.connect(DB_PATH) as conn:
        price = _get_price(conn, ticker)
        if price is None:
            return None, None

        pe_df = pd.read_sql_query("""
            SELECT Ticker, Date, PE_Ratio
            FROM Index_PE_History
            WHERE PE_Type = 'Forward' AND PE_Ratio > 0
            ORDER BY Ticker, Date
        """, conn)

        latest = (
            pe_df.dropna()
                 .sort_values(["Ticker", "Date"])
                 .drop_duplicates(subset=["Ticker"], keep="last")
        )
        if latest.empty:
            return None, None

        # Calculate Forward-EPS for every index ticker
        latest["Ticker"] = latest["Ticker"].str.upper()
        latest["Forward_EPS"] = latest.apply(
            lambda r: _get_price(conn, r["Ticker"]) / r["PE_Ratio"], axis=1
        )

        row = latest[latest["Ticker"] == ticker.upper()]
        if row.empty:
            return None, None

        this_eps = price / row["PE_Ratio"].iloc[0]
        pct      = percentileofscore(latest["Forward_EPS"], this_eps)  # 0-100 scale
        return round(this_eps, 2), round(pct, 2)

# ─── HTML summary ───────────────────────────────────────────
def _build_summary_html(summary, tk):
    out_html = os.path.join(OUTPUT_DIR, f"{tk.lower()}_growth_summary.html")
    if not summary:
        Path(out_html).write_text("<p>No implied-growth data yet.</p>", encoding="utf-8")
        return

    rows = []
    link = f'<a href="{tk.lower()}_growth.html">{tk}</a>'
    for gtype, stats in summary.items():
        for stat, val in stats.items():
            rows.append(
                {"Ticker": link, "Growth Type": gtype, "Statistic": stat, "Value": f"{val:.2%}"}
            )

    fwd_eps, fwd_pct = _get_forward_eps_info(tk)
    if fwd_eps is not None:
        rows.append(
            {"Ticker": link, "Growth Type": "—", "Statistic": "Forward EPS", "Value": f"{fwd_eps:.2f}"}
        )
        rows.append(
            {"Ticker": link, "Growth Type": "—", "Statistic": "Forward EPS %tile", "Value": f"{fwd_pct:.2f}"}
        )

    pd.DataFrame(rows).to_html(out_html, index=False, escape=False)

# ─── Public entrypoint (importable) ─────────────────────────
def render_index_growth_charts():
    print("[index_growth_charts] Building SPY & QQQ growth assets …")
    for tk in INDEXES:
        df      = _fetch_history(tk)
        summary = _compute_summary(df)
        _build_summary_html(summary, tk)
        _build_chart(df, summary, tk)
    print("[index_growth_charts] Done.")

# ─── CLI run ────────────────────────────────────────────────
if __name__ == "__main__":
    render_index_growth_charts()

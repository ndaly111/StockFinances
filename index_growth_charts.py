import os, sqlite3, pandas as pd, matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

DB_PATH     = "Stock Data.db"
TABLE       = "Index_Growth_History"
OUTPUT_DIR  = "charts"
INDEXES     = ["SPY", "QQQ"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

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

def get_percentile(value, series):
    series = [v for v in series if v is not None]
    if not series:
        return None
    count = sum(1 for v in series if v <= value)
    return round(100 * count / len(series), 2)

def _get_forward_eps_from_pe_history(ticker):
    with sqlite3.connect(DB_PATH) as conn:
        # Get latest market price
        price_df = pd.read_sql_query("SELECT last_price FROM MarketData WHERE ticker = ?", conn, params=(ticker,))
        if price_df.empty:
            return None, None
        price = price_df.iloc[0]["last_price"]

        # Get latest PE values for all tickers
        all_pe = pd.read_sql_query("""
            SELECT Ticker, MAX(Date) as MaxDate
            FROM Index_PE_History
            WHERE PE_Type = 'Forward'
            GROUP BY Ticker
        """, conn)

        pe_df = pd.read_sql_query("""
            SELECT Ticker, Date, PE_Ratio
            FROM Index_PE_History
            WHERE PE_Type = 'Forward'
        """, conn)

        # Get latest per ticker
        latest_pe = pd.merge(all_pe, pe_df, left_on=["Ticker", "MaxDate"], right_on=["Ticker", "Date"])
        latest_pe = latest_pe[["Ticker", "PE_Ratio"]].dropna()
        latest_pe["Ticker"] = latest_pe["Ticker"].str.upper()
        latest_pe = latest_pe[latest_pe["PE_Ratio"] > 0]

        # Calculate Forward EPS = Price / Forward PE
        latest_pe["Forward_EPS"] = latest_pe["Ticker"].map(
            lambda tk: _get_price(conn, tk)
        ) / latest_pe["PE_Ratio"]

        if ticker.upper() not in latest_pe["Ticker"].values:
            return None, None

        forward_eps = price / latest_pe[latest_pe["Ticker"] == ticker.upper()]["PE_Ratio"].values[0]
        percentile = get_percentile(forward_eps, latest_pe["Forward_EPS"].tolist())
        return forward_eps, percentile

def _get_price(conn, ticker):
    result = pd.read_sql_query("SELECT last_price FROM MarketData WHERE ticker = ?", conn, params=(ticker,))
    if result.empty:
        return None
    return result.iloc[0]["last_price"]

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

    forward_eps, forward_eps_pct = _get_forward_eps

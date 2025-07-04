# index_growth_charts.py
# --------------------------------------------------------------------
# Generates historical implied growth charts and summary tables
# for SPY and QQQ using the Index_Growth_History table
# --------------------------------------------------------------------

import sqlite3
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

DB_PATH = "Stock Data.db"
OUTPUT_DIR = "charts"
TABLE_NAME = "Index_Growth_History"
TICKERS = ["SPY", "QQQ"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

def fetch_growth_data(ticker):
    """Pull historical growth data for a given index"""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(f"""
        SELECT Date, Growth_Type, Implied_Growth
        FROM {TABLE_NAME}
        WHERE Ticker = ?
        ORDER BY Date ASC
    """, conn, params=(ticker,))
    conn.close()

    if df.empty:
        return None

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.pivot(index="Date", columns="Growth_Type", values="Implied_Growth")
    return df

def compute_summary(df):
    """Compute avg, median, min, max for TTM and Forward"""
    summary = {}
    for col in ["TTM", "Forward"]:
        if col in df:
            summary[col] = {
                "Average": df[col].mean(),
                "Median": df[col].median(),
                "Min": df[col].min(),
                "Max": df[col].max()
            }
        else:
            summary[col] = {}
    return summary

def plot_growth_chart(df, summary, ticker):
    """Plot TTM and Forward growth lines with summary stats"""
    fig, ax = plt.subplots(figsize=(10, 6))

    if "TTM" in df:
        ax.plot(df.index, df["TTM"], label="TTM Growth", color="blue", linewidth=2)
        for label, style in [("Average", ":"), ("Median", "--"), ("Min", "-."), ("Max", "-.")]:
            if label in summary["TTM"]:
                ax.axhline(y=summary["TTM"][label], color="blue", linestyle=style, linewidth=1)

    if "Forward" in df:
        ax.plot(df.index, df["Forward"], label="Forward Growth", color="green", linewidth=2)
        for label, style in [("Average", ":"), ("Median", "--"), ("Min", "-."), ("Max", "-.")]:
            if label in summary["Forward"]:
                ax.axhline(y=summary["Forward"][label], color="green", linestyle=style, linewidth=1)

    ax.set_title(f"{ticker} Implied Growth Rates Over Time")
    ax.set_ylabel("Implied Growth Rate")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    output_path = os.path.join(OUTPUT_DIR, f"{ticker.lower()}_growth_chart.png")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def render_summary_table(summary, ticker):
    """Save summary stats as an HTML table with linked Ticker column"""
    rows = []
    link = f'<a href="{ticker.lower()}_growth.html">{ticker}</a>'
    for col in ["TTM", "Forward"]:
        for stat in ["Average", "Median", "Min", "Max"]:
            value = summary.get(col, {}).get(stat)
            if value is not None:
                rows.append({
                    "Ticker": link,
                    "Growth Type": col,
                    "Statistic": stat,
                    "Value": f"{value:.2%}"
                })

    df = pd.DataFrame(rows)
    html = df.to_html(index=False, escape=False)
    output_path = os.path.join(OUTPUT_DIR, f"{ticker.lower()}_growth_summary.html")
    with open(output_path, "w") as f:
        f.write(html)

def render_index_growth_charts():
    for ticker in TICKERS:
        df = fetch_growth_data(ticker)
        if df is None or len(df) < 3:
            print(f"Not enough data to render chart for {ticker}")
            continue
        summary = compute_summary(df)
        plot_growth_chart(df, summary, ticker)
        render_summary_table(summary, ticker)
        print(f"Generated chart and summary for {ticker}")

# Mini-main
if __name__ == "__main__":
    render_index_growth_charts()

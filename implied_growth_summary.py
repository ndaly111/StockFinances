import os
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from statistics import mean, median, stdev
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
DB_PATH = "Stock Data.db"
OUTPUT_DIR = "charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TIME_WINDOWS_YEARS = {
    "1Y": 365,
    "3Y": 365 * 3,
    "5Y": 365 * 5,
    "10Y": 365 * 10,
}

MIN_REQUIRED_ENTRIES = 30

# -------------------------------------------------------------------------
# Step 1: Load Data
# -------------------------------------------------------------------------
def fetch_growth_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM Implied_Growth_History", conn, parse_dates=["date_recorded"])
    conn.close()
    return df

# -------------------------------------------------------------------------
# Step 2: Compute Summary Stats
# -------------------------------------------------------------------------
def summarize_growth(df, ticker, growth_type, today):
    subset = df[(df["ticker"] == ticker) & (df["growth_type"] == growth_type)].copy()
    if subset.empty:
        return {}

    subset.sort_values("date_recorded", inplace=True)
    current_row = subset[subset["date_recorded"] == today]
    current_val = current_row["growth_value"].iloc[0] if not current_row.empty else None

    summary = {}
    for label, days in TIME_WINDOWS_YEARS.items():
        cutoff = today - timedelta(days=days)
        window_df = subset[subset["date_recorded"] >= cutoff]

        if len(window_df) < MIN_REQUIRED_ENTRIES:
            summary[label] = {
                "avg": "N/A",
                "median": "N/A",
                "std_dev": "N/A",
                "percentile": "N/A",
                "current": f"{current_val:.2%}" if current_val else "N/A",
                "note": f"Only {len(window_df)} entries"
            }
            continue

        values = window_df["growth_value"].tolist()
        avg = mean(values)
        med = median(values)
        std = stdev(values) if len(values) > 1 else 0
        percentile = round(np.sum(np.array(values) <= current_val) / len(values) * 100) if current_val is not None else "N/A"

        summary[label] = {
            "avg": f"{avg:.2%}",
            "median": f"{med:.2%}",
            "std_dev": f"{std:.2%}",
            "percentile": f"{percentile}%",
            "current": f"{current_val:.2%}" if current_val else "N/A",
            "note": ""
        }

    return summary

# -------------------------------------------------------------------------
# Step 3: Render Summary HTML
# -------------------------------------------------------------------------
def render_summary_html(ticker, summaries):
    html = f"<h2 style='text-align:center;'>Implied Growth Summary for {ticker}</h2>\n"
    for growth_type, breakdown in summaries.items():
        html += f"<h3 style='text-align:center;'>{growth_type} Implied Growth</h3>\n"
        html += """
        <table border="1" cellspacing="0" cellpadding="6" style="margin:auto; text-align:center;">
        <tr>
            <th>Time Window</th>
            <th>Current</th>
            <th>Average</th>
            <th>Median</th>
            <th>Std Dev</th>
            <th>Percentile Rank</th>
            <th>Note</th>
        </tr>
        """
        for label, stats in breakdown.items():
            html += f"<tr><td>{label}</td><td>{stats['current']}</td><td>{stats['avg']}</td><td>{stats['median']}</td><td>{stats['std_dev']}</td><td>{stats['percentile']}</td><td>{stats['note']}</td></tr>"
        html += "</table><br><br>\n"

    file_path = os.path.join(OUTPUT_DIR, f"{ticker}_implied_growth_summary.html")
    with open(file_path, "w") as f:
        f.write(html)
    print(f"Saved summary: {file_path}")

# -------------------------------------------------------------------------
# Step 4: Generate Line Chart
# -------------------------------------------------------------------------
def generate_implied_growth_chart(df, ticker):
    df_ttm = df[(df["ticker"] == ticker) & (df["growth_type"] == "TTM")].copy()
    df_fwd = df[(df["ticker"] == ticker) & (df["growth_type"] == "Forward")].copy()

    if df_ttm.empty or df_fwd.empty:
        print(f"Not enough data for {ticker}, skipping chart.")
        return None

    df_ttm.sort_values("date_recorded", inplace=True)
    df_fwd.sort_values("date_recorded", inplace=True)

    dates = pd.to_datetime(df_ttm["date_recorded"])
    ttm_vals = df_ttm["growth_value"]
    fwd_vals = df_fwd.set_index("date_recorded").reindex(df_ttm["date_recorded"])["growth_value"]

    mean_val = ttm_vals.mean()
    median_val = ttm_vals.median()
    std_val = ttm_vals.std()

    upper_band = mean_val + std_val
    lower_band = mean_val - std_val

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates, ttm_vals, label="TTM Implied Growth", color="blue", linewidth=1.5)
    ax.plot(dates, fwd_vals, label="Forward Implied Growth", color="green", linewidth=1.5)

    ax.axhline(mean_val, color="gray", linestyle="--", linewidth=1, label="TTM Avg")
    ax.axhline(median_val, color="gray", linestyle=":", linewidth=1, label="TTM Median")
    ax.axhline(upper_band, color="lightgray", linestyle="-", linewidth=1, label="+1 Std Dev")
    ax.axhline(lower_band, color="lightgray", linestyle="-", linewidth=1, label="-1 Std Dev")

    ax.set_title(f"{ticker} Implied Growth History")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth Rate")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=45)
    ax.legend()
    plt.tight_layout()

    file_path = f"{OUTPUT_DIR}/{ticker}_implied_growth_plot.png"
    plt.savefig(file_path, dpi=150)
    plt.close()
    print(f"Saved chart: {file_path}")

# -------------------------------------------------------------------------
# Step 5: Main Routine
# -------------------------------------------------------------------------
def generate_all_summaries():
    df = fetch_growth_data()
    if df.empty:
        print("No data found.")
        return

    today = pd.to_datetime(datetime.today().strftime("%Y-%m-%d"))
    tickers = df["ticker"].unique()

    for ticker in tickers:
        summaries = {}
        for growth_type in ["TTM", "Forward"]:
            breakdown = summarize_growth(df, ticker, growth_type, today)
            if breakdown:
                summaries[growth_type] = breakdown
        if summaries:
            render_summary_html(ticker, summaries)
            generate_implied_growth_chart(df, ticker)

if __name__ == "__main__":
    generate_all_summaries()

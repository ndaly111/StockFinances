#implied_growth_summary.py
import os
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from statistics import mean, median, stdev
import numpy as np

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

def fetch_growth_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM Implied_Growth_History", conn, parse_dates=["date_recorded"])
    conn.close()
    return df

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
        percentile = round(np.percentile(values, np.searchsorted(sorted(values), current_val)) if current_val is not None else 0)

        summary[label] = {
            "avg": f"{avg:.2%}",
            "median": f"{med:.2%}",
            "std_dev": f"{std:.2%}",
            "percentile": f"{percentile}%",
            "current": f"{current_val:.2%}" if current_val else "N/A",
            "note": ""
        }

    return summary

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
    print(f"Saved: {file_path}")

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

if __name__ == "__main__":
    generate_all_summaries()

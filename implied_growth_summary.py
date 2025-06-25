import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
DB_PATH = 'Stock Data.db'
TABLE_NAME = 'Implied_Growth_History'
CHART_DIR = 'charts'
HTML_TABLE_PATH = os.path.join(CHART_DIR, 'implied_growth_summary_table.html')
CHART_PATH      = os.path.join(CHART_DIR, 'implied_growth_summary_chart.png')

TIME_FRAMES = {
    '1 Year':  365,
    '3 Years': 365 * 3,
    '5 Years': 365 * 5,
    '10 Years': 365 * 10,
}

# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------
def ensure_output_directory():
    os.makedirs(CHART_DIR, exist_ok=True)

def calculate_summary_stats(df, value_column):
    if df.empty:
        return {'Average': '-', 'Median': '-', 'Std Dev': '-', 'Current': '-', 'Percentile': '-'}
    vals = df[value_column]
    avg = vals.mean()
    med = vals.median()
    std = vals.std()
    curr = vals.iloc[-1]
    pct  = vals.rank(pct=True).iloc[-1] * 100
    return {
        'Average':   f"{avg:.2%}",
        'Median':    f"{med:.2%}",
        'Std Dev':   f"{std:.2%}",
        'Current':   f"{curr:.2%}",
        'Percentile': f"{pct:.1f}th"
    }

def generate_summary_table(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date_recorded'])
    now = datetime.now()

    rows = []
    for label, days in TIME_FRAMES.items():
        cutoff    = now - pd.Timedelta(days=days)
        window_df = df[df['date'] >= cutoff]
        for typ in ['TTM','Forward']:
            subset = window_df[window_df['growth_type'] == typ]
            stats  = calculate_summary_stats(subset, 'growth_value')
            rows.append({
                'Timeframe': label,
                'Type':       typ,
                **stats
            })

    summary_df = pd.DataFrame(rows)
    summary_df.to_html(HTML_TABLE_PATH, index=False, na_rep='-', justify='center')
    return HTML_TABLE_PATH

def plot_growth_chart(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date_recorded'])

    fig, ax = plt.subplots(figsize=(10,6))
    for typ, color in [('TTM','blue'),('Forward','green')]:
        series = df[df['growth_type']==typ].sort_values('date')
        if series.empty:
            continue
        ax.plot(series['date'], series['growth_value'],
                label=f'{typ} Implied Growth', color=color, linewidth=1.5)
        mean   = series['growth_value'].mean()
        median = series['growth_value'].median()
        std    = series['growth_value'].std()
        # stat lines
        ax.axhline(mean,   linestyle='--', label=f'{typ} Avg',    color=color, alpha=0.7)
        ax.axhline(median, linestyle=':',  label=f'{typ} Median', color=color, alpha=0.7)
        ax.axhline(mean+std, linestyle='-.', label=f'{typ} +1σ', color=color, alpha=0.5)
        ax.axhline(mean-std, linestyle='-.', label=f'{typ} -1σ', color=color, alpha=0.5)

    ax.set_title("Implied Growth Rates Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth Rate")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(loc='upper left', fontsize='small')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=150)
    plt.close()
    return CHART_PATH

def load_growth_data():
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)

# -------------------------------------------------------------------------
# Main entrypoint (exports generate_all_summaries)
# -------------------------------------------------------------------------
def generate_all_summaries():
    ensure_output_directory()
    df = load_growth_data()

    # No data case
    if df.empty:
        pd.DataFrame([{'Note':'No implied growth data available.'}]) \
          .to_html(HTML_TABLE_PATH, index=False)
        fig, ax = plt.subplots(figsize=(6,2))
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(CHART_PATH, dpi=100)
        plt.close()
        return HTML_TABLE_PATH, CHART_PATH

    # Generate table & chart
    html_path  = generate_summary_table(df)
    chart_path = plot_growth_chart(df)
    return html_path, chart_path

# Allow standalone run
if __name__ == "__main__":
    generate_all_summaries()

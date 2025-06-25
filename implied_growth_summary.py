import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Constants & Config
# ─────────────────────────────────────────────────────────────────────────────
DB_PATH        = 'Stock Data.db'
TABLE_NAME     = 'Implied_Growth_History'
CHART_DIR      = 'charts'
HTML_TABLE_PATH = os.path.join(CHART_DIR, 'implied_growth_summary_table.html')
CHART_PATH      = os.path.join(CHART_DIR, 'implied_growth_summary_chart.png')

TIME_FRAMES = {
    '1 Year':  365,
    '3 Years': 365 * 3,
    '5 Years': 365 * 5,
    '10 Years':365 * 10,
}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def ensure_output_directory():
    os.makedirs(CHART_DIR, exist_ok=True)

def load_growth_data():
    """Load all rows from Implied_Growth_History into a DataFrame."""
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)

def calculate_summary_stats(df, col):
    """Return a dict of stats (avg, med, std, current, percentile) or dashes."""
    if df.empty or col not in df:
        return {'Average':'-','Median':'-','Std Dev':'-','Current':'-','Percentile':'-'}
    vals = df[col].dropna()
    if vals.empty:
        return {'Average':'-','Median':'-','Std Dev':'-','Current':'-','Percentile':'-'}
    avg = vals.mean(); med = vals.median(); std = vals.std()
    cur = vals.iloc[-1]
    pct = vals.rank(pct=True).iloc[-1] * 100
    return {
        'Average':   f"{avg:.2%}",
        'Median':    f"{med:.2%}",
        'Std Dev':   f"{std:.2%}",
        'Current':   f"{cur:.2%}",
        'Percentile':f"{pct:.1f}th"
    }

# ─────────────────────────────────────────────────────────────────────────────
# Rendering
# ─────────────────────────────────────────────────────────────────────────────
def generate_summary_table(df):
    """Generate the HTML summary table for each timeframe and TTM/Forward."""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date_recorded'])
    now = datetime.now()
    rows = []
    for label, days in TIME_FRAMES.items():
        cutoff = now - pd.Timedelta(days=days)
        window = df[df['date'] >= cutoff]
        for typ in ['TTM','Forward']:
            subset = window[window['growth_type']==typ]
            stats = calculate_summary_stats(subset, 'growth_value')
            rows.append({
                'Timeframe': label,
                'Type':       typ,
                **stats
            })
    summary_df = pd.DataFrame(rows)
    summary_df.to_html(HTML_TABLE_PATH, index=False, na_rep='-', justify='center')
    return HTML_TABLE_PATH

def plot_growth_chart(df):
    """Plot both TTM & Forward lines plus mean/median/±1σ for each."""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date_recorded'])

    fig, ax = plt.subplots(figsize=(10,6))
    for typ, color in [('TTM','blue'),('Forward','green')]:
        series = df[df['growth_type']==typ].sort_values('date')
        if series.empty: continue
        ax.plot(series['date'], series['growth_value'],
                label=f'{typ} Growth', color=color, linewidth=1.5)
        m = series['growth_value'].mean()
        med = series['growth_value'].median()
        s = series['growth_value'].std()
        # draw horizontal stat lines
        ax.axhline(m,    linestyle='--', label=f'{typ} Avg',    color=color, alpha=0.7)
        ax.axhline(med,  linestyle=':',  label=f'{typ} Median', color=color, alpha=0.7)
        ax.axhline(m+s,  linestyle='-.', label=f'{typ} +1σ',    color=color, alpha=0.5)
        ax.axhline(m-s,  linestyle='-.', label=f'{typ} -1σ',    color=color, alpha=0.5)

    ax.set_title("Implied Growth Rates Over Time")
    ax.set_xlabel("Date"); ax.set_ylabel("Growth Rate")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(loc='upper left', fontsize='small')
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=150)
    plt.close()
    return CHART_PATH

# ─────────────────────────────────────────────────────────────────────────────
# Public Entrypoint: generate_all_summaries
# ─────────────────────────────────────────────────────────────────────────────
def generate_all_summaries():
    """
    Main function to be imported by main_remote.py:
      1) Ensures output dir
      2) Loads data
      3) Writes summary HTML and chart PNG
      4) Returns (html_path, chart_path)
    """
    ensure_output_directory()
    df = load_growth_data()
    if df.empty:
        # No-data fallback
        pd.DataFrame([{'Note':'No data available.'}]) \
          .to_html(HTML_TABLE_PATH, index=False)
        fig, ax = plt.subplots(figsize=(6,2))
        ax.text(0.5,0.5,'No data available',ha='center',va='center')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(CHART_PATH, dpi=100)
        plt.close()
        return HTML_TABLE_PATH, CHART_PATH

    html = generate_summary_table(df)
    chart = plot_growth_chart(df)
    return html, chart

# ─────────────────────────────────────────────────────────────────────────────
# Allow standalone execution
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    generate_all_summaries()

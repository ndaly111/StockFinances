
import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

# Constants
DB_PATH = 'Stock Data.db'
TABLE_NAME = 'ImpliedGrowthRates'
CHART_DIR = 'charts'
CHART_PATH = os.path.join(CHART_DIR, 'implied_growth_summary_chart.png')
HTML_TABLE_PATH = os.path.join(CHART_DIR, 'implied_growth_summary_table.html')

TIME_FRAMES = {
    '1 Year': 365,
    '3 Years': 365 * 3,
    '5 Years': 365 * 5,
    '10 Years': 365 * 10
}

def ensure_output_directory():
    os.makedirs(CHART_DIR, exist_ok=True)

def calculate_summary_stats(df, value_column):
    if df.empty:
        return {'Average': '-', 'Median': '-', 'Std Dev': '-', 'Current': '-', 'Percentile': '-'}
    stats = {}
    stats['Average'] = f"{df[value_column].mean():.2%}"
    stats['Median'] = f"{df[value_column].median():.2%}"
    stats['Std Dev'] = f"{df[value_column].std():.2%}"
    stats['Current'] = f"{df[value_column].iloc[-1]:.2%}"
    stats['Percentile'] = f"{(df[value_column].rank(pct=True).iloc[-1] * 100):.1f}th"
    return stats

def generate_summary_table(df):
    df['date'] = pd.to_datetime(df['date'])
    now = datetime.now()

    table_rows = []
    for label, days in TIME_FRAMES.items():
        cutoff = now - pd.Timedelta(days=days)
        recent_df = df[df['date'] >= cutoff]
        for rate_type in ['TTM', 'Forward']:
            subset = recent_df[recent_df['rate_type'] == rate_type]
            stats = calculate_summary_stats(subset, 'growth_rate')
            table_rows.append({
                'Timeframe': label,
                'Type': rate_type,
                **stats
            })

    summary_df = pd.DataFrame(table_rows)
    summary_df.to_html(HTML_TABLE_PATH, index=False, na_rep='-', justify='center')
    return HTML_TABLE_PATH

def plot_growth_chart(df):
    df['date'] = pd.to_datetime(df['date'])
    fig, ax = plt.subplots(figsize=(10, 6))

    for rate_type, color in [('TTM', 'blue'), ('Forward', 'green')]:
        subset = df[df['rate_type'] == rate_type].sort_values('date')
        if subset.empty:
            continue
        ax.plot(subset['date'], subset['growth_rate'], label=f'{rate_type} Implied Growth', color=color)
        mean = subset['growth_rate'].mean()
        std = subset['growth_rate'].std()
        median = subset['growth_rate'].median()
        for ref, style, val in [('Avg', '--', mean), ('Median', ':', median),
                                ('+1 Std Dev', '-', mean + std), ('-1 Std Dev', '-', mean - std)]:
            ax.axhline(y=val, linestyle=style, linewidth=1, label=f'{rate_type} {ref}', alpha=0.5, color=color)

    ax.set_title("Implied Growth Rates Over Time")
    ax.set_ylabel("Growth Rate")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(CHART_PATH)
    plt.close()
    return CHART_PATH

def load_growth_data():
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(f"SELECT * FROM {TABLE_NAME};", conn)

def generate_implied_growth_summary():
    ensure_output_directory()
    df = load_growth_data()

    if df.empty:
        pd.DataFrame([{'Note': 'No implied growth data available.'}]).to_html(HTML_TABLE_PATH, index=False)
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=12)
        ax.axis('off')
        plt.savefig(CHART_PATH)
        plt.close()
        return HTML_TABLE_PATH, CHART_PATH

    html_path = generate_summary_table(df)
    chart_path = plot_growth_chart(df)
    return html_path, chart_path

# Run directly
if __name__ == "__main__":
    generate_implied_growth_summary()

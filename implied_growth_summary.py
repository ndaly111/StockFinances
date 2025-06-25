import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
DB_PATH        = 'Stock Data.db'
TABLE_NAME     = 'Implied_Growth_History'
CHART_DIR      = 'charts'
HTML_TEMPLATE  = os.path.join(CHART_DIR, '{ticker}_implied_growth_summary.html')
CHART_TEMPLATE = os.path.join(CHART_DIR, '{ticker}_implied_growth_plot.png')

TIME_FRAMES = {
    '1 Year':   365,
    '3 Years':  365 * 3,
    '5 Years':  365 * 5,
    '10 Years': 365 * 10,
}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def ensure_output_directory():
    os.makedirs(CHART_DIR, exist_ok=True)

def load_growth_data():
    """
    Load the Implied_Growth_History table into a DataFrame,
    or return empty if missing.
    """
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    with sqlite3.connect(DB_PATH) as conn:
        try:
            return pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)
        except sqlite3.OperationalError:
            return pd.DataFrame()

def calculate_summary_stats(df: pd.DataFrame, col: str) -> dict:
    """
    Compute avg, median, std, current, percentile (%) for a column.
    Returns '-' placeholders if no data.
    """
    keys = ['Average','Median','Std Dev','Current','Percentile']
    if df.empty or col not in df:
        return dict.fromkeys(keys, '-')
    vals = df[col].dropna()
    if vals.empty:
        return dict.fromkeys(keys, '-')
    avg  = vals.mean()
    med  = vals.median()
    std  = vals.std()
    cur  = vals.iloc[-1]
    pct  = vals.rank(pct=True).iloc[-1] * 100
    return {
        'Average':    f"{avg:.2%}",
        'Median':     f"{med:.2%}",
        'Std Dev':    f"{std:.2%}",
        'Current':    f"{cur:.2%}",
        'Percentile': f"{pct:.1f}%"
    }

# ─────────────────────────────────────────────────────────────────────────────
# Rendering per‐ticker
# ─────────────────────────────────────────────────────────────────────────────
def generate_summary_table(df: pd.DataFrame, ticker: str) -> str:
    """
    Writes an HTML summary table for one ticker.
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date_recorded'])
    now = datetime.now()

    rows = []
    for label, days in TIME_FRAMES.items():
        cutoff = now - pd.Timedelta(days=days)
        window = df[df['date'] >= cutoff]
        for typ in ['TTM','Forward']:
            subset = window[window['growth_type'] == typ]
            stats  = calculate_summary_stats(subset, 'growth_value')
            row = {'Timeframe': label, 'Type': typ}
            row.update(stats)
            rows.append(row)

    summary_df = pd.DataFrame(rows)
    path = HTML_TEMPLATE.format(ticker=ticker)
    summary_df.to_html(path, index=False, na_rep='-', justify='center')
    return path

def plot_growth_chart(df: pd.DataFrame, ticker: str) -> str:
    """
    Saves a PNG plot of TTM & Forward implied-growth lines plus stat lines.
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date_recorded'])

    fig, ax = plt.subplots(figsize=(10,6))
    for typ, color in [('TTM','blue'), ('Forward','green')]:
        series = df[df['growth_type'] == typ].sort_values('date')
        if series.empty:
            continue
        ax.plot(series['date'], series['growth_value'],
                label=f"{typ} Growth",
                color=color, linewidth=1.5)
        m, med, s = (series['growth_value'].mean(),
                     series['growth_value'].median(),
                     series['growth_value'].std())
        ax.axhline(m,   linestyle='--', label=f'{typ} Avg',    color=color, alpha=0.7)
        ax.axhline(med, linestyle=':',  label=f'{typ} Median', color=color, alpha=0.7)
        ax.axhline(m+s, linestyle='-.', label=f'{typ} +1σ',    color=color, alpha=0.5)
        ax.axhline(m-s, linestyle='-.', label=f'{typ} -1σ',    color=color, alpha=0.5)

    ax.set_title("Implied Growth Rates Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth Rate")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(loc='upper left', fontsize='small')
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()

    path = CHART_TEMPLATE.format(ticker=ticker)
    plt.savefig(path, dpi=150)
    plt.close()
    return path

# ─────────────────────────────────────────────────────────────────────────────
# Public Entrypoint
# ─────────────────────────────────────────────────────────────────────────────
def generate_all_summaries():
    """
    Loop through each ticker in the history table,
    generating both HTML and PNG for each.
    """
    ensure_output_directory()
    df_all = load_growth_data()
    if 'ticker' not in df_all.columns:
        return
    for ticker in df_all['ticker'].unique():
        df_t = df_all[df_all['ticker'] == ticker]
        generate_summary_table(df_t, ticker)
        plot_growth_chart(df_t, ticker)

# ─────────────────────────────────────────────────────────────────────────────
# Standalone Execution
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    generate_all_summaries()

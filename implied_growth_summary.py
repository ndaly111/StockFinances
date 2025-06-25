import os
import sqlite3
import numpy as np                       # NEW
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# ───────────────────────────────────────────────────────────
# Configuration
# ───────────────────────────────────────────────────────────
DB_PATH        = 'Stock Data.db'
TABLE_NAME     = 'Implied_Growth_History'
CHARTS_DIR     = 'charts'
HTML_TEMPLATE  = os.path.join(CHARTS_DIR, '{ticker}_implied_growth_summary.html')
CHART_TEMPLATE = os.path.join(CHARTS_DIR, '{ticker}_implied_growth_plot.png')

TIME_FRAMES = {
    '1 Year':   365,
    '3 Years':  365 * 3,
    '5 Years':  365 * 5,
    '10 Years': 365 * 10,
}

# ───────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────
def ensure_output_directory():
    os.makedirs(CHARTS_DIR, exist_ok=True)

def load_growth_data():
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    with sqlite3.connect(DB_PATH) as conn:
        try:
            return pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)
        except (sqlite3.OperationalError, pd.errors.DatabaseError) as e:
            print(f"[load_growth_data] skipping load ({e})")
            return pd.DataFrame()

def clean_series(series: pd.Series) -> pd.Series:
    return series.apply(
        lambda x: x if isinstance(x, (int, float)) and not isinstance(x, complex) else None
    ).dropna()

def calculate_summary_stats(series: pd.Series):
    if series.empty:
        return '-', '-', '-', '-', '-'
    avg = series.mean()
    med = series.median()
    std = series.std()
    cur = series.iloc[-1]
    pct = series.rank(pct=True).iloc[-1] * 100
    # guard against NaN std
    std_fmt = f"{std:.2%}" if pd.notnull(std) else '–'
    return (f"{avg:.2%}", f"{med:.2%}", std_fmt, f"{cur:.2%}", f"{pct:.1f}%")

def write_placeholder_html(ticker: str):
    content = """
    <style>
        .placeholder {
            text-align: center;
            font-family: Arial, sans-serif;
            font-size: 16px;
            padding: 40px;
            color: #888;
        }
    </style>
    <div class="placeholder">No data available for Implied Growth Summary</div>
    """
    out_path = HTML_TEMPLATE.format(ticker=ticker)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(content)
    return out_path

# ───────────────────────────────────────────────────────────
# Per-Ticker Rendering
# ───────────────────────────────────────────────────────────
def generate_summary_table(df: pd.DataFrame, ticker: str) -> str:
    df = df.copy()
    df['date'] = pd.to_datetime(df['date_recorded'])
    now = datetime.now()

    rows = []
    for label, days in TIME_FRAMES.items():
        cutoff = now - pd.Timedelta(days=days)
        window = df[df['date'] >= cutoff]
        for typ in ['TTM', 'Forward']:
            vals = clean_series(window[window['growth_type'] == typ]['growth_value'])
            avg, med, std, cur, pct = calculate_summary_stats(vals)
            rows.append({
                'Timeframe': label,
                'Type':       typ,
                'Average':    avg,
                'Median':     med,
                'Std Dev':    std,
                'Current':    cur,
                'Percentile': pct
            })

    if not any(r['Average'] != '-' for r in rows):
        print(f"[generate_summary_table] No usable data for {ticker}, writing placeholder.")
        return write_placeholder_html(ticker)

    # ---------- PIVOT & CLEAN ----------
    df_rows = pd.DataFrame(rows)

    # pivot so each timeframe is one row, metrics grouped, TTM/Fwd side-by-side
    pivot = (df_rows
             .pivot(index='Timeframe', columns='Type')
             .swaplevel(axis=1)           # put TTM/Fwd after metric for sane ordering
             .sort_index(axis=1, level=0))

    ordered_metrics = ['Average', 'Median', 'Std Dev', 'Current', 'Percentile']
    pivot = pivot[ordered_metrics]        # keep desired metric order

    # replace NaNs / 'nan%' with an en-dash
    pivot = (pivot
             .replace({'nan%': '–'})
             .replace({np.nan: '–'}))

    out_path = HTML_TEMPLATE.format(ticker=ticker)
    pivot.to_html(
        out_path,
        index=True,
        na_rep='–',
        justify='center',
        classes='table table-striped implied-growth-table'  # extra CSS hook
    )
    return out_path

def plot_growth_chart(df: pd.DataFrame, ticker: str) -> str:
    df = df.copy()
    df['date'] = pd.to_datetime(df['date_recorded'])
    fig, ax = plt.subplots(figsize=(10, 6))

    found_data = False
    for typ, color in [('TTM', 'blue'), ('Forward', 'green')]:
        series = df[df['growth_type'] == typ].sort_values('date')
        series['growth_value'] = clean_series(series['growth_value'])
        if series.empty:
            continue
        found_data = True
        ax.plot(series['date'], series['growth_value'],
                label=f"{typ} Growth", color=color, linewidth=1.5)
        m, med, s = (series['growth_value'].mean(),
                     series['growth_value'].median(),
                     series['growth_value'].std())
        ax.axhline(m,   ls='--', label=f"{typ} Avg",    color=color, alpha=0.7)
        ax.axhline(med, ls=':',  label=f"{typ} Median", color=color, alpha=0.7)
        ax.axhline(m+s, ls='-.', label=f"{typ} +1σ",    color=color, alpha=0.5)
        ax.axhline(m-s, ls='-.', label=f"{typ} -1σ",    color=color, alpha=0.5)

    if not found_data:
        print(f"[plot_growth_chart] No valid data to plot for {ticker}")
        return ""

    ax.set_title("Implied Growth Rates Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth Rate")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _:
                                                   f"{y:.0%}"))
    ax.legend(loc='upper left', fontsize='small')
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()

    out_path = CHART_TEMPLATE.format(ticker=ticker)
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path

# ───────────────────────────────────────────────────────────
# Public Entrypoint - Mini-main
# ───────────────────────────────────────────────────────────
def generate_all_summaries():
    """Generate HTML + PNG implied-growth summaries for every ticker."""
    ensure_output_directory()
    df_all = load_growth_data()
    if 'ticker' not in df_all.columns:
        print("[generate_all_summaries] No data found.")
        return
    for ticker in df_all['ticker'].unique():
        df_t = df_all[df_all['ticker'] == ticker]
        print(f"[generate_all_summaries] Processing {ticker}")
        generate_summary_table(df_t, ticker)
        plot_growth_chart(df_t, ticker)

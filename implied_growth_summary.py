# implied_growth_summary.py
# -----------------------------------------------------------
# Build per-ticker “Implied Growth Summary” HTML table + plot
# -----------------------------------------------------------
import os
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# ───────────────────────────────────
# Configuration
# ───────────────────────────────────
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

# ───────────────────────────────────
# Helpers
# ───────────────────────────────────
def ensure_output_directory():
    os.makedirs(CHARTS_DIR, exist_ok=True)

def load_growth_data() -> pd.DataFrame:
    """Load the entire Implied_Growth_History table (or empty df)."""
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    with sqlite3.connect(DB_PATH) as conn:
        try:
            return pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)
        except (sqlite3.OperationalError, pd.errors.DatabaseError) as e:
            print(f"[load_growth_data] skipping load ({e})")
            return pd.DataFrame()

def clean_series(series: pd.Series) -> pd.Series:
    """Remove NaN / complex / non-numeric values."""
    return (
        series.apply(
            lambda x: x
            if isinstance(x, (int, float)) and not isinstance(x, complex)
            else None
        )
        .dropna()
    )

def calculate_summary_stats(series: pd.Series):
    """Return (avg, med, std, cur, pct) – nicely formatted, or dashes."""
    if series.empty:
        return ('–',) * 5
    avg = series.mean()
    med = series.median()
    std = series.std()
    cur = series.iloc[-1]
    pct = series.rank(pct=True).iloc[-1] * 100

    std_fmt = f"{std:.2%}" if pd.notnull(std) else '–'
    return (
        f"{avg:.2%}",
        f"{med:.2%}",
        std_fmt,
        f"{cur:.2%}",
        f"{pct:.1f}%",
    )

def write_placeholder_html(ticker: str):
    """Write a simple placeholder when no data exists."""
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

# ───────────────────────────────────
# Per-ticker HTML table
# ───────────────────────────────────
def generate_summary_table(df: pd.DataFrame, ticker: str) -> str:
    """
    Build & save the HTML stats table for one ticker.
    Returns output path (or placeholder path).
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date_recorded'])

    now = datetime.now()
    rows = []
    for label, days in TIME_FRAMES.items():
        cutoff = now - pd.Timedelta(days=days)
        window = df[df['date'] >= cutoff]
        for typ in ('TTM', 'Forward'):
            vals = clean_series(window[window['growth_type'] == typ]['growth_value'])
            avg, med, std, cur, pct = calculate_summary_stats(vals)
            rows.append(
                {
                    'Timeframe': label,
                    'Type': typ,
                    'Average': avg,
                    'Median': med,
                    'Std Dev': std,
                    'Current': cur,
                    'Percentile': pct,
                }
            )

    if not any(r['Average'] != '–' for r in rows):
        print(f"[generate_summary_table] No usable data for {ticker}.")
        return write_placeholder_html(ticker)

    # ---------- pivot to prettier layout ----------
    df_rows = pd.DataFrame(rows).set_index(['Timeframe', 'Type'])

    # columns become MultiIndex (metric, TTM/Forward)
    pivot = df_rows.unstack('Type')           # outer = metric, inner = type
    pivot = pivot.swaplevel(axis=1)           # centre metrics together

    # enforce metric order only for metrics that exist
    ordered = ['Average', 'Median', 'Std Dev', 'Current', 'Percentile']
    existing = [m for m in ordered if m in pivot.columns.get_level_values(0)]
    pivot = pivot[existing]

    # clean NaNs / "nan%" → en-dash
    pivot = pivot.replace({'nan%': '–', np.nan: '–'})

    out_path = HTML_TEMPLATE.format(ticker=ticker)
    pivot.to_html(
        out_path,
        index=True,
        na_rep='–',
        justify='center',
        classes='table table-striped implied-growth-table',
    )
    return out_path

# ───────────────────────────────────
# Per-ticker line plot
# ───────────────────────────────────
def plot_growth_chart(df: pd.DataFrame, ticker: str) -> str:
    df = df.copy()
    df['date'] = pd.to_datetime(df['date_recorded'])
    fig, ax = plt.subplots(figsize=(10, 6))

    found = False
    for typ, color in (('TTM', 'blue'), ('Forward', 'green')):
        series = (
            df[df['growth_type'] == typ]
            .sort_values('date')
            .assign(growth_value=lambda d: clean_series(d['growth_value']))
        )
        if series['growth_value'].empty:
            continue

        found = True
        ax.plot(
            series['date'],
            series['growth_value'],
            label=f"{typ} Growth",
            color=color,
            linewidth=1.5,
        )

        m, med, s = (
            series['growth_value'].mean(),
            series['growth_value'].median(),
            series['growth_value'].std(),
        )
        ax.axhline(m, ls='--', label=f"{typ} Avg", color=color, alpha=0.7)
        ax.axhline(med, ls=':', label=f"{typ} Median", color=color, alpha=0.7)
        ax.axhline(m + s, ls='-.', label=f"{typ} +1σ", color=color, alpha=0.5)
        ax.axhline(m - s, ls='-.', label=f"{typ} -1σ", color=color, alpha=0.5)

    if not found:
        print(f"[plot_growth_chart] No valid data for {ticker}")
        return ""

    ax.set_title("Implied Growth Rates Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth Rate")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(loc='upper left', fontsize='small')
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()

    out_path = CHART_TEMPLATE.format(ticker=ticker)
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path

# ───────────────────────────────────
# mini-main – batch regenerate
# ───────────────────────────────────
def generate_all_summaries():
    """Generate HTML + PNG implied-growth summaries for every ticker."""
    ensure_output_directory()
    df_all = load_growth_data()
    if 'ticker' not in df_all.columns:
        print("[generate_all_summaries] No Implied_Growth_History table found.")
        return

    for ticker in df_all['ticker'].unique():
        print(f"[generate_all_summaries] Processing {ticker}")
        df_t = df_all[df_all['ticker'] == ticker]
        generate_summary_table(df_t, ticker)
        plot_growth_chart(df_t, ticker)


# Allow manual execution
if __name__ == "__main__":
    generate_all_summaries()
    

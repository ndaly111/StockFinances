# implied_growth_summary.py
# -----------------------------------------------------------
# Build per-ticker “Implied Growth Summary” HTML table + plot
# -----------------------------------------------------------
import os, sqlite3, numpy as np, pandas as pd, matplotlib.pyplot as plt
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
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    with sqlite3.connect(DB_PATH) as conn:
        try:
            return pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)
        except (sqlite3.OperationalError, pd.errors.DatabaseError):
            return pd.DataFrame()

def clean_series(s: pd.Series) -> pd.Series:
    return s.apply(lambda x: x if isinstance(x, (int, float)) else None).dropna()

def calc_stats(s: pd.Series):
    if s.empty:
        return ('–',) * 5
    avg, med, std = s.mean(), s.median(), s.std()
    cur = s.iloc[-1]
    pct = s.rank(pct=True).iloc[-1] * 100
    return (f"{avg:.2%}", f"{med:.2%}",
            f"{std:.2%}" if pd.notnull(std) else '–',
            f"{cur:.2%}", f"{pct:.1f}%")

def write_placeholder(tkr: str) -> str:
    html = ('<div style="text-align:center;padding:40px;'
            'font-family:Arial;color:#888">'
            'No data available for Implied Growth Summary</div>')
    out = HTML_TEMPLATE.format(ticker=tkr)
    with open(out, 'w', encoding='utf-8') as f:
        f.write(html)
    return out

# ───────────────────────────────────
# Per-ticker SUMMARY TABLE
# ───────────────────────────────────
def generate_summary_table(df: pd.DataFrame, ticker: str) -> str:
    df = df.copy()
    df['date'] = pd.to_datetime(df['date_recorded'])

    rows, now = [], datetime.now()
    for label, days in TIME_FRAMES.items():
        win = df[df['date'] >= now - pd.Timedelta(days=days)]
        for typ in ('TTM', 'Forward'):
            stats = calc_stats(clean_series(win.loc[win['growth_type'] == typ, 'growth_value']))
            rows.append(dict(Timeframe=label, Type=typ,
                             Average=stats[0], Median=stats[1],
                             **{'Std Dev': stats[2], 'Current': stats[3], 'Percentile': stats[4]}))

    if not any(r['Average'] != '–' for r in rows):
        return write_placeholder(ticker)

    # -------- pivot (metric 1st, TTM/Forward 2nd) ----------
    tbl = (pd.DataFrame(rows)
           .set_index(['Timeframe', 'Type'])   # multi-index rows
           .unstack('Type'))                   # columns: (metric, TTM/Fwd)

    wanted = ['Average', 'Median', 'Std Dev', 'Current', 'Percentile']
    for metric in wanted:                      # ensure every sub-column exists
        for col in ('TTM', 'Forward'):
            if (metric, col) not in tbl.columns:
                tbl[(metric, col)] = '–'

    tbl = tbl[wanted]                          # enforce order
    tbl.sort_index(axis=1, level=[0, 1], inplace=True)
    tbl.replace({'nan%': '–', np.nan: '–'}, inplace=True)

    out = HTML_TEMPLATE.format(ticker=ticker)
    tbl.to_html(out, index=True, na_rep='–',
                justify='center',
                classes='table table-striped implied-growth-table')
    return out

# ───────────────────────────────────
# Per-ticker LINE PLOT (unchanged)
# ───────────────────────────────────
def plot_growth_chart(df: pd.DataFrame, ticker: str) -> str:
    df = df.copy()
    df['date'] = pd.to_datetime(df['date_recorded'])
    fig, ax = plt.subplots(figsize=(10, 6))
    have_data = False

    for typ, color in (('TTM', 'blue'), ('Forward', 'green')):
        s = (df[df['growth_type'] == typ]
             .sort_values('date')
             .assign(growth_value=lambda d: clean_series(d['growth_value'])))
        if s['growth_value'].empty:
            continue

        have_data = True
        ax.plot(s['date'], s['growth_value'], label=f"{typ} Growth",
                color=color, lw=1.5)

        m, med, sd = s['growth_value'].mean(), s['growth_value'].median(), s['growth_value'].std()
        for y, ls in ((m, '--'), (med, ':'), (m + sd, '-.'), (m - sd, '-.')):
            ax.axhline(y, ls=ls, color=color, alpha=.6)

    if not have_data:
        return ""

    ax.set(title="Implied Growth Rates Over Time",
           xlabel="Date", ylabel="Growth Rate")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(loc='upper left', fontsize='small')
    ax.grid(True, ls='--', alpha=.4)
    plt.tight_layout()

    out = CHART_TEMPLATE.format(ticker=ticker)
    plt.savefig(out, dpi=150)
    plt.close()
    return out

# ───────────────────────────────────
# mini-main – batch regenerate
# ───────────────────────────────────
def generate_all_summaries():
    ensure_output_directory()
    df_all = load_growth_data()
    if 'ticker' not in df_all.columns:
        print("[generate_all_summaries] Implied_Growth_History table missing.")
        return
    for tkr in df_all['ticker'].unique():
        print(f"[generate_all_summaries] {tkr}")
        dft = df_all[df_all['ticker'] == tkr]
        generate_summary_table(dft, tkr)
        plot_growth_chart(dft, tkr)

if __name__ == "__main__":
    generate_all_summaries()

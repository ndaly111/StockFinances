#!/usr/bin/env python3
# implied_growth_summary.py
# -----------------------------------------------------------
# • Builds per-ticker Implied-Growth summary HTML + plot
# • Back-fills helper table Index_Growth_Pctile
# -----------------------------------------------------------
import os, sqlite3, numpy as np, pandas as pd, matplotlib.pyplot as plt
from datetime import datetime
from itertools import product

# ──────────────────────────
# Configuration
# ──────────────────────────
DB_PATH        = 'Stock Data.db'
TABLE_NAME     = 'Implied_Growth_History'
PCT_TABLE      = 'Index_Growth_Pctile'       # helper table

CHARTS_DIR     = 'charts'
HTML_TEMPLATE  = os.path.join(CHARTS_DIR, '{ticker}_implied_growth_summary.html')
CHART_TEMPLATE = os.path.join(CHARTS_DIR, '{ticker}_implied_growth_plot.png')

TIME_FRAMES = {
    '1 Year':   365,
    '3 Years':  365 * 3,
    '5 Years':  365 * 5,
    '10 Years': 365 * 10,
}

ROW_ORDER   = ['1 Year', '3 Years', '5 Years', '10 Years']
COL_METRICS = ['Average', 'Median', 'Std Dev', 'Current', 'Percentile']
COL_TYPES   = ['TTM', 'Forward']        # TTM first

# ──────────────────────────
# Helpers
# ──────────────────────────
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

# ──────────────────────────
# Percentile back-fill (fixed)
# ──────────────────────────
def update_percentile_table(df: pd.DataFrame):
    """Populate/refresh Index_Growth_Pctile for all rows in df."""
    if df.empty:
        return

    # Ensure a datetime column once; keep original text as fallback
    df["_date_dt"] = pd.to_datetime(df["date_recorded"], errors="coerce")

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(f"""CREATE TABLE IF NOT EXISTS {PCT_TABLE} (
                           Date TEXT, Ticker TEXT, Growth_Type TEXT, Percentile REAL,
                           PRIMARY KEY (Date,Ticker,Growth_Type));""")

        rows = []
        for (ticker, gtype), sub in df.groupby(['ticker', 'growth_type']):
            s = sub.sort_values("_date_dt")['growth_value'].dropna()
            if s.empty:
                continue

            pct_series = s.rank(pct=True) * 100
            # Safe date: parsed dt if available else raw string
            dates = sub["_date_dt"].dt.strftime("%Y-%m-%d").where(
                        sub["_date_dt"].notna(), sub["date_recorded"])

            rows.extend([
                (d, ticker, gtype, round(p, 1))
                for d, p in zip(dates, pct_series)
            ])

        if rows:
            conn.executemany(f"INSERT OR REPLACE INTO {PCT_TABLE} VALUES (?,?,?,?)", rows)
            conn.commit()
            print(f"[update_percentile_table] {len(rows)} rows upserted into {PCT_TABLE}")

# ──────────────────────────
# Summary-table generator
# ──────────────────────────
def generate_summary_table(df: pd.DataFrame, ticker: str) -> str:
    df = df.copy()
    df['date'] = pd.to_datetime(df['date_recorded'])

    rows, now = [], datetime.now()
    for label, days in TIME_FRAMES.items():
        win = df[df['date'] >= now - pd.Timedelta(days=days)]
        for typ in COL_TYPES:
            stats = calc_stats(clean_series(
                win.loc[win['growth_type'] == typ, 'growth_value']
            ))
            rows.append(dict(Timeframe=label, Type=typ,
                             Average=stats[0], Median=stats[1],
                             **{'Std Dev': stats[2],
                                'Current': stats[3],
                                'Percentile': stats[4]}))

    if not any(r['Average'] != '–' for r in rows):
        return write_placeholder(ticker)

    pivot = (pd.DataFrame(rows)
             .set_index(['Timeframe', 'Type'])
             .unstack('Type'))

    desired_cols = pd.MultiIndex.from_tuples(
        [(m, t) for m, t in product(COL_METRICS, COL_TYPES)]
    )
    pivot = pivot.reindex(columns=desired_cols, fill_value='–')
    pivot = pivot.reindex(ROW_ORDER)
    pivot.replace({'nan%': '–', np.nan: '–'}, inplace=True)

    out = HTML_TEMPLATE.format(ticker=ticker)
    pivot.to_html(out, index=True, na_rep='–', justify='center',
                  classes='table table-striped implied-growth-table')
    return out

# ──────────────────────────
# Line-plot generator
# ──────────────────────────
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
        ax.plot(s['date'], s['growth_value'],
                label=f"{typ} Growth", color=color, lw=1.5)
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

# ──────────────────────────
# Batch regenerate
# ──────────────────────────
def generate_all_summaries():
    ensure_output_directory()
    df_all = load_growth_data()
    if 'ticker' not in df_all.columns:
        print("[generate_all_summaries] Implied_Growth_History table missing.")
        return

    update_percentile_table(df_all)

    for tkr in df_all['ticker'].unique():
        print(f"[generate_all_summaries] {tkr}")
        dft = df_all[df_all['ticker'] == tkr]
        generate_summary_table(dft, tkr)
        plot_growth_chart(dft, tkr)

if __name__ == "__main__":
    generate_all_summaries()

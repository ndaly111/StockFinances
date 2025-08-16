#!/usr/bin/env python3
"""
generate_index_growth_pages.py — Create interactive SPY/QQQ valuation pages

This module reads historical implied growth and P/E data from your SQLite
database and produces self‑contained HTML pages for SPY and QQQ that
feature interactive Plotly charts.  The pages are written to the
filenames specified in OUTPUT_FILES, preserving your existing URL
structure (e.g., `qqq_growth.html` and `spy_growth.html`).

Key features:

  • Plots both TTM and Forward implied growth on a single chart.
    Users can toggle between daily, weekly, and monthly sampling.
  • Displays horizontal average and ±1σ lines for both series based on
    the full daily history.
  • Includes a 1, 3, 5, and 10‑year statistics table, showing
    average, median, standard deviation, current value, and percentile
    for TTM and Forward implied growth.
  • Supports optional P/E and 10‑year yield columns if you choose to
    extend the script later, though they are not plotted by default.

To integrate this generator, call `generate_index_growth_pages()` from
your build pipeline after your database has been updated (including
running backfill_index_growth).  For example:

    from generate_index_growth_pages import generate_index_growth_pages
    generate_index_growth_pages()

This will write `qqq_growth.html` and `spy_growth.html` into your
OUTPUT_DIR.  You can adjust OUTPUT_DIR to place the files under a
specific web root.
"""

import os
import sqlite3
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.io import to_html


# ========= CONFIG =========
DB_PATH = "Stock Data.db"
OUTPUT_FILES = {"SPY": "spy_growth.html", "QQQ": "qqq_growth.html"}
PAGE_TITLES = {
    "SPY": "SPY — Implied Growth (TTM & Forward)",
    "QQQ": "QQQ — Implied Growth (TTM & Forward)",
}
OUTPUT_DIR = "."  # write pages to current directory; adjust as needed

# Column name synonyms recognized by the loader
DATE_COLS    = ["date", "as_of", "dt"]
TICKER_COLS  = ["ticker", "symbol"]
IG_TTM_COLS  = [
    "implied_growth_ttm", "implied_growth_pct", "implied_growth", "ig_ttm",
    "implied_growth_percent",
]
IG_FWD_COLS  = [
    "implied_growth_forward", "implied_growth_fwd", "forward_implied_growth",
    "ig_forward", "forward_growth", "implied_growth_forward_pct", "forward_growth_pct",
]
PE_TTM_COLS  = ["pe_ttm", "pe_ratio_ttm", "pe", "pe_ratio"]  # optional
PE_FWD_COLS  = ["pe_forward", "pe_fwd", "forward_pe", "pe_ntm", "pe_next12m", "pe_fy1"]  # optional
TENY_COLS    = ["tnx_yield", "us10y", "ust10y", "ten_year_yield", "ten_year", "10y"]  # optional


# ========= UTILITIES =========
def _list_tables(conn: sqlite3.Connection) -> List[str]:
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    return [r[0] for r in cur.fetchall()]


def _table_cols(conn: sqlite3.Connection, table: str) -> List[str]:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info('{table}')")
    return [r[1] for r in cur.fetchall()]


def _first_hit(cols: List[str], candidates: List[str]) -> Optional[str]:
    """Return the first column from `cols` that matches any candidate case‑insensitively."""
    look = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in look:
            return look[cand]
    return None


def _choose_source_table(conn: sqlite3.Connection, ticker: str) -> Optional[Tuple[str, Dict[str, str]]]:
    """
    Select the best table containing implied growth data for a given ticker.

    The table must have a date column and at least one of the TTM or Forward
    implied growth columns.  Among eligible tables, the one with the
    greatest number of rows for the ticker is chosen.

    Returns (table_name, column_mapping) where column_mapping maps keys:
      'date', 'ticker' (optional), 'ig' (optional), 'ig_fwd' (optional),
      'pe' (optional), 'pe_fwd' (optional), 'tnx' (optional)
    """
    best: Optional[Tuple[str, Dict[str, str]]] = None
    best_rows: int = -1
    for t in _list_tables(conn):
        cols = _table_cols(conn, t)
        date_c  = _first_hit(cols, DATE_COLS)
        tick_c  = _first_hit(cols, TICKER_COLS)
        ig_c    = _first_hit(cols, IG_TTM_COLS)
        igf_c   = _first_hit(cols, IG_FWD_COLS)
        pe_c    = _first_hit(cols, PE_TTM_COLS)
        pef_c   = _first_hit(cols, PE_FWD_COLS)
        tnx_c   = _first_hit(cols, TENY_COLS)
        # need date and at least one implied growth column
        if not date_c or not (ig_c or igf_c):
            continue
        cur = conn.cursor()
        if tick_c:
            cur.execute(f"SELECT COUNT(*) FROM '{t}' WHERE {tick_c}=?", (ticker,))
        else:
            cur.execute(f"SELECT COUNT(*) FROM '{t}'")
        n = int(cur.fetchone()[0])
        if n > best_rows:
            best_rows = n
            best = (t, {
                "date": date_c,
                "ticker": tick_c,
                "ig": ig_c,
                "ig_fwd": igf_c,
                "pe": pe_c,
                "pe_fwd": pef_c,
                "tnx": tnx_c,
            })
    return best



def _load_series(conn: sqlite3.Connection, ticker: str) -> pd.DataFrame:
    """Load implied growth and P/E series for a given ticker."""
    pick = _choose_source_table(conn, ticker)
    if not pick:
        raise RuntimeError(
            f"Could not find a table with usable columns for {ticker}. "
            f"Expected columns like {DATE_COLS} + one of {IG_TTM_COLS} or {IG_FWD_COLS}."
        )
    table, cm = pick
    selects = [cm["date"] + " AS date"]
    if cm["ig"]:
        selects.append(cm["ig"] + " AS ig")
    if cm["ig_fwd"]:
        selects.append(cm["ig_fwd"] + " AS ig_fwd")
    if cm["pe"]:
        selects.append(cm["pe"] + " AS pe")
    if cm["pe_fwd"]:
        selects.append(cm["pe_fwd"] + " AS pe_fwd")
    if cm["tnx"]:
        selects.append(cm["tnx"] + " AS tnx")
    where = ""
    params: Tuple = tuple()
    if cm["ticker"]:
        where = f"WHERE {cm['ticker']} = ?"
        params = (ticker,)
    sql = f"SELECT {', '.join(selects)} FROM '{table}' {where} ORDER BY {cm['date']} ASC"
    df = pd.read_sql_query(sql, conn, params=params, parse_dates=["date"])
    if df.empty:
        raise RuntimeError(f"Table '{table}' has no rows for {ticker}.")
    df = df.drop_duplicates(subset=["date"]).set_index("date").sort_index()
    # Convert implied growth columns to percent if stored as decimals
    for col in ["ig", "ig_fwd"]:
        if col in df.columns:
            s = df[col].dropna()
            if not s.empty and s.median() < 1.0:
                df[col] = df[col] * 100.0
    keep = [c for c in ["ig", "ig_fwd", "pe", "pe_fwd", "tnx"] if c in df.columns]
    return df[keep]



def _resample_frames(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (daily, weekly, monthly) frames using last observation for each period."""
    d = df.copy()
    w = df.resample("W-FRI").last()
    m = df.resample("M").last()
    return (d, w, m)



def _percentile(series: pd.Series, value: float) -> float:
    s = series.dropna().values
    if len(s) == 0:
        return float("nan")
    return 100.0 * (np.sum(s <= value) / len(s))



def _fmt_pct(x: Optional[float]) -> str:
    return "—" if (x is None or pd.isna(x)) else f"{x:.2f}%"



def _fmt_num(x: Optional[float]) -> str:
    return "—" if (x is None or pd.isna(x)) else f"{x:.2f}"



def _timeframe_slice(s: pd.Series, years: int) -> pd.Series:
    if s.dropna().empty:
        return s.iloc[0:0]
    end = s.index.max()
    start = end - pd.Timedelta(days=int(round(365.25 * years)))
    return s.loc[start:end]



def _stats(s: pd.Series) -> Dict[str, float]:
    s = s.dropna()
    if s.empty:
        return dict(avg=np.nan, med=np.nan, std=np.nan, latest=np.nan, pct=np.nan)
    latest = s.iloc[-1]
    return dict(
        avg=s.mean(),
        med=s.median(),
        std=s.std(),
        latest=latest,
        pct=_percentile(s, latest),
    )



def _timeframe_table_html(df_daily: pd.DataFrame) -> str:
    """Build the 1/3/5/10‑year stats table as HTML."""
    timeframes = [1, 3, 5, 10]
    ttm = df_daily["ig"] if "ig" in df_daily.columns else pd.Series(dtype=float)
    fwd = df_daily["ig_fwd"] if "ig_fwd" in df_daily.columns else pd.Series(dtype=float)
    rows_html = []
    for yrs in timeframes:
        ttm_stats = _stats(_timeframe_slice(ttm, yrs)) if not ttm.empty else None
        fwd_stats = _stats(_timeframe_slice(fwd, yrs)) if not fwd.empty else None
        def get(stats: Optional[Dict[str, float]], key: str, is_pct: bool) -> str:
            if stats is None:
                return "—"
            return _fmt_pct(stats[key]) if is_pct else _fmt_num(stats[key])
        tr = f"""
        <tr>
          <td class="tf">{yrs} Year{'s' if yrs != 1 else ''}</td>
          <td>{get(ttm_stats, 'avg', True)}</td><td>{get(fwd_stats, 'avg', True)}</td>
          <td>{get(ttm_stats, 'med', True)}</td><td>{get(fwd_stats, 'med', True)}</td>
          <td>{get(ttm_stats, 'std', True)}</td><td>{get(fwd_stats, 'std', True)}</td>
          <td>{get(ttm_stats, 'latest', True)}</td><td>{get(fwd_stats, 'latest', True)}</td>
          <td>{get(ttm_stats, 'pct', False)}</td><td>{get(fwd_stats, 'pct', False)}</td>
        </tr>"""
        rows_html.append(tr)
    return f"""
    <div class="table-wrap">
      <table class="stats">
        <thead>
          <tr>
            <th rowspan="2" class="tf">Timeframe</th>
            <th colspan="2">Average</th>
            <th colspan="2">Median</th>
            <th colspan="2">Std Dev</th>
            <th colspan="2">Current</th>
            <th colspan="2">Percentile</th>
          </tr>
          <tr>
            <th>TTM</th><th>Forward</th>
            <th>TTM</th><th>Forward</th>
            <th>TTM</th><th>Forward</th>
            <th>TTM</th><th>Forward</th>
            <th>TTM</th><th>Forward</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows_html)}
        </tbody>
      </table>
    </div>
    """



def _stat_lines(df_daily: pd.DataFrame) -> List[go.Scatter]:
    """Return horizontal avg and ±1σ lines for TTM and Forward implied growth."""
    traces: List[go.Scatter] = []
    if "ig" in df_daily.columns and not df_daily["ig"].dropna().empty:
        avg = df_daily["ig"].mean()
        std = df_daily["ig"].std()
        x = df_daily.index
        traces += [
            go.Scatter(x=x, y=[avg] * len(x), name="TTM Avg", mode="lines", line=dict(dash="dash")),
            go.Scatter(x=x, y=[avg + std] * len(x), name="TTM +1σ", mode="lines", line=dict(dash="dot")),
            go.Scatter(x=x, y=[avg - std] * len(x), name="TTM -1σ", mode="lines", line=dict(dash="dot")),
        ]
    if "ig_fwd" in df_daily.columns and not df_daily["ig_fwd"].dropna().empty:
        avg = df_daily["ig_fwd"].mean()
        std = df_daily["ig_fwd"].std()
        x = df_daily.index
        traces += [
            go.Scatter(x=x, y=[avg] * len(x), name="Forward Avg", mode="lines", line=dict(dash="dash")),
            go.Scatter(x=x, y=[avg + std] * len(x), name="Forward +1σ", mode="lines", line=dict(dash="dot")),
            go.Scatter(x=x, y=[avg - std] * len(x), name="Forward -1σ", mode="lines", line=dict(dash"],"dot")),
        ]
    return traces



def _figure(df_d: pd.DataFrame, df_w: pd.DataFrame, df_m: pd.DataFrame, ticker: str) -> go.Figure:
    """Build the Plotly figure for implied growth."""
    def mk_traces(df: pd.DataFrame, tag: str) -> List[go.Scatter]:
        traces = []
        if "ig" in df.columns:
            traces.append(go.Scatter(
                x=df.index, y=df["ig"], name=f"TTM ({tag})", mode="lines",
                hovertemplate="%{y:.2f}%<extra></extra>",
            ))
        if "ig_fwd" in df.columns:
            traces.append(go.Scatter(
                x=df.index, y=df["ig_fwd"], name=f"Forward ({tag})", mode="lines",
                hovertemplate="%{y:.2f}%<extra></extra>",
            ))
        return traces
    traces_d = mk_traces(df_d, "Daily")
    traces_w = mk_traces(df_w, "Weekly")
    traces_m = mk_traces(df_m, "Monthly")
    stat_lines = _stat_lines(df_d)
    fig = go.Figure(data=traces_d + traces_w + traces_m + stat_lines)
    n_d, n_w, n_m, n_s = len(traces_d), len(traces_w), len(traces_m), len(stat_lines)
    # Visibility masks: default to weekly sampling with stat lines visible
    vis_daily   = [True]  * n_d + [False] * n_w + [False] * n_m + [True] * n_s
    vis_weekly  = [False] * n_d + [True]  * n_w + [False] * n_m + [True] * n_s
    vis_monthly = [False] * n_d + [False] * n_w + [True]  * n_m + [True] * n_s
    for i, v in enumerate(vis_weekly):
        fig.data[i].visible = v
    fig.update_layout(
        title=f"{ticker} — Implied Growth (TTM & Forward) — avg and ±1σ shown",
        margin=dict(l=50, r=50, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        xaxis=dict(
            title="Date",
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(step="all", label="All"),
                ])
            ),
            rangeslider=dict(visible=True),
            type="date",
        ),
        yaxis=dict(title="Implied Growth (%)", side="left"),
        template=None,
    )
    # Add toggle buttons for Daily/Weekly/Monthly
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.0, xanchor="left",
                y=1.16, yanchor="top",
                buttons=[
                    dict(label="Daily",   method="update", args=[{"visible": vis_daily},   {}]),
                    dict(label="Weekly",  method="update", args=[{"visible": vis_weekly},  {}]),
                    dict(label="Monthly", method="update", args=[{"visible": vis_monthly}, {}]),
                ],
            )
        ]
    )
    return fig



def _page_html(title: str, chart_html: str, timeframe_table_html: str) -> str:
    """Assemble the full HTML page with chart and stats table."""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>{title}</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 16px; background: #f7f7fb; }}
    h1 {{ font-size: 1.6rem; margin: 0 0 10px; }}
    .wrap {{ max-width: 1100px; margin: 0 auto; }}
    .chart {{ margin-top: 8px; background: #fff; border: 1px solid #e6e6f0; border-radius: 8px; padding: 8px; }}
    .table-wrap {{ margin-top: 14px; background: #f4f5ff; border-radius: 8px; padding: 10px; }}
    table.stats {{ width: 100%; border-collapse: collapse; background: #fff; border: 2px solid #3a4bff20; }}
    table.stats th, table.stats td {{ border: 1px solid #dfe3ff; padding: 8px 10px; text-align: center; }}
    table.stats thead th {{ background: #eef1ff; font-weight: 600; }}
    td.tf, th.tf {{ text-align: left; white-space: nowrap; }}
    .back {{ margin-top: 12px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>{title}</h1>
    <div class="chart">
      {chart_html}
    </div>
    {timeframe_table_html}
    <p class="back"><a href="index.html">← Back to Dashboard</a></p>
  </div>
</body>
</html>"""



def _write_page(out_path: str, html: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)



def _build_one(ticker: str, df: pd.DataFrame) -> None:
    """Build a single valuation page for a ticker."""
    df_d, df_w, df_m = _resample_frames(df)
    fig = _figure(df_d, df_w, df_m, ticker)
    chart_html = to_html(fig, include_plotlyjs="cdn", f
ull_html=False, default_height="600px")
    tf_table = _timeframe_table_html(df_d)
    page = _page_html(PAGE_TITLES[ticker], chart_html, tf_table)
    out_file = os.path.join(OUTPUT_DIR, OUTPUT_FILES[ticker])
    _write_page(out_file, page)



def generate_index_growth_pages(db_path: str = DB_PATH) -> None:
    """Entry point to generate pages for SPY and QQQ."""
    conn = sqlite3.connect(db_path)
    try:
        for ticker in ("SPY", "QQQ"):
            df = _load_series(conn, ticker)
            # Drop rows where both TTM and Forward implied growth are missing
            if {"ig", "ig_fwd"}.intersection(df.columns):
                df = df.dropna(subset=[c for c in ["ig", "ig_fwd"] if c in df.columns], how="all")
            _build_one(ticker, df)
    finally:
        conn.close()


# Allow running this file directly as a script
if __name__ == "__main__":
    generate_index_growth_pages()

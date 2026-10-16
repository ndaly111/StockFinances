#!/usr/bin/env python3
"""
generate_index_growth_pages.py — Create interactive SPY/QQQ valuation pages

This module reads historical implied growth and P/E data from your SQLite
database and produces self-contained HTML pages for SPY and QQQ that
feature interactive Plotly charts. The pages are written to the
filenames specified in OUTPUT_FILES, preserving your existing URL
structure (e.g., `qqq_growth.html` and `spy_growth.html`).

Key features:
  • Plots both TTM and Forward implied growth on a single chart.
    Users can toggle between daily, weekly, and monthly sampling.
  • Displays horizontal average and ±1σ lines for both series based on
    the full daily history.
  • Includes a 1, 3, 5, and 10-year statistics table, showing
    average, median, standard deviation, current value, and percentile
    for TTM and Forward implied growth.
  • Supports optional P/E and 10-year yield columns.

Integrate by calling:
    from generate_index_growth_pages import generate_index_growth_pages
    generate_index_growth_pages()
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
OUTPUT_DIR = "."  # write pages to current directory

# Column name synonyms recognized by the loader
DATE_COLS   = ["date", "as_of", "dt"]
TICKER_COLS = ["ticker", "symbol"]
IG_TTM_COLS = [
    "implied_growth_ttm", "implied_growth_pct", "implied_growth", "ig_ttm",
    "implied_growth_percent",
]
IG_FWD_COLS = [
    "implied_growth_forward", "implied_growth_fwd", "forward_implied_growth",
    "ig_forward", "forward_growth", "implied_growth_forward_pct", "forward_growth_pct",
]
PE_TTM_COLS = ["pe_ttm", "pe_ratio_ttm", "pe", "pe_ratio"]  # optional
PE_FWD_COLS = ["pe_forward", "pe_fwd", "forward_pe", "pe_ntm", "pe_next12m", "pe_fy1"]  # optional
TENY_COLS   = ["tnx_yield", "us10y", "ust10y", "ten_year_yield", "ten_year", "10y"]  # optional

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
    """Return the first column from `cols` that matches any candidate case-insensitively."""
    look = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in look:
            return look[cand]
    return None

def _choose_source_table(conn: sqlite3.Connection, ticker: str) -> Optional[Tuple[str, Dict[str, str]]]:
    """
    Pick the best table containing implied growth for `ticker`.
    Must have a date column + at least one of TTM/Forward IG.
    Returns (table_name, column_mapping).
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
                "date":   date_c,
                "ticker": tick_c,
                "ig":     ig_c,
                "ig_fwd": igf_c,
                "pe":     pe_c,
                "pe_fwd": pef_c,
                "tnx":    tnx_c,
            })
    return best

def _load_series(conn: sqlite3.Connection, ticker: str) -> pd.DataFrame:
    """Load implied growth (and optional PE, 10y) for `ticker`."""
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
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.normalize()
    df = df.drop_duplicates(subset=["date"]).set_index("date").sort_index()
    # Convert implied growth columns to percent if stored as decimals
    for col in ["ig", "ig_fwd"]:
        if col in df.columns:
            s = df[col].dropna()
            if not s.empty and s.median() < 1.0:
                df[col] = df[col] * 100.0
    keep = [c for c in ["ig", "ig_fwd", "pe", "pe_fwd", "tnx"] if c in df.columns]
    df = df[keep]

    # Attach EPS if available.
    eps = _index_eps_series(conn, ticker)
    if not eps.empty:
        df = df.join(eps, how="outer")

    return df


_INDEX_EPS_DIVISOR = {"SPY": 10.0, "QQQ": 4.0}

def _index_eps_series(conn: sqlite3.Connection, ticker: str) -> pd.Series:
    """Index-level EPS history: TTM_REPORTED, extended with TTM_DAILY, then
    IMPLIED_FROM_PE (ETF-level, scaled by the index divisor). Mirrors the prior
    Bokeh _series_eps so SPY and QQQ both get a full series."""
    def _read(eps_type):
        df = pd.read_sql_query(
            "SELECT Date, EPS FROM Index_EPS_History WHERE Ticker=? AND EPS_Type=? ORDER BY Date",
            conn, params=(ticker.upper(), eps_type), parse_dates=["Date"])
        if df.empty:
            return pd.Series(dtype=float)
        s = pd.to_numeric(df.set_index(pd.to_datetime(df["Date"], utc=True).dt.normalize())["EPS"],
                          errors="coerce").dropna()
        return s[~s.index.duplicated(keep="last")]
    reported = _read("TTM_REPORTED")
    daily = _read("TTM_DAILY")
    implied = _read("IMPLIED_FROM_PE") * _INDEX_EPS_DIVISOR.get(ticker.upper(), 1.0)
    parts = [s for s in (reported, daily, implied) if not s.empty]
    if not parts:
        return pd.Series(dtype=float, name="eps")
    combined = parts[0]
    for p in parts[1:]:
        combined = combined.combine_first(p)
    combined = combined.sort_index(); combined.name = "eps"
    return combined


def _load_eps_series(conn: sqlite3.Connection, ticker: str) -> Optional[pd.Series]:
    """Return the ETF EPS history as a Series indexed by date."""

    query = (
        "SELECT Date, EPS FROM Index_EPS_History "
        "WHERE Ticker=? AND EPS_Type='TTM_REPORTED' ORDER BY Date"
    )
    df = pd.read_sql_query(query, conn, params=(ticker,), parse_dates=["Date"])
    if df.empty:
        return None
    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.normalize()
    eps = pd.to_numeric(df["EPS"], errors="coerce").dropna()
    if eps.empty:
        return None
    series = eps.groupby(df.loc[eps.index, "Date"]).last()
    series.name = "eps"
    return series

def _latest_forward_eps(conn: sqlite3.Connection, ticker: str):
    """Return the most-recent displayable forward-EPS row for ticker, or None."""
    try:
        row = conn.execute(
            """SELECT forward_eps_index, horizon_date, growth_this_fy, growth_next_fy,
                      coverage_weight, displayable, source
                 FROM Index_Forward_EPS_History
                WHERE ticker=? ORDER BY date_recorded DESC LIMIT 1""",
            (ticker.upper(),)).fetchone()
    except Exception:
        return None
    if not row or row[0] is None or not row[5]:
        return None
    return {"forward_eps_index": float(row[0]), "horizon_date": row[1],
            "growth_this_fy": row[2], "growth_next_fy": row[3],
            "coverage_weight": row[4], "source": row[6]}


def _resample_frames(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (daily, weekly, monthly) frames using last observation for each period."""
    d = df.copy()
    w = df.resample("W-FRI").last()
    m = df.resample("ME").last()
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
    """Build the 1/3/5/10-year stats table as HTML."""
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
            go.Scatter(x=x, y=[avg] * len(x), name="TTM Avg",  mode="lines", line=dict(dash="dash")),
            go.Scatter(x=x, y=[avg + std] * len(x), name="TTM +1σ", mode="lines", line=dict(dash="dot")),
            go.Scatter(x=x, y=[avg - std] * len(x), name="TTM -1σ", mode="lines", line=dict(dash="dot")),
        ]
    if "ig_fwd" in df_daily.columns and not df_daily["ig_fwd"].dropna().empty:
        avg = df_daily["ig_fwd"].mean()
        std = df_daily["ig_fwd"].std()
        x = df_daily.index
        traces += [
            go.Scatter(x=x, y=[avg] * len(x), name="Forward Avg",  mode="lines", line=dict(dash="dash")),
            go.Scatter(x=x, y=[avg + std] * len(x), name="Forward +1σ", mode="lines", line=dict(dash="dot")),
            go.Scatter(x=x, y=[avg - std] * len(x), name="Forward -1σ", mode="lines", line=dict(dash="dot")),
        ]
    return traces

_RETRO_COLORWAY = ["#000080", "#CC0000", "#008000", "#B8860B", "#3333AA", "#800080"]

def _apply_retro_layout(fig):
    """Style a Plotly figure to match the site's 1999 NASDAQ-terminal theme
    (navy/gray Win95, Times/Arial/Courier fonts)."""
    fig.update_layout(
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FBFBF4",
        font=dict(family='"Times New Roman", Times, serif', size=12, color="#000080"),
        colorway=_RETRO_COLORWAY,
        legend=dict(font=dict(family="Arial, Helvetica, sans-serif", size=11, color="#000080"),
                    bgcolor="rgba(255,255,255,0.65)", bordercolor="#C0C0C0", borderwidth=1),
        margin=dict(l=52, r=20, t=22, b=30),
        hoverlabel=dict(font=dict(family='"Courier New", monospace', size=12), bgcolor="#000080"),
    )
    fig.update_xaxes(gridcolor="#E4E4EC", linecolor="#000080",
                     tickfont=dict(family="Arial, sans-serif", size=10, color="#000080"))
    fig.update_yaxes(gridcolor="#E4E4EC", linecolor="#000080",
                     tickfont=dict(family="Arial, sans-serif", size=10, color="#000080"))
    # The card <h2> already titles each chart, and the range-slider mini-chart adds
    # clutter under a terminal aesthetic — drop both for a cleaner look.
    fig.update_layout(title=None)
    fig.update_xaxes(rangeslider_visible=False)
    # Plotly's built-in range selector moved x but left y stale (squashed lines),
    # and drag/pinch zoom fought page scrolling on phones. Range buttons are now
    # HTML (see _range_bar / setRange in _page_html) that set x AND y in one
    # relayout; the plot itself is locked so a swipe just scrolls the page.
    fig.update_xaxes(rangeselector=dict(visible=False), fixedrange=True, title=None)
    fig.update_yaxes(fixedrange=True)
    fig.update_layout(
        dragmode=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0.0),
        margin=dict(l=48, r=12, t=30, b=28),
    )
    return fig

def _growth_figure(df_d: pd.DataFrame, df_w: pd.DataFrame, df_m: pd.DataFrame, ticker: str) -> go.Figure:
    """Build the Plotly figure for implied growth."""

    def mk_traces(df: pd.DataFrame) -> List[go.Scatter]:
        traces: List[go.Scatter] = []
        if "ig" in df.columns:
            traces.append(
                go.Scatter(
                    x=df.index,
                    y=df["ig"],
                    name="TTM",
                    mode="lines",
                    hovertemplate="%{y:.2f}%<extra></extra>",
                )
            )
        if "ig_fwd" in df.columns:
            traces.append(
                go.Scatter(
                    x=df.index,
                    y=df["ig_fwd"],
                    name="Forward",
                    mode="lines",
                    hovertemplate="%{y:.2f}%<extra></extra>",
                )
            )
        return traces

    traces_m = mk_traces(df_m)
    stat_lines = _stat_lines(df_d)  # horizontal refs stay on full daily history

    fig = go.Figure(data=traces_m + stat_lines)

    fig.update_layout(
        title=f"{ticker} — Implied Growth (TTM & Forward) — avg and ±1σ shown",
        margin=dict(l=50, r=50, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        xaxis=dict(
            title="Date",
            rangeselector=dict(buttons=[
                dict(count=1,  label="1Y",  step="year", stepmode="backward"),
                dict(count=3,  label="3Y",  step="year", stepmode="backward"),
                dict(count=5,  label="5Y",  step="year", stepmode="backward"),
                dict(count=10, label="10Y", step="year", stepmode="backward"),
                dict(step="all", label="All"),
            ]),
            rangeslider=dict(visible=True),
            type="date",
        ),
        yaxis=dict(title="Implied Growth (%)", side="left"),
        template=None,
    )
    if len(df_m.index):
        xmax_hist = pd.Timestamp(df_m.index.max())
        xmin = xmax_hist - pd.DateOffset(years=5)
        xmax = xmax_hist
        fig.update_xaxes(range=[xmin, xmax])
    return fig


def _eps_forecast_trace(last_dt, last_eps, fwd, scale=1.0) -> go.Scatter:
    """Return a dashed forecast scatter for the EPS chart.

    *last_dt* must be tz-aware (it comes from the EPS series index).
    *scale* lets the indexed chart pass 100/base so the trace lives in the
    same coordinate space.
    """
    h1 = pd.Timestamp(fwd["horizon_date"], tz="UTC")
    xs = [last_dt, h1]
    ys = [last_eps, fwd["forward_eps_index"] * scale]
    g2 = fwd.get("growth_next_fy")
    if g2 is not None:
        xs.append(h1 + pd.DateOffset(years=1))
        ys.append(fwd["forward_eps_index"] * (1 + g2) * scale)
    return go.Scatter(
        x=xs, y=ys,
        name="Forecast (this/next FY)",
        mode="lines+markers",
        line=dict(dash="dash", color="#ff8800"),
        marker=dict(symbol="diamond", size=11, color="#ff8800"),
    )


def _eps_figure(
    df_d: pd.DataFrame,
    df_w: pd.DataFrame,
    df_m: pd.DataFrame,
    ticker: str,
    fwd=None,
) -> Optional[go.Figure]:
    """Build the Plotly figure for EPS if data is available."""

    def mk_traces(df: pd.DataFrame) -> List[go.Scatter]:
        if "eps" not in df.columns:
            return []
        return [
            go.Scatter(
                x=df.index,
                y=df["eps"],
                name="EPS",
                mode="lines",
                hovertemplate="$%{y:.2f}<extra></extra>",
            )
        ]

    traces_m = mk_traces(df_m)

    if not traces_m:
        return None

    s_monthly = df_m["eps"].dropna() if "eps" in df_m.columns else pd.Series(dtype=float)

    # Forecast trace (always visible — treated like stat lines in _growth_figure).
    forecast_traces: List[go.Scatter] = []
    if fwd is not None and not s_monthly.empty:
        forecast_traces = [
            _eps_forecast_trace(s_monthly.index[-1], float(s_monthly.iloc[-1]), fwd)
        ]

    fig = go.Figure(data=traces_m + forecast_traces)

    fig.update_layout(
        title=f"{ticker} — EPS (TTM)",
        margin=dict(l=50, r=50, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        xaxis=dict(
            title="Date",
            rangeselector=dict(buttons=[
                dict(count=1,  label="1Y",  step="year", stepmode="backward"),
                dict(count=3,  label="3Y",  step="year", stepmode="backward"),
                dict(count=5,  label="5Y",  step="year", stepmode="backward"),
                dict(count=10, label="10Y", step="year", stepmode="backward"),
                dict(step="all", label="All"),
            ]),
            rangeslider=dict(visible=True),
            type="date",
        ),
        yaxis=dict(title="EPS (USD)", tickprefix="$", type="linear"),
        template=None,
    )
    if len(df_m.index):
        xmax_hist = pd.Timestamp(df_m.index.max())
        xmin = xmax_hist - pd.DateOffset(years=5)
        xmax = xmax_hist
        try:
            if fwd and fwd.get("horizon_date"):
                fh = pd.Timestamp(fwd["horizon_date"], tz="UTC") + pd.DateOffset(years=1)
                if fh > xmax:
                    xmax = fh
        except Exception:
            pass
        fig.update_xaxes(range=[xmin, xmax])
    return fig

def _eps_indexed_figure(
    df_d: pd.DataFrame,
    df_w: pd.DataFrame,
    df_m: pd.DataFrame,
    ticker: str,
    fwd=None,
) -> Optional[go.Figure]:
    """Build an EPS chart normalised to 100 at the start of the default view.

    The base is the EPS ~5 years ago (the start of the default x-window), NOT the
    first observation — for indices like SPY the reported series runs back to 1970
    (~$5), so indexing to that made the default view show values in the thousands
    on a log axis (unreadable). Anchoring to the window start keeps the default
    view around 100-180 on a linear axis. Returns None if that base is <= 0.
    """
    b = df_m["eps"].dropna() if "eps" in df_m.columns else pd.Series(dtype=float)
    if b.empty:
        return None
    _win_start = pd.Timestamp(b.index.max()) - pd.DateOffset(years=5)
    _in_win = b[b.index >= _win_start]
    base = float(_in_win.iloc[0]) if not _in_win.empty else float(b.iloc[0])
    if base <= 0:
        return None

    def mk_traces(df: pd.DataFrame) -> List[go.Scatter]:
        if "eps" not in df.columns:
            return []
        s = df["eps"].dropna()
        if s.empty:
            return []
        return [
            go.Scatter(
                x=s.index,
                y=s / base * 100,
                name="EPS Indexed",
                mode="lines",
                hovertemplate="%{y:.1f}<extra></extra>",
            )
        ]

    traces_m = mk_traces(df_m)

    # Forecast trace (always visible).
    forecast_traces: List[go.Scatter] = []
    if fwd is not None and not b.empty:
        last_eps_indexed = float(b.iloc[-1]) / base * 100
        forecast_traces = [
            _eps_forecast_trace(b.index[-1], last_eps_indexed, fwd, scale=100.0 / base)
        ]

    fig = go.Figure(data=traces_m + forecast_traces)

    fig.update_layout(
        title=f"{ticker} — EPS (Indexed to 100)",
        margin=dict(l=50, r=50, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        xaxis=dict(
            title="Date",
            rangeselector=dict(buttons=[
                dict(count=1,  label="1Y",  step="year", stepmode="backward"),
                dict(count=3,  label="3Y",  step="year", stepmode="backward"),
                dict(count=5,  label="5Y",  step="year", stepmode="backward"),
                dict(count=10, label="10Y", step="year", stepmode="backward"),
                dict(step="all", label="All"),
            ]),
            rangeslider=dict(visible=True),
            type="date",
        ),
        yaxis=dict(title="Indexed EPS (5Y ago = 100)", type="linear"),
        template=None,
    )
    if len(df_m.index):
        xmax_hist = pd.Timestamp(df_m.index.max())
        xmin = xmax_hist - pd.DateOffset(years=5)
        xmax = xmax_hist
        try:
            if fwd and fwd.get("horizon_date"):
                fh = pd.Timestamp(fwd["horizon_date"], tz="UTC") + pd.DateOffset(years=1)
                if fh > xmax:
                    xmax = fh
        except Exception:
            pass
        fig.update_xaxes(range=[xmin, xmax])
    return fig


def _forward_growth_rates(conn: sqlite3.Connection, ticker: str):
    """Latest forward FY growth rates for the growth scorecard.

    Unlike _latest_forward_eps (which feeds the dollar forecast overlay and is
    held to the `displayable` band), the scorecard only shows growth RATES, which
    are scale-free — so a bottom-up row is accepted on coverage alone (>=85% of
    index weight). The band (bench ±20pts) was hiding QQQ's real +50%-plus year.
    """
    try:
        rows = conn.execute(
            """SELECT growth_this_fy, growth_next_fy, horizon_date, coverage_weight,
                      displayable, source, date_recorded
                 FROM Index_Forward_EPS_History
                WHERE ticker=? AND growth_this_fy IS NOT NULL
                ORDER BY date_recorded DESC LIMIT 30""",
            (ticker.upper(),)).fetchall()
    except Exception:
        return None
    # Newest qualifying row within 14 days — e.g. SPY's FactSet PDF fetch can
    # miss days, and the low-coverage bottom-up fallback row isn't good enough.
    row = None
    for r in rows:
        if (pd.Timestamp(rows[0][6]) - pd.Timestamp(r[6])).days > 14:
            break
        if r[4] or (r[3] or 0) >= 0.85:
            row = r
            break
    if not row:
        return None
    g1, g2, horizon, cov, disp, src, asof = row
    try:
        fy = pd.Timestamp(horizon).year
    except Exception:
        return None
    return {"g1": float(g1), "g2": None if g2 is None else float(g2), "fy": fy,
            "coverage": cov, "source": src, "asof": asof}


def _eps_growth_summary(eps: pd.Series, fwd) -> Optional[dict]:
    """Year-over-year TTM EPS growth by calendar year, 5Y/10Y CAGR, and forward
    FY growth with the implied 2-year CAGR."""
    s = eps.dropna()
    s = s[s > 0]
    if len(s) < 2:
        return None
    last_dt, last = s.index[-1], float(s.iloc[-1])

    def _at(years: int) -> Optional[float]:
        prior = s[s.index <= last_dt - pd.DateOffset(years=years)]
        # Require the series to actually reach back that far (within ~1 month).
        if prior.empty or (last_dt - pd.DateOffset(years=years) - prior.index[-1]).days > 31:
            return None
        return float(prior.iloc[-1])

    def _cagr(years: int) -> Optional[float]:
        base = _at(years)
        return None if not base else (last / base) ** (1.0 / years) - 1

    ye = s.resample("YE").last().dropna()
    bars = []  # (label, growth, kind)
    for i in range(1, len(ye)):
        if ye.index[i].year == last_dt.year:
            break
        bars.append((f"'{ye.index[i].year % 100:02d}", ye.iloc[i] / ye.iloc[i - 1] - 1, "hist"))
    bars = bars[-10:]
    one_y = _cagr(1)
    if one_y is not None:
        bars.append(("TTM", one_y, "ttm"))

    out = {"one_y": one_y, "five_y": _cagr(5), "ten_y": _cagr(10),
           "bars": bars, "fwd": fwd, "two_y": None}
    if fwd:
        bars.append((f"{fwd['fy'] % 100:02d}E", fwd["g1"], "fwd"))
        if fwd["g2"] is not None:
            bars.append((f"{(fwd['fy'] + 1) % 100:02d}E", fwd["g2"], "fwd"))
            out["two_y"] = ((1 + fwd["g1"]) * (1 + fwd["g2"])) ** 0.5 - 1
    return out


_BAR_COLORS = {"hist": "#000080", "ttm": "#3333AA", "fwd": "#ff8800"}


def _eps_growth_figure(summary: dict) -> go.Figure:
    """Static bar chart: calendar-year EPS growth, then TTM-now, then estimates."""
    labels = [b[0] for b in summary["bars"]]
    vals = [b[1] * 100 for b in summary["bars"]]
    fig = go.Figure(go.Bar(
        x=labels, y=vals,
        marker=dict(color=[_BAR_COLORS[b[2]] for b in summary["bars"]],
                    line=dict(width=0)),
        text=[f"{v:+.0f}%" for v in vals], textposition="outside", cliponaxis=False,
        constraintext="none",
        textfont=dict(family="Arial, sans-serif", size=10, color="#000080"),
        hovertemplate="%{x}: %{y:+.1f}%<extra></extra>",
    ))
    lo, hi = min(vals + [0]), max(vals + [0])
    pad = (hi - lo) * 0.15 or 5
    fig.update_layout(showlegend=False, bargap=0.25, template=None)
    fig.update_yaxes(ticksuffix="%", range=[lo - (pad if lo < 0 else 0), hi + pad],
                     zeroline=True, zerolinecolor="#000080", zerolinewidth=1)
    _apply_retro_layout(fig)
    fig.update_xaxes(type="category", tickangle=0)
    fig.update_layout(margin=dict(l=44, r=8, t=14, b=28))
    return fig


def _growth_tiles_html(summary: dict) -> str:
    def tile(label, val, fwd=False):
        txt = "—" if val is None else f"{val * 100:+.1f}%"
        cls = "tile fwd" if fwd else "tile"
        return f'<div class="{cls}"><div class="tl">{label}</div><div class="tv">{txt}</div></div>'

    f = summary["fwd"]
    tiles = [tile("1Y (TTM)", summary["one_y"]),
             tile("5Y CAGR", summary["five_y"]),
             tile("10Y CAGR", summary["ten_y"])]
    if f:
        fy1, fy2 = f"FY{f['fy'] % 100:02d}", f"FY{(f['fy'] + 1) % 100:02d}"
        tiles += [tile(f"{fy1} est.", f["g1"], True),
                  tile(f"{fy2} est.", f["g2"], True),
                  tile("2Y implied CAGR", summary["two_y"], True)]
        if (f["source"] or "").lower() == "factset":
            src = "FactSet S&amp;P 500 consensus"
        else:
            src = f"bottom-up analyst estimates, {(f['coverage'] or 0):.0%} of index weight"
        note = f"Estimates: {src}, as of {f['asof']}. "
    else:
        note = "No forward estimate available. "
    note += ("Bars = calendar-year change in trailing-12-month EPS; "
             "TTM = latest vs one year ago; E = estimate.")
    return (f'<div class="tiles">{"".join(tiles)}</div>'
            f'<p class="src-note">{note}</p>')


def _range_bar(div_id: str, default: int = 5) -> str:
    """HTML range buttons for a time-series chart (wired up by setRange)."""
    btns = "".join(
        f'<button class="rng{" on" if y == default else ""}" data-y="{y}">{lbl}</button>'
        for lbl, y in (("1Y", 1), ("3Y", 3), ("5Y", 5), ("10Y", 10), ("All", 0)))
    return f'<div class="rngbar" data-chart="{div_id}" data-default="{default}">{btns}</div>'


def _page_html(title: str, growth_chart_html: str, eps_chart_html: Optional[str],
               eps_indexed_html: Optional[str], timeframe_table_html: str, callout: str,
               eps_growth_html: str = "") -> str:
    """Assemble the full HTML page in the site's 1999 NASDAQ-terminal theme."""
    callout_section = ""
    if callout:
        callout_section = f'      <p class="callout">{callout}</p>\n'

    def _measured_card(title_text, chart_html, ro_id, clear_target):
        return f"""    <div class="chart-card">
      <h2>{title_text}</h2>
      <div class="chart-body">
      {_range_bar(clear_target)}
      {chart_html}
      <div class="measure-readout" id="{ro_id}">Click a point, then another, to measure % change.</div>
      <button class="win-btn" onclick="clearMeasure('{clear_target}')">Clear</button>
      </div>
    </div>
"""

    eps_section = _measured_card("EPS ($)", eps_chart_html, "ro-eps", "eps-chart") if eps_chart_html else ""
    growth_section = ""
    if eps_growth_html:
        growth_section = f"""    <div class="chart-card">
      <h2>EPS Growth: Year over Year, 5Y, 10Y &amp; Estimates</h2>
      <div class="chart-body">{eps_growth_html}</div>
    </div>
"""
    eps_indexed_section = _measured_card("EPS Growth (indexed = 100)", eps_indexed_html, "ro-eps-indexed", "eps-indexed-chart") if eps_indexed_html else ""

    measure_script = """<script>
(function(){
  function attach(divId, roId){
    var gd=document.getElementById(divId), ro=document.getElementById(roId);
    if(!gd||!ro||!gd.on) return;
    var anchor=null, first=null;
    function ms(x){ return (typeof x==='number')?x:new Date(x).getTime(); }
    gd.on('plotly_click', function(d){
      var p=d.points[0], x=ms(p.x), y=p.y;
      if(first===null) first=y;
      if(anchor===null){ anchor={x:x,y:y}; ro.innerHTML='Anchor set. Click another point.'; return; }
      var s=anchor, e={x:x,y:y};
      if(e.x < s.x){ var t=s; s=e; e=t; }
      var pct=(e.y/s.y-1)*100, yrs=(e.x-s.x)/(365.25*86400000);
      var span = yrs>=1 ? yrs.toFixed(1)+' years' : Math.round(yrs*12)+' months';
      var txt='<b>'+(pct>=0?'+':'')+pct.toFixed(1)+'%</b> over '+span;
      if(yrs>=0.02){ var ann=(Math.pow(e.y/s.y,1/yrs)-1)*100;
        if(isFinite(ann)) txt+=' ('+(ann>=0?'+':'')+ann.toFixed(1)+'% annualized)'; }
      ro.innerHTML=txt; anchor=null;
    });
    gd.on('plotly_hover', function(d){
      if(anchor!==null||first===null) return;
      var y=d.points[0].y, pct=(y/first-1)*100;
      ro.innerHTML='% from first point: <b>'+(pct>=0?'+':'')+pct.toFixed(1)+'%</b>';
    });
    gd.__clear=function(){ anchor=null; ro.innerHTML='Click a point, then another, to measure % change.'; };
  }
  function init(){
    attach('eps-chart','ro-eps'); attach('eps-indexed-chart','ro-eps-indexed');
    document.querySelectorAll('.rngbar').forEach(function(bar){
      var id=bar.getAttribute('data-chart');
      bar.querySelectorAll('button').forEach(function(b){
        b.addEventListener('click', function(){
          bar.querySelectorAll('button').forEach(function(o){ o.classList.remove('on'); });
          b.classList.add('on'); setRange(id, +b.getAttribute('data-y'));
        });
      });
      setRange(id, +bar.getAttribute('data-default'));
    });
  }
  if(document.readyState!=='loading') init(); else document.addEventListener('DOMContentLoaded', init);
})();
function clearMeasure(id){ var gd=document.getElementById(id); if(gd&&gd.__clear) gd.__clear(); }
// Set the x window AND fit y to the visible data in a single relayout, so a
// range change never leaves the line squashed against a stale y-axis.
// years=0 means All. The window ends at the last point of any trace (so the
// forecast stays in view) and starts N years before the last HISTORICAL point.
function setRange(divId, years){
  var gd=document.getElementById(divId);
  // _fullData holds decoded arrays (gd.data may be plotly's binary {bdata} form).
  var data=gd._fullData||gd.data;
  if(!data||!window.Plotly) return;
  function t(x){ return new Date(x).getTime(); }
  var lo=Infinity, histMax=-Infinity, allMax=-Infinity;
  data.forEach(function(tr, k){
    (tr.x||[]).forEach(function(x){ var v=t(x); if(!isFinite(v)) return;
      if(v<lo) lo=v; if(v>allMax) allMax=v; if(k===0&&v>histMax) histMax=v; });
  });
  if(!isFinite(lo)) return;
  if(!isFinite(histMax)) histMax=allMax;
  var x0 = years ? Math.max(lo, histMax - years*365.25*86400000) : lo, x1 = allMax;
  var ylo=Infinity, yhi=-Infinity;
  data.forEach(function(tr){
    var xs=tr.x||[], ys=tr.y||[];
    for(var i=0;i<xs.length;i++){ var xv=t(xs[i]), yv=ys[i];
      if(xv<x0||xv>x1||yv==null||!isFinite(yv)) continue;
      if(yv<ylo) ylo=yv; if(yv>yhi) yhi=yv; }
  });
  var upd={'xaxis.range':[new Date(x0).toISOString(), new Date(x1).toISOString()]};
  if(yhi>ylo){ var p=(yhi-ylo)*0.08; upd['yaxis.range']=[ylo-p, yhi+p]; }
  Plotly.relayout(gd, upd);
}
</script>"""

    marquee = ("SPY / QQQ VALUATION TERMINAL   *   IMPLIED GROWTH   *   FORWARD EPS   *   "
               "P/E PERCENTILES   *   POWERED BY NICK'S STOCK FINANCIALS   *   FIGURES DELAYED   *   ")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>{title}</title>
  <style>
    :root{{ --navy:#000080; --gray:#C0C0C0; --gray-shade:#808080; --hi-yellow:#FFFF00; --text:#000; }}
    *{{ box-sizing:border-box }} html,body{{ margin:0; padding:0 }}
    body{{ font-family:"Times New Roman",Times,serif; color:var(--text); font-size:13px; line-height:1.35;
      background-color:#9999BB;
      background-image:repeating-linear-gradient(45deg, rgba(255,255,255,.18) 0 2px, transparent 2px 8px),
        repeating-linear-gradient(-45deg, rgba(0,0,0,.10) 0 2px, transparent 2px 8px),
        linear-gradient(180deg,#9faabe 0%,#7e8aa6 100%);
      background-attachment:fixed; min-height:100vh; }}
    .wrap{{ max-width:1000px; margin:0 auto; padding:0 6px 20px }}
    .marquee{{ overflow:hidden; background:#000; border-top:2px solid var(--navy); border-bottom:2px solid var(--navy);
      box-shadow:inset 0 -1px 0 #FFF, inset 0 1px 0 #FFF; margin:0 -6px 6px; }}
    .marquee span{{ display:inline-block; white-space:nowrap; padding:5px 0; color:var(--hi-yellow);
      font-family:"Courier New",monospace; font-size:12px; font-weight:700; letter-spacing:.5px;
      text-shadow:0 0 4px rgba(255,255,0,.5); animation:ticker 34s linear infinite; }}
    @keyframes ticker{{ from{{transform:translateX(100%)}} to{{transform:translateX(-100%)}} }}
    .hero{{ font-family:"Arial Black",Arial,sans-serif; font-weight:900; font-size:18px; color:#FFF;
      background:var(--navy); margin:0 0 8px; padding:6px 10px 5px; letter-spacing:.5px; text-transform:uppercase;
      border:2px outset var(--gray); box-shadow:inset 0 1px 0 #4444CC; }}
    .chart-card{{ background:#FFF; border:2px outset var(--gray); margin-bottom:8px; }}
    .chart-card > h2{{ font-family:"Arial Black",Arial,sans-serif; font-weight:900; font-size:13px; color:#FFF;
      background:var(--navy); margin:0; padding:4px 8px 3px; letter-spacing:.5px; text-transform:uppercase;
      border-bottom:2px ridge var(--gray); border-top:1px solid #4444CC; box-shadow:inset 0 1px 0 #4444CC; }}
    .chart-body{{ padding:6px; }}
    .measure-readout{{ font-family:"Courier New",Courier,monospace; font-size:13px; font-weight:700;
      color:#33FF33; text-shadow:0 0 5px rgba(51,255,51,.5); background:#000; border:2px inset var(--gray);
      padding:5px 8px; margin:6px 0 4px; min-height:1.5em; box-shadow:inset 0 0 0 1px #222; }}
    .measure-readout b{{ color:#FFCC33; }}
    .win-btn{{ font-family:Arial,sans-serif; font-size:14px; font-weight:700; padding:7px 22px;
      background:var(--gray); border:2px outset var(--gray); cursor:pointer;
      min-height:36px; }}
    .win-btn:active{{ border-style:inset; }}
    .callout{{ font-family:"Times New Roman",serif; font-size:13px; background:#FFFFEE;
      border:2px inset var(--gray); padding:6px 10px; margin:6px; }}
    .table-wrap{{ padding:6px; }}
    table.stats{{ width:100%; border-collapse:collapse; border:2px solid var(--navy); background:#FFF; font-size:12px; }}
    table.stats th, table.stats td{{ border:1px solid #8080FF; padding:5px 8px; text-align:center; font-family:"Times New Roman",serif; }}
    table.stats thead th{{ background:var(--navy); color:#FFF; font-family:"Arial Black",Arial,sans-serif;
      font-weight:900; text-transform:uppercase; font-size:11px; letter-spacing:.5px; }}
    table.stats td.tf, table.stats th.tf{{ text-align:left; white-space:nowrap; background:#C0C0FF; font-weight:700; }}
    .rngbar{{ display:flex; gap:4px; margin:0 0 6px; }}
    .rngbar button{{ flex:1; font-family:Arial,sans-serif; font-size:14px; font-weight:700; color:var(--navy);
      background:#E0E0E0; border:2px outset var(--gray); padding:6px 0; min-height:36px; cursor:pointer; }}
    .rngbar button.on{{ background:#C0C0FF; border-style:inset; }}
    .tiles{{ display:grid; grid-template-columns:repeat(3,1fr); gap:4px; margin:0 0 6px; }}
    @media (min-width:700px){{ .tiles{{ grid-template-columns:repeat(6,1fr); }} }}
    .tile{{ background:#000; border:2px inset var(--gray); padding:5px 4px; text-align:center; }}
    .tile .tl{{ font-family:Arial,sans-serif; font-size:10px; font-weight:700; color:#C0C0C0; text-transform:uppercase; }}
    .tile .tv{{ font-family:"Courier New",monospace; font-size:17px; font-weight:700; color:#33FF33; }}
    .tile.fwd .tv{{ color:#FFAA33; }}
    .src-note{{ font-size:11px; color:#333; margin:6px 2px 0; }}
    .back{{ margin:10px 6px 0; }}
    .back a{{ display:inline-block; font-family:Arial,sans-serif; font-weight:700; font-size:12px; color:#000;
      background:var(--gray); border:2px outset var(--gray); padding:3px 10px; text-decoration:none; }}
    .back a:active{{ border-style:inset; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="marquee"><span>{marquee}{marquee}</span></div>
    <h1 class="hero">{title}</h1>
    <div class="chart-card">
      <h2>Implied Growth and P/E</h2>
      <div class="chart-body">{_range_bar('growth-chart')}{growth_chart_html}</div>
    </div>
{callout_section}{eps_section}{growth_section}{eps_indexed_section}    <div class="chart-card">
      <h2>10-Year Statistics</h2>
      {timeframe_table_html}
    </div>
    <p class="back"><a href="index.html">Back to Dashboard</a></p>
  </div>
{measure_script}
</body>
</html>"""

def _forward_callout(fwd) -> str:
    """Return an HTML callout string summarising forward EPS growth, or '' if unavailable."""
    if not fwd or fwd.get("growth_this_fy") is None:
        return ""
    g1 = fwd["growth_this_fy"]
    g2 = fwd.get("growth_next_fy")
    src = (fwd.get("source") or "").lower()
    if src == "factset":
        # Whole-index professional consensus (no coverage caveat).
        parts = [f"Forward earnings growth (S&amp;P 500 consensus, FactSet): "
                 f"<b>{g1:+.1%}</b> this fiscal year"]
        if g2 is not None:
            parts.append(f", <b>{g2:+.1%}</b> next")
        parts.append(".")
        return "".join(parts)
    cov = fwd.get("coverage_weight") or 0
    parts = [f"Forward earnings growth (bottom-up): <b>{g1:+.1%}</b> this fiscal year"]
    if g2 is not None:
        parts.append(f", <b>{g2:+.1%}</b> next")
    parts.append(f". Based on {cov:.0%} of index weight.")
    return "".join(parts)

def _write_page(out_path: str, html: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

# Pinned plotly.js URL — same file ticker.html hardcodes, so the browser reuses
# one cached copy. NEVER pass the floating 'cdn' keyword here: it tracks the
# installed plotly python package, and CI once emitted plotly-3.7.0.min.js which
# the CDN 403s (unpublished) -> every chart on both growth pages silently blank.
_PLOTLY_JS_URL = "https://cdn.plot.ly/plotly-3.6.0.min.js"
_PLOT_CONFIG = {"displayModeBar": False, "scrollZoom": False, "responsive": True}


def _build_one(conn: sqlite3.Connection, ticker: str, df: pd.DataFrame) -> None:
    """Build a single valuation page for a ticker."""
    fwd = _latest_forward_eps(conn, ticker)
    df_d, df_w, df_m = _resample_frames(df)
    fig_growth = _growth_figure(df_d, df_w, df_m, ticker)
    _apply_retro_layout(fig_growth)
    chart_growth_html = to_html(fig_growth, include_plotlyjs=_PLOTLY_JS_URL, full_html=False, default_height="600px", div_id="growth-chart", config=_PLOT_CONFIG)
    fig_eps = _eps_figure(df_d, df_w, df_m, ticker, fwd=fwd)
    chart_eps_html = None
    if fig_eps is not None:
        _apply_retro_layout(fig_eps)
        chart_eps_html = to_html(fig_eps, include_plotlyjs=False, full_html=False,
                                 default_height="450px", div_id="eps-chart", config=_PLOT_CONFIG)
    fig_eps_idx = _eps_indexed_figure(df_d, df_w, df_m, ticker, fwd=fwd)
    chart_eps_idx_html = None
    if fig_eps_idx is not None:
        _apply_retro_layout(fig_eps_idx)
        chart_eps_idx_html = to_html(fig_eps_idx, include_plotlyjs=False, full_html=False,
                                     default_height="450px", div_id="eps-indexed-chart", config=_PLOT_CONFIG)
    eps_growth_html = ""
    if "eps" in df_d.columns:
        summary = _eps_growth_summary(df_d["eps"], _forward_growth_rates(conn, ticker))
        if summary:
            fig_g = _eps_growth_figure(summary)
            eps_growth_html = _growth_tiles_html(summary) + to_html(
                fig_g, include_plotlyjs=False, full_html=False, default_height="340px",
                div_id="eps-growth-chart", config=_PLOT_CONFIG)
    tf_table = _timeframe_table_html(df_d)
    callout = _forward_callout(fwd)
    page = _page_html(PAGE_TITLES[ticker], chart_growth_html, chart_eps_html,
                      chart_eps_idx_html, tf_table, callout, eps_growth_html)
    out_file = os.path.join(OUTPUT_DIR, OUTPUT_FILES[ticker])
    _write_page(out_file, page)

def generate_index_growth_pages(db_path: str = DB_PATH) -> None:
    """Entry point to generate pages for SPY and QQQ."""
    conn = sqlite3.connect(db_path)
    try:
        for ticker in ("SPY", "QQQ"):
            df = _load_series(conn, ticker)
            # Drop rows where all plotted series are missing
            subset = [c for c in ["ig", "ig_fwd", "eps"] if c in df.columns]
            if subset:
                df = df.dropna(subset=subset, how="all")
            _build_one(conn, ticker, df)
    finally:
        conn.close()

if __name__ == "__main__":
    generate_index_growth_pages()

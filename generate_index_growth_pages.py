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
                      coverage_weight, displayable
                 FROM Index_Forward_EPS_History
                WHERE ticker=? ORDER BY date_recorded DESC LIMIT 1""",
            (ticker.upper(),)).fetchone()
    except Exception:
        return None
    if not row or row[0] is None or not row[5]:
        return None
    return {"forward_eps_index": float(row[0]), "horizon_date": row[1],
            "growth_this_fy": row[2], "growth_next_fy": row[3],
            "coverage_weight": row[4]}


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

def _growth_figure(df_d: pd.DataFrame, df_w: pd.DataFrame, df_m: pd.DataFrame, ticker: str) -> go.Figure:
    """Build the Plotly figure for implied growth."""

    def mk_traces(df: pd.DataFrame, tag: str) -> List[go.Scatter]:
        traces: List[go.Scatter] = []
        if "ig" in df.columns:
            traces.append(
                go.Scatter(
                    x=df.index,
                    y=df["ig"],
                    name=f"TTM ({tag})",
                    mode="lines",
                    hovertemplate="%{y:.2f}%<extra></extra>",
                )
            )
        if "ig_fwd" in df.columns:
            traces.append(
                go.Scatter(
                    x=df.index,
                    y=df["ig_fwd"],
                    name=f"Forward ({tag})",
                    mode="lines",
                    hovertemplate="%{y:.2f}%<extra></extra>",
                )
            )
        return traces

    traces_d = mk_traces(df_d, "Daily")
    traces_w = mk_traces(df_w, "Weekly")
    traces_m = mk_traces(df_m, "Monthly")
    stat_lines = _stat_lines(df_d)

    fig = go.Figure(data=traces_d + traces_w + traces_m + stat_lines)
    n_d, n_w, n_m, n_s = len(traces_d), len(traces_w), len(traces_m), len(stat_lines)

    # default visibility: Weekly + stat lines
    vis_daily = [True] * n_d + [False] * n_w + [False] * n_m + [True] * n_s
    vis_weekly = [False] * n_d + [True] * n_w + [False] * n_m + [True] * n_s
    vis_monthly = [False] * n_d + [False] * n_w + [True] * n_m + [True] * n_s
    for i, v in enumerate(vis_weekly):
        fig.data[i].visible = v

    fig.update_layout(
        title=f"{ticker} — Implied Growth (TTM & Forward) — avg and ±1σ shown",
        margin=dict(l=50, r=50, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        xaxis=dict(
            title="Date",
            rangeselector=dict(
                buttons=list(
                    [
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(count=3, label="3M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(step="all", label="All"),
                    ]
                )
            ),
            rangeslider=dict(visible=True),
            type="date",
        ),
        yaxis=dict(title="Implied Growth (%)", side="left"),
        template=None,
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.0,
                xanchor="left",
                y=1.16,
                yanchor="top",
                buttons=[
                    dict(label="Daily", method="update", args=[{"visible": vis_daily}, {}]),
                    dict(label="Weekly", method="update", args=[{"visible": vis_weekly}, {}]),
                    dict(label="Monthly", method="update", args=[{"visible": vis_monthly}, {}]),
                ],
            )
        ],
    )
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

    def mk_traces(df: pd.DataFrame, tag: str) -> List[go.Scatter]:
        if "eps" not in df.columns:
            return []
        return [
            go.Scatter(
                x=df.index,
                y=df["eps"],
                name=f"EPS ({tag})",
                mode="lines",
                hovertemplate="$%{y:.2f}<extra></extra>",
            )
        ]

    traces_d = mk_traces(df_d, "Daily")
    traces_w = mk_traces(df_w, "Weekly")
    traces_m = mk_traces(df_m, "Monthly")

    if not any((traces_d, traces_w, traces_m)):
        return None

    # Decide y-axis type: log only when every daily EPS value is strictly positive.
    s_daily = df_d["eps"].dropna() if "eps" in df_d.columns else pd.Series(dtype=float)
    yaxis_type = "log" if (not s_daily.empty and (s_daily > 0).all()) else "linear"

    # Forecast trace (always visible — treated like stat lines in _growth_figure).
    forecast_traces: List[go.Scatter] = []
    if fwd is not None and not s_daily.empty:
        forecast_traces = [
            _eps_forecast_trace(s_daily.index[-1], float(s_daily.iloc[-1]), fwd)
        ]

    fig = go.Figure(data=traces_d + traces_w + traces_m + forecast_traces)
    n_d, n_w, n_m, n_f = (
        len(traces_d), len(traces_w), len(traces_m), len(forecast_traces)
    )

    vis_daily   = [True]  * n_d + [False] * n_w + [False] * n_m + [True] * n_f
    vis_weekly  = [False] * n_d + [True]  * n_w + [False] * n_m + [True] * n_f
    vis_monthly = [False] * n_d + [False] * n_w + [True]  * n_m + [True] * n_f
    for i, v in enumerate(vis_weekly):
        fig.data[i].visible = v

    fig.update_layout(
        title=f"{ticker} — EPS (TTM)",
        margin=dict(l=50, r=50, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        xaxis=dict(
            title="Date",
            rangeselector=dict(
                buttons=list(
                    [
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(count=3, label="3M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(step="all", label="All"),
                    ]
                )
            ),
            rangeslider=dict(visible=True),
            type="date",
        ),
        yaxis=dict(title="EPS (USD)", tickprefix="$", type=yaxis_type),
        template=None,
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.0,
                xanchor="left",
                y=1.16,
                yanchor="top",
                buttons=[
                    dict(label="Daily",   method="update", args=[{"visible": vis_daily},   {}]),
                    dict(label="Weekly",  method="update", args=[{"visible": vis_weekly},  {}]),
                    dict(label="Monthly", method="update", args=[{"visible": vis_monthly}, {}]),
                ],
            )
        ],
    )
    return fig

def _eps_indexed_figure(
    df_d: pd.DataFrame,
    df_w: pd.DataFrame,
    df_m: pd.DataFrame,
    ticker: str,
    fwd=None,
) -> Optional[go.Figure]:
    """Build an EPS chart normalised to 100 at the first observation.

    Returns None when the first EPS value is zero or negative (log axis
    would be undefined).  The y-axis is always log because the series is
    strictly positive by construction once the base guard passes.
    """
    b = df_d["eps"].dropna() if "eps" in df_d.columns else pd.Series(dtype=float)
    if b.empty or b.iloc[0] <= 0:
        return None
    base = float(b.iloc[0])

    def mk_traces(df: pd.DataFrame, tag: str) -> List[go.Scatter]:
        if "eps" not in df.columns:
            return []
        s = df["eps"].dropna()
        if s.empty:
            return []
        return [
            go.Scatter(
                x=s.index,
                y=s / base * 100,
                name=f"EPS Indexed ({tag})",
                mode="lines",
                hovertemplate="%{y:.1f}<extra></extra>",
            )
        ]

    traces_d = mk_traces(df_d, "Daily")
    traces_w = mk_traces(df_w, "Weekly")
    traces_m = mk_traces(df_m, "Monthly")

    # Forecast trace (always visible).
    forecast_traces: List[go.Scatter] = []
    if fwd is not None and not b.empty:
        last_eps_indexed = float(b.iloc[-1]) / base * 100
        forecast_traces = [
            _eps_forecast_trace(b.index[-1], last_eps_indexed, fwd, scale=100.0 / base)
        ]

    fig = go.Figure(data=traces_d + traces_w + traces_m + forecast_traces)
    n_d, n_w, n_m, n_f = (
        len(traces_d), len(traces_w), len(traces_m), len(forecast_traces)
    )

    vis_daily   = [True]  * n_d + [False] * n_w + [False] * n_m + [True] * n_f
    vis_weekly  = [False] * n_d + [True]  * n_w + [False] * n_m + [True] * n_f
    vis_monthly = [False] * n_d + [False] * n_w + [True]  * n_m + [True] * n_f
    for i, v in enumerate(vis_weekly):
        fig.data[i].visible = v

    fig.update_layout(
        title=f"{ticker} — EPS (Indexed to 100)",
        margin=dict(l=50, r=50, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        xaxis=dict(
            title="Date",
            rangeselector=dict(
                buttons=list(
                    [
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(count=3, label="3M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(step="all", label="All"),
                    ]
                )
            ),
            rangeslider=dict(visible=True),
            type="date",
        ),
        yaxis=dict(title="EPS (start=100)", type="log"),
        template=None,
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.0,
                xanchor="left",
                y=1.16,
                yanchor="top",
                buttons=[
                    dict(label="Daily",   method="update", args=[{"visible": vis_daily},   {}]),
                    dict(label="Weekly",  method="update", args=[{"visible": vis_weekly},  {}]),
                    dict(label="Monthly", method="update", args=[{"visible": vis_monthly}, {}]),
                ],
            )
        ],
    )
    return fig


def _page_html(title: str, growth_chart_html: str, eps_chart_html: Optional[str],
               eps_indexed_html: Optional[str], timeframe_table_html: str, callout: str) -> str:
    """Assemble the full HTML page with charts and stats table."""
    callout_section = ""
    if callout:
        callout_section = f'    <p class="callout">{callout}</p>\n'

    eps_section = ""
    if eps_chart_html:
        eps_section = f"""
    <div class="chart">
      {eps_chart_html}
      <div class="measure-readout" id="ro-eps">Click a point, then another, to measure % change.</div>
      <button onclick="clearMeasure('eps-chart')">Clear</button>
    </div>
"""

    eps_indexed_section = ""
    if eps_indexed_html:
        eps_indexed_section = f"""
    <div class="chart">
      {eps_indexed_html}
      <div class="measure-readout" id="ro-eps-indexed">Click a point, then another, to measure % change.</div>
      <button onclick="clearMeasure('eps-indexed-chart')">Clear</button>
    </div>
"""

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
  function init(){ attach('eps-chart','ro-eps'); attach('eps-indexed-chart','ro-eps-indexed'); }
  if(document.readyState!=='loading') init(); else document.addEventListener('DOMContentLoaded', init);
})();
function clearMeasure(id){ var gd=document.getElementById(id); if(gd&&gd.__clear) gd.__clear(); }
</script>"""

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
    .measure-readout {{ margin: 6px 0 4px; font-size: 0.9rem; color: #333; min-height: 1.4em; }}
    .callout {{ margin: 8px 0; padding: 8px 12px; background: #fffbe6; border-left: 4px solid #f5a623; border-radius: 4px; font-size: 0.95rem; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>{title}</h1>
    <div class="chart">
      {growth_chart_html}
    </div>
{callout_section}{eps_section}{eps_indexed_section}    {timeframe_table_html}
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
    cov = fwd.get("coverage_weight") or 0
    parts = [f"Forward earnings growth (bottom-up): <b>{g1:+.1%}</b> this fiscal year"]
    if g2 is not None:
        parts.append(f", <b>{g2:+.1%}</b> next")
    parts.append(f". Based on {cov:.0%} of index weight.")
    return "".join(parts)

def _write_page(out_path: str, html: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

def _build_one(conn: sqlite3.Connection, ticker: str, df: pd.DataFrame) -> None:
    """Build a single valuation page for a ticker."""
    fwd = _latest_forward_eps(conn, ticker)
    df_d, df_w, df_m = _resample_frames(df)
    fig_growth = _growth_figure(df_d, df_w, df_m, ticker)
    chart_growth_html = to_html(fig_growth, include_plotlyjs="cdn", full_html=False, default_height="600px")
    fig_eps = _eps_figure(df_d, df_w, df_m, ticker, fwd=fwd)
    chart_eps_html = None
    if fig_eps is not None:
        chart_eps_html = to_html(fig_eps, include_plotlyjs=False, full_html=False,
                                 default_height="450px", div_id="eps-chart")
    fig_eps_idx = _eps_indexed_figure(df_d, df_w, df_m, ticker, fwd=fwd)
    chart_eps_idx_html = None
    if fig_eps_idx is not None:
        chart_eps_idx_html = to_html(fig_eps_idx, include_plotlyjs=False, full_html=False,
                                     default_height="450px", div_id="eps-indexed-chart")
    tf_table = _timeframe_table_html(df_d)
    callout = _forward_callout(fwd)
    page = _page_html(PAGE_TITLES[ticker], chart_growth_html, chart_eps_html,
                      chart_eps_idx_html, tf_table, callout)
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

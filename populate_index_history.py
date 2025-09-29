#!/usr/bin/env python3
"""Populate SPY/QQQ P/E history with a ten year look-back.

This module downloads publicly available historical aggregates for the
S&P 500 (used as a SPY proxy) and Nasdaq-100 constituents (used as a QQQ
proxy) to extend the ``Index_PE_History`` and ``Index_Growth_History``
tables.  It avoids direct broker/finance APIs – which are often blocked
in automated environments – and instead relies on static CSV datasets
hosted on GitHub.

Two sources are used:

* ``datasets/s-and-p-500`` provides monthly S&P 500 price/earnings data.
* ``SheepBoss/.../nasdaq100_metrics_ratios`` stores yearly P/E ratios for
  Nasdaq-100 constituents, which we convert into a smoothed daily
  average for QQQ.

The script calculates the implied growth rate with the same formula used
elsewhere in the repository (``growth = yield * PE - 1``) by merging
historical ten year treasury yields that already live in
``economic_data``.

Running ``populate_index_history.py`` refreshes the two tables and
produces data suitable for the index growth charts that currently only
show a short window starting in July 2025.
"""

from __future__ import annotations

import io
import sqlite3
from datetime import date
from typing import Callable, Dict

import numpy as np
import pandas as pd
import requests


# ──────────────── configuration ───────────────────────────────────────────
DB_PATH = "Stock Data.db"
IMPLIED_GROWTH_TABLE = "Implied_Growth_History"
SPY_DATA_URL = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500/master/data/data.csv"
)
QQQ_PE_URL = (
    "https://raw.githubusercontent.com/SheepBoss/"
    "Project-on-ML-dataset-and-models-for-stock-performance-predictions-"
    "based-on-financial-ratios/master/nasdaq100_metrics_ratios.csv"
)
YEARS_DEFAULT = 10


FetchFunc = Callable[[str], str]


def _http_fetch(url: str, timeout: int = 30) -> str:
    """Return the UTF-8 text for *url* or raise a descriptive error."""

    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()

    # GitHub occasionally responds with an HTML page that states
    # "Binary files are not supported" when the ``raw`` endpoint is not
    # available.  That page renders as text/html even though it doesn't
    # contain the CSV payload we expect.  Surface a helpful error so the
    # caller knows the feed needs attention rather than letting Pandas
    # attempt to parse the HTML blob as a CSV.
    body = resp.text
    lowered = body.strip().lower()
    if "binary files are not supported" in lowered:
        raise RuntimeError(
            "Upstream source responded with 'Binary files are not supported'. "
            "Please verify the download URL (is it a GitHub raw URL?) or try again later."
        )

    content_type = resp.headers.get("content-type", "").lower()
    if "text" not in content_type and "," not in body:
        raise RuntimeError(
            f"Unexpected content-type '{content_type}' returned from {url}."
        )

    return body


def _to_daily(series: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Interpolate *series* to a daily frequency between *start* and *end*."""

    if series.empty:
        return series

    base = series.sort_index()
    # Ensure boundary values are present so interpolation covers the window.
    if start < base.index.min():
        base = pd.concat([pd.Series(base.iloc[0], index=[start]), base])
    if end > base.index.max():
        base = pd.concat([base, pd.Series(base.iloc[-1], index=[end])])

    idx = pd.date_range(start=start, end=end, freq="D")
    daily = (
        base.reindex(base.index.union(idx))
        .interpolate(method="time")
        .reindex(idx)
        .ffill()
    )
    daily.index.name = "Date"
    return daily


def _spy_series(
    start: pd.Timestamp,
    end: pd.Timestamp,
    fetch: FetchFunc = _http_fetch,
) -> pd.Series:
    """Return daily SPY P/E (TTM) values between *start* and *end*."""

    text = fetch(SPY_DATA_URL)
    df = pd.read_csv(io.StringIO(text))
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "SP500", "Earnings"])
    df = df.set_index("Date").sort_index()

    earnings = pd.to_numeric(df["Earnings"], errors="coerce")
    price = pd.to_numeric(df["SP500"], errors="coerce")
    pe = (price / earnings).replace([np.inf, -np.inf], np.nan).dropna()
    pe = pe[~pe.index.duplicated(keep="last")]
    pe.name = "PE"

    trimmed = pe.loc[(pe.index >= start - pd.DateOffset(months=1)) & (pe.index <= end)]
    return _to_daily(trimmed, start, end)


def _qqq_series(
    start: pd.Timestamp,
    end: pd.Timestamp,
    fetch: FetchFunc = _http_fetch,
) -> pd.Series:
    """Return a daily QQQ P/E (TTM) series between *start* and *end*."""

    text = fetch(QQQ_PE_URL)
    df = pd.read_csv(io.StringIO(text))

    # Yearly average P/E from constituent ratios.
    pe_cols: Dict[int, str] = {
        2017: "price_to_earnings_ratio_2017",
        2018: "price_to_earnings_ratio_2018",
        2019: "price_to_earnings_ratio_2019",
        2020: "price_to_earnings_ratio_2020",
        2021: "price_to_earnings_ratio_2021",
        2022: "price_to_earnings_ratio_2022",
        # ``_latest`` serves as the most recent estimate (2023+).
        2023: "price_to_earnings_ratio_latest",
        2024: "price_to_earnings_ratio_latest",
        2025: "price_to_earnings_ratio_latest",
    }

    year_map: Dict[int, float] = {}
    for year, col in pe_cols.items():
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if vals.empty:
            continue
        year_map[year] = float(vals.mean())

    if not year_map:
        raise RuntimeError("Unable to derive any QQQ P/E ratios from source data.")

    years = sorted(year_map)
    dates = [pd.Timestamp(year=year, month=7, day=1) for year in years]
    series = pd.Series([year_map[y] for y in years], index=dates, name="PE")

    return _to_daily(series, start, end)


def _load_yields(conn: sqlite3.Connection, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Return daily 10y treasury yields (decimal form) between dates."""

    query = (
        "SELECT date AS Date, value FROM economic_data "
        "WHERE indicator='DGS10' AND date BETWEEN ? AND ?"
    )
    df = pd.read_sql_query(query, conn, params=(start.date(), end.date()))
    if df.empty:
        raise RuntimeError("No DGS10 yield data available in economic_data table.")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.dropna(subset=["value"])
    yields = pd.to_numeric(df["value"], errors="coerce").dropna() / 100.0
    series = yields.groupby(df["Date"]).mean()
    return _to_daily(series, start, end)


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    cur = conn.cursor()
    cur.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    )
    return cur.fetchone() is not None


def _write_history(
    conn: sqlite3.Connection,
    ticker: str,
    pe_series: pd.Series,
    yield_series: pd.Series,
) -> None:
    """Upsert P/E and implied growth history for *ticker* into the DB."""

    merged = pd.concat([pe_series, yield_series], axis=1, join="inner").dropna()
    merged.columns = ["PE", "Yield"]
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna()
    if merged.empty:
        raise RuntimeError(f"No overlapping data for {ticker} to store.")

    merged["Growth"] = merged["Yield"] * merged["PE"] - 1.0
    rows = [
        (idx.strftime("%Y-%m-%d"), ticker, "TTM", float(row["PE"]))
        for idx, row in merged.iterrows()
    ]
    growth_rows = [
        (idx.strftime("%Y-%m-%d"), ticker, "TTM", float(row["Growth"]))
        for idx, row in merged.iterrows()
    ]
    implied_rows = [
        (ticker, "TTM", float(row["Growth"]), idx.strftime("%Y-%m-%d"))
        for idx, row in merged.iterrows()
    ]

    start_str = merged.index.min().strftime("%Y-%m-%d")
    end_str = merged.index.max().strftime("%Y-%m-%d")

    cur = conn.cursor()
    cur.execute(
        "DELETE FROM Index_PE_History WHERE Ticker=? AND PE_Type='TTM'",
        (ticker,),
    )
    cur.execute(
        "DELETE FROM Index_Growth_History WHERE Ticker=? AND Growth_Type='TTM'",
        (ticker,),
    )
    cur.executemany(
        "INSERT OR REPLACE INTO Index_PE_History(Date,Ticker,PE_Type,PE_Ratio)"
        " VALUES (?,?,?,?)",
        rows,
    )
    cur.executemany(
        "INSERT OR REPLACE INTO Index_Growth_History(Date,Ticker,Growth_Type,Implied_Growth)"
        " VALUES (?,?,?,?)",
        growth_rows,
    )

    if _table_exists(conn, IMPLIED_GROWTH_TABLE):
        cur.execute(
            f"DELETE FROM {IMPLIED_GROWTH_TABLE} "
            "WHERE ticker=? AND growth_type='TTM' AND date_recorded BETWEEN ? AND ?",
            (ticker, start_str, end_str),
        )
    else:
        cur.execute(
            f"CREATE TABLE IF NOT EXISTS {IMPLIED_GROWTH_TABLE} ("
            "ticker TEXT, growth_type TEXT, growth_value REAL, date_recorded TEXT)"
        )

    cur.executemany(
        f"INSERT OR REPLACE INTO {IMPLIED_GROWTH_TABLE} (ticker,growth_type,growth_value,date_recorded)"
        " VALUES (?,?,?,?)",
        implied_rows,
    )
    conn.commit()


def populate_index_history(
    *,
    db_path: str = DB_PATH,
    years: int = YEARS_DEFAULT,
    today: date | None = None,
    fetch: FetchFunc = _http_fetch,
) -> None:
    """Populate SPY & QQQ P/E + implied growth for the last *years* years."""

    if years <= 0:
        raise ValueError("years must be positive")

    today_dt = pd.Timestamp(today or date.today())
    start_dt = today_dt - pd.DateOffset(years=years)

    spy = _spy_series(start_dt, today_dt, fetch)
    qqq = _qqq_series(start_dt, today_dt, fetch)

    with sqlite3.connect(db_path) as conn:
        yields = _load_yields(conn, start_dt, today_dt)
        _write_history(conn, "SPY", spy, yields)
        _write_history(conn, "QQQ", qqq, yields)


if __name__ == "__main__":  # pragma: no cover - manual invocation
    populate_index_history()

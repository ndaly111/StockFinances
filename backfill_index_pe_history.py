#!/usr/bin/env python3
"""Backfill SPY/QQQ P/E history using Yahoo Finance earnings + price data.

This module downloads the last N years of daily closing prices alongside
quarterly EPS information in order to reconstruct historical trailing and
forward P/E ratios.  Results are persisted to the ``Index_PE_History`` table
and can subsequently be converted into implied growth via the existing
``backfill_index_growth`` helper.

Only lightweight helper functions are unit-tested; the network calls rely on
``yfinance`` and are expected to run in the production environment.
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Optional

import pandas as pd
import yfinance as yf


DB_PATH = "Stock Data.db"
DEFAULT_TICKERS: tuple[str, ...] = ("SPY", "QQQ")


@dataclass
class PEHistory:
    """Container for computed trailing/forward P/E rows."""

    date: pd.Timestamp
    pe_ttm: Optional[float]
    pe_forward: Optional[float]


def _ensure_tables(conn: sqlite3.Connection) -> None:
    """Guarantee the P/E history table exists."""

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS Index_PE_History (
            Date    TEXT,
            Ticker  TEXT,
            PE_Type TEXT,
            PE_Ratio REAL,
            PRIMARY KEY (Date, Ticker, PE_Type)
        );
        """
    )


def _has_history(conn: sqlite3.Connection, ticker: str, start_date: datetime) -> bool:
    """Return True if the DB already has history reaching *start_date*."""

    row = conn.execute(
        """
        SELECT MIN(Date)
          FROM Index_PE_History
         WHERE Ticker=? AND PE_Type='TTM'
        """,
        (ticker,),
    ).fetchone()
    if not row or not row[0]:
        return False
    try:
        earliest = datetime.strptime(row[0], "%Y-%m-%d")
    except ValueError:
        return False
    return earliest <= start_date


def _compute_eps_windows(raw: pd.DataFrame) -> pd.DataFrame:
    """Compute trailing and forward EPS aggregates from earnings history."""

    if raw is None or raw.empty:
        return pd.DataFrame(columns=["date", "ttm_eps", "forward_eps"])

    df = raw.copy()
    if df.index.name != "quarter":
        df.index.name = "quarter"
    df = df.sort_index()
    for col in ("epsActual", "epsEstimate"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = pd.Series(dtype=float)

    df["ttm_eps"] = df["epsActual"].rolling(window=4, min_periods=4).sum()
    df["forward_eps"] = df["epsEstimate"][::-1].rolling(window=4, min_periods=4).sum()[::-1]

    out = (
        df[["ttm_eps", "forward_eps"]]
        .dropna(how="all")
        .reset_index()
        .rename(columns={"quarter": "date"})
    )
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    return out.dropna(subset=["date"]).sort_values("date")


def _merge_prices_with_eps(prices: pd.DataFrame, eps: pd.DataFrame) -> pd.DataFrame:
    """Align daily closes with the most recent EPS aggregates."""

    if prices is None or prices.empty or eps is None or eps.empty:
        return pd.DataFrame(columns=["date", "pe_ttm", "pe_forward"])

    price_df = prices.copy()
    price_df["date"] = pd.to_datetime(price_df["date"], errors="coerce").dt.normalize()
    price_df = price_df.dropna(subset=["date", "close"]).sort_values("date")

    eps_df = eps.copy()
    eps_df["date"] = pd.to_datetime(eps_df["date"], errors="coerce").dt.normalize()
    eps_df = eps_df.dropna(subset=["date"]).sort_values("date")

    merged = pd.merge_asof(price_df, eps_df, on="date", direction="backward")
    if merged.empty:
        return pd.DataFrame(columns=["date", "pe_ttm", "pe_forward"])

    merged["pe_ttm"] = merged["close"] / merged["ttm_eps"]
    merged.loc[(merged["ttm_eps"] <= 0) | merged["ttm_eps"].isna(), "pe_ttm"] = pd.NA

    if "forward_eps" in merged.columns:
        merged["pe_forward"] = merged["close"] / merged["forward_eps"]
        merged.loc[
            (merged["forward_eps"] <= 0) | merged["forward_eps"].isna(), "pe_forward"
        ] = pd.NA
    else:
        merged["pe_forward"] = pd.NA

    keep_cols = ["date", "pe_ttm", "pe_forward"]
    return merged[keep_cols].dropna(subset=["pe_ttm"], how="all").sort_values("date")


def _download_prices(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch daily close prices via yfinance."""

    data = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        progress=False,
        interval="1d",
        auto_adjust=False,
    )
    if data.empty:
        return pd.DataFrame(columns=["date", "close"])
    data = data.reset_index()
    return data.rename(columns={"Date": "date", "Close": "close"})[["date", "close"]]


def _compute_history(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Return a dataframe of daily P/E values between *start* and *end*."""

    ticker_obj = yf.Ticker(ticker)
    earnings = ticker_obj.get_earnings_history()
    eps_df = _compute_eps_windows(earnings)
    prices = _download_prices(ticker, start, end)
    history = _merge_prices_with_eps(prices, eps_df)
    if history.empty:
        return history
    history = history[(history["date"] >= pd.Timestamp(start)) & (history["date"] <= pd.Timestamp(end))]
    return history


def _iter_pe_rows(df: pd.DataFrame) -> Iterable[PEHistory]:
    for _, row in df.iterrows():
        date = pd.to_datetime(row["date"], errors="coerce")
        if pd.isna(date):
            continue
        pe_ttm = row.get("pe_ttm")
        pe_forward = row.get("pe_forward") if "pe_forward" in row else None

        def _clean(value: Optional[float]) -> Optional[float]:
            if pd.isna(value):
                return None
            if isinstance(value, (float, int)) and math.isfinite(float(value)):
                return float(value)
            return None

        yield PEHistory(date=date, pe_ttm=_clean(pe_ttm), pe_forward=_clean(pe_forward))


def _upsert_history(conn: sqlite3.Connection, ticker: str, history: pd.DataFrame) -> int:
    """Insert P/E values into the database, returning number of rows written."""

    if history is None or history.empty:
        return 0

    rows = []
    for item in _iter_pe_rows(history):
        date_str = item.date.strftime("%Y-%m-%d")
        if item.pe_ttm is not None:
            rows.append((date_str, ticker, "TTM", item.pe_ttm))
        if item.pe_forward is not None:
            rows.append((date_str, ticker, "Forward", item.pe_forward))

    if not rows:
        return 0

    conn.executemany(
        """
        INSERT OR REPLACE INTO Index_PE_History (Date, Ticker, PE_Type, PE_Ratio)
        VALUES (?, ?, ?, ?)
        """,
        rows,
    )
    return len(rows)


def backfill_index_pe_history(
    years: int = 10,
    tickers: Iterable[str] = DEFAULT_TICKERS,
    db_path: str = DB_PATH,
    force: bool = False,
) -> None:
    """Populate ``Index_PE_History`` with up to *years* of history for each ticker."""

    end = datetime.utcnow().date()
    start = end - timedelta(days=int(years * 365.25))
    end_dt = datetime.combine(end + timedelta(days=1), datetime.min.time())
    start_dt = datetime.combine(start, datetime.min.time())

    with sqlite3.connect(db_path) as conn:
        _ensure_tables(conn)

        for ticker in tickers:
            if not force and _has_history(conn, ticker, start_dt):
                print(f"[backfill_index_pe_history] {ticker}: history already covers {start_dt.date()}")
                continue

            try:
                history = _compute_history(ticker, start_dt, end_dt)
            except Exception as exc:  # pragma: no cover - network errors
                print(f"[backfill_index_pe_history] Failed to compute history for {ticker}: {exc}")
                continue

            inserted = _upsert_history(conn, ticker, history)
            print(f"[backfill_index_pe_history] {ticker}: wrote {inserted} rows")

        conn.commit()


if __name__ == "__main__":  # pragma: no cover - manual invocation
    backfill_index_pe_history()

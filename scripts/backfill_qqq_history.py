#!/usr/bin/env python3
"""Backfill ~10 years of QQQ P/E, EPS, growth, and price history.

QQQ rows in the DB only go back to 2025-07-04. This script extends the
history backward by combining:

* QQQ daily close from yfinance (back to *years* years ago).
* QQQ trailing P/E approximated from Nasdaq-100 constituent yearly P/E
  averages (SheepBoss CSV, same source used by
  populate_index_history.py), interpolated to daily.
* 10-year Treasury yield from Treasury_Yield_History.

Implied growth uses the same formula as the rest of the repo:
    growth = (PE / 10) ** 0.1 + yield - 1

This script is upsert-only (``INSERT OR REPLACE`` keyed on
``(Date, Ticker, *_Type)``); it never deletes existing rows and never
touches tickers other than QQQ. By default it only writes dates that do
not already have a TTM PE row, preserving newer data captured by the
daily collector.
"""

from __future__ import annotations

import argparse
import io
import sqlite3
from datetime import date

import numpy as np
import pandas as pd
import requests
import yfinance as yf


DB_PATH = "Stock Data.db"
QQQ_PE_URL = (
    "https://raw.githubusercontent.com/SheepBoss/"
    "Project-on-ML-dataset-and-models-for-stock-performance-predictions-"
    "based-on-financial-ratios/master/nasdaq100_metrics_ratios.csv"
)
TICKER = "QQQ"
IMPLIED_GROWTH_TABLE = "Implied_Growth_History"
EPS_TYPE_IMPLIED = "IMPLIED_FROM_PE"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default=DB_PATH)
    p.add_argument("--years", type=int, default=10, help="Years of history to backfill (default 10).")
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, overwrite existing QQQ TTM rows in the backfill window. Default: skip dates already present.",
    )
    return p.parse_args()


def _yearly_pe_from_constituents() -> pd.Series:
    """Return a yearly Series of average Nasdaq-100 constituent trailing P/E."""

    resp = requests.get(QQQ_PE_URL, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))

    columns = {
        2017: "price_to_earnings_ratio_2017",
        2018: "price_to_earnings_ratio_2018",
        2019: "price_to_earnings_ratio_2019",
        2020: "price_to_earnings_ratio_2020",
        2021: "price_to_earnings_ratio_2021",
        2022: "price_to_earnings_ratio_2022",
        2023: "price_to_earnings_ratio_latest",
        2024: "price_to_earnings_ratio_latest",
        2025: "price_to_earnings_ratio_latest",
    }
    rows = {}
    for year, col in columns.items():
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if vals.empty:
            continue
        # Trim extreme outliers (negative P/E or absurd values) before averaging.
        vals = vals[(vals > 0) & (vals < 500)]
        rows[year] = float(vals.mean())

    if not rows:
        raise RuntimeError("Unable to derive QQQ P/E ratios from constituent CSV.")
    return pd.Series({pd.Timestamp(year=y, month=7, day=1): v for y, v in rows.items()}).sort_index()


def _interp_daily(pe_yearly: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    base = pe_yearly.copy()
    if start < base.index.min():
        base = pd.concat([pd.Series(base.iloc[0], index=[start]), base])
    if end > base.index.max():
        base = pd.concat([base, pd.Series(base.iloc[-1], index=[end])])
    base = base[~base.index.duplicated(keep="last")].sort_index()
    idx = pd.date_range(start=start, end=end, freq="D")
    daily = base.reindex(base.index.union(idx)).interpolate(method="time").reindex(idx).ffill()
    daily.name = "PE"
    return daily


def _qqq_price_history(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    hist = yf.Ticker(TICKER).history(start=start, end=end + pd.Timedelta(days=1), auto_adjust=False, actions=False)
    if hist.empty:
        raise RuntimeError("yfinance returned no QQQ history.")
    if getattr(hist.index, "tz", None) is not None:
        hist.index = hist.index.tz_localize(None)
    s = hist["Close"].astype(float)
    s.index = pd.DatetimeIndex(s.index.date)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    s.name = "Close"
    return s


def _load_yields(conn: sqlite3.Connection, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    df = pd.read_sql_query(
        "SELECT Date, TenYr FROM Treasury_Yield_History WHERE Date BETWEEN ? AND ? ORDER BY Date",
        conn,
        params=(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")),
    )
    if df.empty:
        raise RuntimeError("Treasury_Yield_History is empty in the requested window.")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.dropna()
    s = df.set_index("Date")["TenYr"].astype(float)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    # Forward-fill to daily so non-trading days inherit the prior yield.
    full_idx = pd.date_range(start=s.index.min(), end=end, freq="D")
    s = s.reindex(full_idx).ffill()
    return s


def _existing_dates(conn: sqlite3.Connection) -> set[str]:
    cur = conn.execute(
        "SELECT Date FROM Index_PE_History WHERE Ticker=? AND PE_Type='TTM'",
        (TICKER,),
    )
    return {r[0] for r in cur.fetchall()}


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    cur = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,))
    return cur.fetchone() is not None


def backfill(db_path: str, years: int, overwrite: bool) -> int:
    today = pd.Timestamp(date.today())
    start = today - pd.DateOffset(years=years)

    print(f"Window: {start.date()} → {today.date()}  ({years} years)")

    yearly_pe = _yearly_pe_from_constituents()
    print(f"Constituent yearly P/E points: {len(yearly_pe)}  ({yearly_pe.index.min().year}–{yearly_pe.index.max().year})")
    pe_daily = _interp_daily(yearly_pe, start, today)
    px_daily = _qqq_price_history(start, today)
    print(f"yfinance QQQ prices: {len(px_daily)} rows ({px_daily.index.min().date()} → {px_daily.index.max().date()})")

    with sqlite3.connect(db_path) as conn:
        yields = _load_yields(conn, start, today)

        existing = _existing_dates(conn)
        # Align price and PE to the price index (business days). Yields ffilled to daily.
        df = pd.DataFrame({"Close": px_daily}).join(pe_daily, how="left").join(yields.rename("Yield"), how="left")
        df = df.dropna()
        df = df[(df["Close"] > 0) & (df["PE"] > 0)]

        if not overwrite:
            df = df[~df.index.strftime("%Y-%m-%d").isin(existing)]
            print(f"Skipping {len(existing.intersection(set(d.strftime('%Y-%m-%d') for d in df.index)))} dates already present.")

        if df.empty:
            print("No new rows to write.")
            return 0

        df["EPS"] = df["Close"] / df["PE"]
        df["Growth"] = (df["PE"] / 10.0) ** 0.1 + df["Yield"] - 1.0

        rows_pe   = [(d.strftime("%Y-%m-%d"), TICKER, "TTM", float(r["PE"]))       for d, r in df.iterrows()]
        rows_eps  = [(d.strftime("%Y-%m-%d"), TICKER, EPS_TYPE_IMPLIED, float(r["EPS"]))  for d, r in df.iterrows()]
        rows_gr   = [(d.strftime("%Y-%m-%d"), TICKER, "TTM", float(r["Growth"]))   for d, r in df.iterrows()]
        rows_px   = [(d.strftime("%Y-%m-%d"), TICKER, float(r["Close"]))           for d, r in df.iterrows()]

        cur = conn.cursor()
        cur.executemany(
            "INSERT OR REPLACE INTO Index_PE_History(Date,Ticker,PE_Type,PE_Ratio) VALUES (?,?,?,?)",
            rows_pe,
        )
        cur.executemany(
            "INSERT OR REPLACE INTO Index_EPS_History(Date,Ticker,EPS_Type,EPS) VALUES (?,?,?,?)",
            rows_eps,
        )
        cur.executemany(
            "INSERT OR REPLACE INTO Index_Growth_History(Date,Ticker,Growth_Type,Implied_Growth) VALUES (?,?,?,?)",
            rows_gr,
        )
        cur.executemany(
            "INSERT OR REPLACE INTO Index_Price_History(Date,Ticker,Close) VALUES (?,?,?)",
            rows_px,
        )

        if _table_exists(conn, IMPLIED_GROWTH_TABLE):
            cur.executemany(
                f"INSERT OR REPLACE INTO {IMPLIED_GROWTH_TABLE}"
                "(ticker,growth_type,growth_value,date_recorded) VALUES (?,?,?,?)",
                [(TICKER, "TTM", float(r["Growth"]), d.strftime("%Y-%m-%d")) for d, r in df.iterrows()],
            )

        conn.commit()
        print(f"Wrote {len(rows_pe)} rows to each of: PE_History, EPS_History, Growth_History, Price_History.")
        return 0


if __name__ == "__main__":
    args = _parse_args()
    raise SystemExit(backfill(args.db, args.years, args.overwrite))

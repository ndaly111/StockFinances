#!/usr/bin/env python3
"""Backfill implied index EPS history for SPY and QQQ.

This script derives EPS from historical P/E ratios and closing prices.
It leaves P/E and implied growth tables untouched.
"""

import argparse
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf


def _parse_args():
    parser = argparse.ArgumentParser(description="Backfill index EPS history from P/E and price data.")
    parser.add_argument("--db", default="Stock Data.db", help="Path to SQLite database")
    parser.add_argument("--years", type=int, default=10, help="Number of years to backfill")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=("SPY", "QQQ"),
        help="Index tickers to backfill (default: SPY QQQ)",
    )
    return parser.parse_args()


def _ensure_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS Index_EPS_History (
            Date    TEXT,
            Ticker  TEXT,
            EPS_Type TEXT,
            EPS     REAL,
            PRIMARY KEY (Date, Ticker, EPS_Type)
        );
        CREATE INDEX IF NOT EXISTS idx_Index_EPS_History_ticker_type_date
            ON Index_EPS_History (Ticker, EPS_Type, Date);

        CREATE TABLE IF NOT EXISTS Index_Price_History (
            Date   TEXT,
            Ticker TEXT,
            Close  REAL,
            PRIMARY KEY (Date, Ticker)
        );
        CREATE INDEX IF NOT EXISTS idx_Index_Price_History_ticker_date
            ON Index_Price_History (Ticker, Date);
        """
    )


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    cur = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (name,),
    )
    return cur.fetchone() is not None


def _load_pe_history(conn: sqlite3.Connection, tickers: tuple[str, ...], years: int) -> pd.DataFrame:
    q = (
        "SELECT Date, Ticker, PE_Ratio "
        "FROM Index_PE_History WHERE PE_Type='TTM' AND Ticker IN (%s)"
    )
    placeholders = ",".join(["?"] * len(tickers))
    df = pd.read_sql_query(q % placeholders, conn, params=tickers)
    if df.empty:
        return df

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "PE_Ratio"])
    df["PE_Ratio"] = pd.to_numeric(df["PE_Ratio"], errors="coerce")
    df = df.dropna(subset=["PE_Ratio"])
    cutoff = pd.Timestamp(datetime.now().date()) - pd.DateOffset(years=years)
    df = df[df["Date"] >= cutoff]
    return df


def _fetch_and_store_prices(
    conn: sqlite3.Connection,
    ticker: str,
    start_date: datetime,
    end_date: datetime,
) -> None:
    hist = yf.Ticker(ticker).history(
        start=start_date, end=end_date, auto_adjust=False, actions=False
    )
    if hist.empty:
        return
    if getattr(hist.index, "tz", None) is not None:
        hist.index = hist.index.tz_localize(None)
    rows = [
        (idx.strftime("%Y-%m-%d"), ticker, float(row.Close))
        for idx, row in hist.iterrows()
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO Index_Price_History(Date, Ticker, Close) VALUES (?,?,?)",
        rows,
    )


def _ensure_prices(conn: sqlite3.Connection, df: pd.DataFrame, tickers: tuple[str, ...]) -> None:
    if df.empty:
        return

    for ticker in tickers:
        tk_df = df[df["Ticker"] == ticker]
        if tk_df.empty:
            continue

        start = tk_df["Date"].min().to_pydatetime()
        end = tk_df["Date"].max().to_pydatetime() + timedelta(days=1)

        existing = pd.read_sql_query(
            """
            SELECT Date FROM Index_Price_History
             WHERE Ticker=? AND Date BETWEEN ? AND ?
            """,
            conn,
            params=(ticker, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")),
        )
        have_dates = set(existing["Date"]) if not existing.empty else set()

        needed_dates = set(tk_df["Date"].dt.strftime("%Y-%m-%d"))
        missing = needed_dates - have_dates
        if missing:
            _fetch_and_store_prices(conn, ticker, start, end)


def _upsert_eps(conn: sqlite3.Connection, df: pd.DataFrame, tickers: tuple[str, ...]) -> int:
    if df.empty:
        return 0

    inserted = 0
    for ticker in tickers:
        tk_df = df[df["Ticker"] == ticker]
        if tk_df.empty:
            continue

        prices = pd.read_sql_query(
            "SELECT Date, Close FROM Index_Price_History WHERE Ticker=?",
            conn,
            params=(ticker,),
        )
        if prices.empty:
            continue

        price_map = dict(zip(prices["Date"], prices["Close"]))
        rows = []
        for _, row in tk_df.iterrows():
            date_key = row["Date"].strftime("%Y-%m-%d")
            price = price_map.get(date_key)
            pe = row["PE_Ratio"]
            if price is None or pe in (None, 0):
                continue
            try:
                price_f = float(price)
                pe_f = float(pe)
                if price_f <= 0 or pe_f <= 0:
                    continue
                eps = price_f / pe_f
            except Exception:
                continue
            rows.append((date_key, ticker, "TTM", eps))

        conn.executemany(
            "INSERT OR REPLACE INTO Index_EPS_History(Date, Ticker, EPS_Type, EPS) "
            "VALUES (?,?,?,?)",
            rows,
        )
        inserted += len(rows)
    return inserted


def main():
    args = _parse_args()
    db_path = Path(args.db)
    tickers = tuple(args.tickers)
    if not db_path.exists():
        raise SystemExit(f"Database not found: {db_path}")

    with sqlite3.connect(db_path) as conn:
        _ensure_tables(conn)
        if not _table_exists(conn, "Index_PE_History"):
            raise SystemExit("Index_PE_History table not found; nothing to backfill.")

        pe_hist = _load_pe_history(conn, tickers, args.years)
        if pe_hist.empty:
            print("[backfill_index_eps] No P/E history found for SPY/QQQ.")
            return

        _ensure_prices(conn, pe_hist, tickers)
        inserted = _upsert_eps(conn, pe_hist, tickers)
        conn.commit()

    print(f"[backfill_index_eps] Upserted {inserted} EPS rows into Index_EPS_History.")


if __name__ == "__main__":
    main()

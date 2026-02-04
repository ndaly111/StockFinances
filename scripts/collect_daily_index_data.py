#!/usr/bin/env python3
"""Collect daily SPY/QQQ price and EPS data.

Fetches current price and trailing P/E from yfinance, derives EPS,
and stores in the database for P/E history tracking.

Run daily via GitHub Actions to keep index valuation data current.
"""

from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import yfinance as yf

DEFAULT_DB = "Stock Data.db"
TICKERS = ["SPY", "QQQ"]

# SPY trades at ~1/10th of S&P 500 index value
# Scale EPS accordingly for consistency with historical data
SPY_INDEX_DIVISOR = 10.0
QQQ_INDEX_DIVISOR = 4.0  # QQQ is ~1/4th of Nasdaq-100


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect daily index price and EPS data")
    parser.add_argument("--db", default=DEFAULT_DB, help="Path to SQLite DB")
    parser.add_argument("--tickers", nargs="+", default=TICKERS, help="Tickers to collect")
    return parser.parse_args()


def _ensure_tables(conn: sqlite3.Connection) -> None:
    """Ensure required tables exist."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS Index_Price_History_Monthly (
            Date   TEXT NOT NULL,
            Ticker TEXT NOT NULL,
            Close  REAL,
            PRIMARY KEY (Date, Ticker)
        );

        CREATE TABLE IF NOT EXISTS Index_EPS_History (
            Date     TEXT NOT NULL,
            Ticker   TEXT NOT NULL,
            EPS_Type TEXT NOT NULL,
            EPS      REAL,
            PRIMARY KEY (Date, Ticker, EPS_Type)
        );

        CREATE TABLE IF NOT EXISTS Index_PE_History (
            Date     TEXT NOT NULL,
            Ticker   TEXT NOT NULL,
            PE_Type  TEXT NOT NULL,
            PE_Ratio REAL,
            PRIMARY KEY (Date, Ticker, PE_Type)
        );

        CREATE TABLE IF NOT EXISTS Daily_Index_Snapshots (
            Date       TEXT NOT NULL,
            Ticker     TEXT NOT NULL,
            Close      REAL,
            PE_Ratio   REAL,
            EPS_TTM    REAL,
            Market_Cap REAL,
            Fetched_At TEXT NOT NULL,
            PRIMARY KEY (Date, Ticker)
        );
    """)
    conn.commit()


def _get_divisor(ticker: str) -> float:
    """Get the index divisor for a ticker."""
    if ticker.upper() == "SPY":
        return SPY_INDEX_DIVISOR
    elif ticker.upper() == "QQQ":
        return QQQ_INDEX_DIVISOR
    return 1.0


def collect_ticker_data(ticker: str, conn: sqlite3.Connection) -> dict | None:
    """Fetch and store data for a single ticker."""
    try:
        stock = yf.Ticker(ticker)
        # Ensure info is always a dict (stock.info can be None or non-dict)
        info = stock.info if isinstance(stock.info, dict) else {}

        # Get current price
        price = info.get("regularMarketPrice") or info.get("previousClose")
        if not price:
            # Fallback to history
            hist = stock.history(period="1d")
            if not hist.empty:
                price = float(hist["Close"].iloc[-1])

        if not price:
            print(f"[{ticker}] Could not get price")
            return None

        # Get trailing P/E
        trailing_pe = info.get("trailingPE")
        if not trailing_pe or trailing_pe <= 0:
            print(f"[{ticker}] No valid trailing P/E available")
            return None

        # Derive EPS from price and P/E
        # EPS = Price / P/E
        raw_eps = price / trailing_pe

        # Scale EPS to match index-level for consistency with historical data
        divisor = _get_divisor(ticker)
        scaled_eps = raw_eps * divisor

        market_cap = info.get("marketCap")

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        now = datetime.now(timezone.utc).isoformat()

        # Store in Daily_Index_Snapshots
        conn.execute("""
            INSERT OR REPLACE INTO Daily_Index_Snapshots
            (Date, Ticker, Close, PE_Ratio, EPS_TTM, Market_Cap, Fetched_At)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (today, ticker.upper(), price, trailing_pe, scaled_eps, market_cap, now))

        # Also update Index_PE_History with current P/E
        conn.execute("""
            INSERT OR REPLACE INTO Index_PE_History
            (Date, Ticker, PE_Type, PE_Ratio)
            VALUES (?, ?, 'TTM', ?)
        """, (today, ticker.upper(), trailing_pe))

        # Update Index_EPS_History
        conn.execute("""
            INSERT OR REPLACE INTO Index_EPS_History
            (Date, Ticker, EPS_Type, EPS)
            VALUES (?, ?, 'TTM_DAILY', ?)
        """, (today, ticker.upper(), scaled_eps))

        # Update monthly price table (for the current month)
        month_start = datetime.now(timezone.utc).replace(day=1).strftime("%Y-%m-%d")
        conn.execute("""
            INSERT OR REPLACE INTO Index_Price_History_Monthly
            (Date, Ticker, Close)
            VALUES (?, ?, ?)
        """, (month_start, ticker.upper(), price))

        conn.commit()

        result = {
            "ticker": ticker,
            "date": today,
            "price": price,
            "pe_ratio": trailing_pe,
            "eps_ttm": scaled_eps,
            "raw_eps": raw_eps,
        }

        print(f"[{ticker}] Price: ${price:.2f}, P/E: {trailing_pe:.2f}, EPS (scaled): ${scaled_eps:.2f}")
        return result

    except Exception as e:
        print(f"[{ticker}] Error: {e}")
        return None


def main() -> int:
    args = _parse_args()

    print("=" * 60)
    print("Daily Index Data Collection")
    print(f"Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        return 1

    conn = sqlite3.connect(str(db_path))
    _ensure_tables(conn)

    results = []
    for ticker in args.tickers:
        result = collect_ticker_data(ticker, conn)
        if result:
            results.append(result)

    conn.close()

    print("\n" + "=" * 60)
    print(f"Collected data for {len(results)}/{len(args.tickers)} tickers")
    print("=" * 60)

    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())

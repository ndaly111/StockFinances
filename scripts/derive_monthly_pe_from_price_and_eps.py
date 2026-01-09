#!/usr/bin/env python3
"""Derive monthly P/E from monthly prices and reported EPS."""

from __future__ import annotations

import argparse
import sqlite3

import pandas as pd

DEFAULT_DB = "Stock Data.db"
DEFAULT_TICKER = "SPY"
EPS_TYPE = "TTM_REPORTED"
PE_TYPE = "TTM_DERIVED_MONTHLY"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Derive monthly P/E from price and EPS")
    parser.add_argument("--db", default=DEFAULT_DB, help="Path to SQLite DB")
    parser.add_argument("--ticker", default=DEFAULT_TICKER, help="Ticker symbol to derive")
    return parser.parse_args()


def _ensure_pe_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS Index_PE_History (
            Date     TEXT NOT NULL,
            Ticker   TEXT NOT NULL,
            PE_Type  TEXT NOT NULL,
            PE_Ratio REAL,
            PRIMARY KEY (Date, Ticker, PE_Type)
        )
        """
    )


def derive_monthly_pe(*, db_path: str, ticker: str = DEFAULT_TICKER) -> int:
    """Derive monthly P/E for *ticker* and return row count."""

    ticker = ticker.upper().strip()

    with sqlite3.connect(db_path) as conn:
        _ensure_pe_table(conn)
        prices = pd.read_sql_query(
            """
            SELECT Date, Close FROM Index_Price_History_Monthly
            WHERE Ticker=?
            """,
            conn,
            params=(ticker,),
        )
        eps = pd.read_sql_query(
            """
            SELECT Date, EPS FROM Index_EPS_History
            WHERE Ticker=? AND EPS_Type=?
            """,
            conn,
            params=(ticker, EPS_TYPE),
        )

        if prices.empty or eps.empty:
            return 0

        prices["Date"] = pd.to_datetime(prices["Date"], errors="coerce")
        eps["Date"] = pd.to_datetime(eps["Date"], errors="coerce")
        prices = prices.dropna(subset=["Date", "Close"])
        eps = eps.dropna(subset=["Date", "EPS"])
        prices["Close"] = pd.to_numeric(prices["Close"], errors="coerce")
        eps["EPS"] = pd.to_numeric(eps["EPS"], errors="coerce")
        prices = prices.dropna(subset=["Close"])
        eps = eps.dropna(subset=["EPS"])

        if prices.empty or eps.empty:
            return 0

        prices["Month"] = prices["Date"].dt.to_period("M")
        eps["Month"] = eps["Date"].dt.to_period("M")
        prices = prices.sort_values("Date").groupby("Month", as_index=False).last()
        eps = eps.sort_values("Date").groupby("Month", as_index=False).last()
        merged = prices.merge(eps, on="Month", how="inner", suffixes=("_price", "_eps"))
        merged = merged[(merged["Close"] > 0) & (merged["EPS"] > 0)]
        if merged.empty:
            return 0

        merged["PE_Ratio"] = merged["Close"] / merged["EPS"]
        merged["Date"] = merged["Month"].dt.to_timestamp("M").dt.strftime("%Y-%m-%d")

        rows = [
            (row["Date"], ticker.upper(), PE_TYPE, float(row["PE_Ratio"]))
            for _, row in merged.iterrows()
        ]
        conn.executemany(
            "INSERT OR REPLACE INTO Index_PE_History (Date, Ticker, PE_Type, PE_Ratio)"
            " VALUES (?,?,?,?)",
            rows,
        )
        conn.commit()

    return len(rows)


def main() -> None:
    args = _parse_args()
    inserted = derive_monthly_pe(db_path=args.db, ticker=args.ticker)
    print(f"Inserted {inserted} derived monthly P/E rows into Index_PE_History.")


if __name__ == "__main__":
    main()

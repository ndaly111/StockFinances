#!/usr/bin/env python3
"""Load monthly index price history from a CSV into Index_Price_History_Monthly."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

DEFAULT_DB = "Stock Data.db"
DEFAULT_CSV = "data/spy_price_history_monthly_1993_present.csv"
DEFAULT_TICKER = "SPY"
DEFAULT_DATE_COLUMN = "Date"
DEFAULT_CLOSE_COLUMN = "Close"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load monthly price CSV into Index_Price_History_Monthly")
    parser.add_argument("--db", default=DEFAULT_DB, help="Path to SQLite DB")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Path to monthly price CSV file")
    parser.add_argument("--ticker", default=DEFAULT_TICKER, help="Ticker symbol to store")
    parser.add_argument(
        "--date-column",
        default=DEFAULT_DATE_COLUMN,
        help="Column name in CSV that contains dates",
    )
    parser.add_argument(
        "--close-column",
        default=DEFAULT_CLOSE_COLUMN,
        help="Column name in CSV that contains close prices",
    )
    return parser.parse_args()


def _ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS Index_Price_History_Monthly (
            Date   TEXT NOT NULL,
            Ticker TEXT NOT NULL,
            Close  REAL,
            PRIMARY KEY (Date, Ticker)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_Index_Price_History_Monthly_ticker_date
            ON Index_Price_History_Monthly (Ticker, Date)
        """
    )


def _normalize_price_frame(
    df: pd.DataFrame, date_column: str, close_column: str
) -> pd.DataFrame:
    if date_column not in df.columns:
        raise ValueError(f"Price CSV is missing required '{date_column}' column")
    if close_column not in df.columns:
        raise ValueError(f"Price CSV is missing required '{close_column}' column")

    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
    df[close_column] = pd.to_numeric(df[close_column], errors="coerce")
    df = df.dropna(subset=[date_column, close_column])
    if df.empty:
        return df

    df = (
        df.groupby(df[date_column].dt.normalize())[close_column]
        .last()
        .reset_index()
        .rename(columns={close_column: "Close", date_column: "Date"})
    )
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    return df


def load_price_csv(
    *,
    db_path: str,
    csv_path: str,
    ticker: str = DEFAULT_TICKER,
    date_column: str = DEFAULT_DATE_COLUMN,
    close_column: str = DEFAULT_CLOSE_COLUMN,
) -> int:
    """Load monthly prices from *csv_path* into *db_path* and return row count."""

    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"Price CSV not found: {csv_file}")

    df = pd.read_csv(csv_file)
    df = _normalize_price_frame(df, date_column, close_column)
    if df.empty:
        return 0

    rows = [
        (row["Date"], ticker.upper(), float(row["Close"]))
        for _, row in df.iterrows()
    ]

    with sqlite3.connect(db_path) as conn:
        _ensure_table(conn)
        conn.executemany(
            "INSERT OR REPLACE INTO Index_Price_History_Monthly (Date, Ticker, Close)"
            " VALUES (?,?,?)",
            rows,
        )
        conn.commit()

    return len(rows)


def main() -> None:
    args = _parse_args()
    inserted = load_price_csv(
        db_path=args.db,
        csv_path=args.csv,
        ticker=args.ticker,
        date_column=args.date_column,
        close_column=args.close_column,
    )
    print(f"Inserted {inserted} price rows into Index_Price_History_Monthly.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Load monthly EPS history from a CSV into Index_EPS_History.

The bundled SPY CSV is a monthly S&P 500 (SPY proxy) trailing twelve-month
EPS series; we store it under EPS_Type=TTM_REPORTED so it stays distinct from
implied EPS derived from price/PE ratios.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

DEFAULT_DB = "Stock Data.db"
DEFAULT_CSV = "data/spy_monthly_eps_1970_present.csv"
DEFAULT_TICKER = "SPY"
DEFAULT_COLUMN = "SPY_EPS"
EPS_TYPE_TTM_REPORTED = "TTM_REPORTED"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load monthly EPS CSV into Index_EPS_History")
    parser.add_argument("--db", default=DEFAULT_DB, help="Path to SQLite DB")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Path to EPS CSV file")
    parser.add_argument("--ticker", default=DEFAULT_TICKER, help="Ticker symbol to store")
    parser.add_argument(
        "--eps-type",
        default=EPS_TYPE_TTM_REPORTED,
        help="EPS_Type string to store in Index_EPS_History",
    )
    parser.add_argument(
        "--column",
        default=DEFAULT_COLUMN,
        help="Column name in CSV that contains EPS values",
    )
    return parser.parse_args()


def _ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS Index_EPS_History (
            Date     TEXT NOT NULL,
            Ticker   TEXT NOT NULL,
            EPS_Type TEXT NOT NULL,
            EPS      REAL,
            PRIMARY KEY (Date, Ticker, EPS_Type)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_Index_EPS_History_ticker_type_date
            ON Index_EPS_History (Ticker, EPS_Type, Date)
        """
    )


def _normalize_eps_frame(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if "Date" not in df.columns:
        raise ValueError("EPS CSV is missing required 'Date' column")
    if column not in df.columns:
        raise ValueError(f"EPS CSV is missing required '{column}' column")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["Date", column])
    if df.empty:
        return df

    df = (
        df.groupby(df["Date"].dt.normalize())[column]
        .last()
        .reset_index()
        .rename(columns={column: "EPS"})
    )
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    return df


def load_eps_csv(
    *,
    db_path: str,
    csv_path: str,
    ticker: str = DEFAULT_TICKER,
    eps_type: str = EPS_TYPE_TTM_REPORTED,
    column: str = DEFAULT_COLUMN,
) -> int:
    """Load EPS data from *csv_path* into *db_path* and return row count."""

    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"EPS CSV not found: {csv_file}")

    df = pd.read_csv(csv_file)
    df = _normalize_eps_frame(df, column)
    if df.empty:
        return 0

    rows = [
        (row["Date"], ticker.upper(), eps_type, float(row["EPS"]))
        for _, row in df.iterrows()
    ]

    with sqlite3.connect(db_path) as conn:
        _ensure_table(conn)
        conn.executemany(
            "INSERT OR REPLACE INTO Index_EPS_History (Date, Ticker, EPS_Type, EPS)"
            " VALUES (?,?,?,?)",
            rows,
        )
        conn.commit()

    return len(rows)


def main() -> None:
    args = _parse_args()
    inserted = load_eps_csv(
        db_path=args.db,
        csv_path=args.csv,
        ticker=args.ticker,
        eps_type=args.eps_type,
        column=args.column,
    )
    print(f"Inserted {inserted} EPS rows into Index_EPS_History.")


if __name__ == "__main__":
    main()

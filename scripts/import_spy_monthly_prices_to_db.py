#!/usr/bin/env python3
"""
Import SPY monthly price history CSV into the SQLite DB WITHOUT overwriting existing daily data.

What this script does:
- Reads a CSV with Date + Close (or Price).
- Connects to your SQLite DB (default: Stock Data.db).
- Inserts into a table that stores index prices (default: Index_Price_History).
- Preserves existing daily history by:
    (a) Only inserting rows older than the earliest existing SPY row in the DB, and
    (b) Using INSERT OR IGNORE (never replaces existing rows).

Expected DB schema (minimum):
    Table: Index_Price_History
    Columns: Date (TEXT), Ticker (TEXT), Close (REAL)
    Primary key typically: (Date, Ticker)

Note: By default, this script expects the table to already exist. Use
--create-table to create a minimal compatible table if needed.
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="Stock Data.db", help="Path to SQLite DB (default: Stock Data.db)")
    parser.add_argument(
        "--csv",
        default="data/spy_price_history_monthly_1993_present.csv",
        help="Path to SPY monthly price CSV committed in repo",
    )
    parser.add_argument("--ticker", default="SPY", help="Ticker to store (default: SPY)")
    parser.add_argument(
        "--table",
        default="Index_Price_History",
        help="Target price table name (default: Index_Price_History)",
    )
    parser.add_argument(
        "--create-table",
        action="store_true",
        help="Create the target table if it does not exist (default: false)",
    )
    return parser.parse_args()


def parse_date(value: str) -> date:
    value = (value or "").strip()
    fmts = ("%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d")
    for fmt in fmts:
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(value).date()
    except Exception as exc:
        raise ValueError(f"Unrecognized date: {value!r}") from exc


def parse_float(value: str) -> float:
    value = (value or "").strip().replace(",", "")
    return float(value)


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


def ensure_table(conn: sqlite3.Connection, table: str, allow_create: bool) -> None:
    if table_exists(conn, table):
        return
    if not allow_create:
        raise SystemExit(f"Expected table not found: {table}. Use --create-table to create it.")
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table} (
            Date   TEXT,
            Ticker TEXT,
            Close  REAL,
            PRIMARY KEY (Date, Ticker)
        );
        """
    )
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_ticker_date ON {table}(Ticker, Date);")
    conn.commit()


def earliest_existing_date(conn: sqlite3.Connection, table: str, ticker: str) -> Optional[date]:
    row = conn.execute(
        f"SELECT MIN(Date) FROM {table} WHERE Ticker=?",
        (ticker,),
    ).fetchone()
    if not row or not row[0]:
        return None
    return parse_date(row[0])


def detect_columns(fieldnames: list[str]) -> Tuple[str, str]:
    lower = {c.strip().lower(): c for c in fieldnames}

    date_col = lower.get("date")
    if not date_col:
        raise ValueError(f"CSV missing a Date column. Found headers: {fieldnames}")

    close_col = lower.get("close") or lower.get("price") or lower.get("adj close") or lower.get("adjclose")
    if not close_col:
        raise ValueError(f"CSV missing a Close/Price column. Found headers: {fieldnames}")

    return date_col, close_col


def main() -> None:
    args = parse_args()
    db_path = Path(args.db)
    csv_path = Path(args.csv)
    ticker = args.ticker.upper()
    table = args.table

    if not db_path.exists():
        raise SystemExit(f"DB not found: {db_path}")
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    with sqlite3.connect(db_path) as conn:
        ensure_table(conn, table, args.create_table)

        cutoff = earliest_existing_date(conn, table, ticker)
        print(f"[import] target table = {table}")
        print(f"[import] ticker       = {ticker}")
        print(f"[import] db           = {db_path}")
        print(f"[import] csv          = {csv_path}")
        print(
            f"[import] earliest existing {ticker} date in DB = {cutoff.isoformat() if cutoff else 'None (no rows yet)'}"
        )

        before_count = conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE Ticker=?",
            (ticker,),
        ).fetchone()[0]

        to_insert = []
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise SystemExit("CSV has no header row.")
            date_col, close_col = detect_columns(reader.fieldnames)

            for row in reader:
                row_date = parse_date(row[date_col])
                if cutoff and row_date >= cutoff:
                    continue
                close = parse_float(row[close_col])
                to_insert.append((row_date.isoformat(), ticker, close))

        print(f"[import] rows read from CSV (after cutoff filter) = {len(to_insert)}")

        if to_insert:
            before_changes = conn.total_changes
            conn.executemany(
                f"INSERT OR IGNORE INTO {table}(Date, Ticker, Close) VALUES (?,?,?)",
                to_insert,
            )
            conn.commit()
            inserted = conn.total_changes - before_changes
        else:
            inserted = 0

        after_count = conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE Ticker=?",
            (ticker,),
        ).fetchone()[0]

        print(f"[import] rows actually inserted = {inserted}")
        print(f"[import] rows before/after for {ticker} = {before_count} -> {after_count}")

        minmax = conn.execute(
            f"SELECT MIN(Date), MAX(Date) FROM {table} WHERE Ticker=?",
            (ticker,),
        ).fetchone()
        print(f"[import] {ticker} date range in DB now = {minmax[0]} -> {minmax[1]}")


if __name__ == "__main__":
    main()

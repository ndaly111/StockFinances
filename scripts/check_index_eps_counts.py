#!/usr/bin/env python3
"""Print row counts for SPY/QQQ EPS history."""

import argparse
import sqlite3
from pathlib import Path


def _parse_args():
    parser = argparse.ArgumentParser(description="Print row counts for SPY/QQQ EPS history.")
    parser.add_argument("--db", default="Stock Data.db", help="Path to SQLite database")
    return parser.parse_args()


def main():
    args = _parse_args()
    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"Database not found: {db_path}")

    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='Index_EPS_History'"
        )
        if cur.fetchone() is None:
            print("Index_EPS_History table not found.")
            return
        cur.execute(
            """
            SELECT Ticker, EPS_Type, COUNT(*) FROM Index_EPS_History
             WHERE Ticker IN ('SPY','QQQ')
            GROUP BY Ticker, EPS_Type
            ORDER BY Ticker, EPS_Type
            """
        )
        rows = cur.fetchall()
    if not rows:
        print("No EPS rows found for SPY/QQQ.")
        return

    for ticker, eps_type, count in rows:
        print(f"{ticker} ({eps_type}): {count} rows in Index_EPS_History")


if __name__ == "__main__":
    main()

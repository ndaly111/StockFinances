#!/usr/bin/env python3
"""
Print the entire Index_Growth_History table for SPY and QQQ
(ordered by Date, Growth_Type).

No data is modified.
"""

import sqlite3
from textwrap import dedent

DB_PATH = "Stock Data.db"           # adjust if your DB lives elsewhere
TICKERS = ("SPY", "QQQ")            # extend this tuple if needed

def main():
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        rows = cur.execute(
            """
            SELECT Date, Ticker, Growth_Type, Growth
            FROM   Index_Growth_History
            WHERE  Ticker IN (?, ?)
            ORDER  BY Date, Ticker, Growth_Type
            """,
            TICKERS
        ).fetchall()

    if not rows:
        print("No matching rows found.")
        return

    # ---- pretty-print -------------------------------------------------
    header = "{:<12} {:<6} {:<10} {:>10}".format("Date", "Ticker", "Type", "Growth %")
    print(header)
    print("-" * len(header))

    for r in rows:
        print("{:<12} {:<6} {:<10} {:>9.2f}".format(
            r["Date"],
            r["Ticker"],
            r["Growth_Type"],
            r["Growth"] * 100      # convert to %
        ))

if __name__ == "__main__":
    main()

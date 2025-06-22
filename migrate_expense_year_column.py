#!/usr/bin/env python3
"""
One-time migration: add `year_int` to ExpenseData and backfill it from year_label.
"""

import os
import sqlite3
import sys

DB_PATH = os.getenv("DB_PATH", "Stock Data.db")
TABLE_NAME = "ExpenseData"
TEXT_COL   = "year_label"
INT_COL    = "year_int"

def column_exists(cursor, table, column):
    cursor.execute(f"PRAGMA table_info({table});")
    cols = [row[1] for row in cursor.fetchall()]
    return column in cols

def main():
    if not os.path.exists(DB_PATH):
        print(f"ERROR: Database file not found at {DB_PATH}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()

    # 1) Add INT_COL if missing
    if not column_exists(cur, TABLE_NAME, INT_COL):
        print(f"Adding column `{INT_COL}` to `{TABLE_NAME}`...")
        cur.execute(f"ALTER TABLE {TABLE_NAME} ADD COLUMN {INT_COL} INTEGER;")
    else:
        print(f"Column `{INT_COL}` already exists, skipping ALTER TABLE.")

    # 2) Backfill: convert any four-digit year strings to ints
    print(f"Backfilling `{INT_COL}` from `{TEXT_COL}` for YYYY labels...")
    cur.execute(f"""
        UPDATE {TABLE_NAME}
           SET {INT_COL} = CAST({TEXT_COL} AS INTEGER)
         WHERE {TEXT_COL} GLOB '[0-9][0-9][0-9][0-9]';
    """)
    print(f"Rows affected: {conn.total_changes}")

    conn.commit()
    conn.close()
    print("Migration complete.")

if __name__ == "__main__":
    main()

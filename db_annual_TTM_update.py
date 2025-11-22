#!/usr/bin/env python3
# db_annual_TTM_update.py — simple version, same DB reference style as your other scripts

import sqlite3
import logging
import os
import shutil
from datetime import datetime

# ───── CONFIG ─────
DB_PATH     = "Stock Data.db"   # same style as other scripts
TICKER      = "CMG" #replace ticker with AAA so you dont break the db
SPLIT_RATIO = 50               # 50-for-1 split
# ──────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def backup_db(db_path):
    """Create a timestamped backup of the database."""
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = f"{db_path}.backup-{ts}"
    shutil.copy2(db_path, backup_path)
    logging.info(f"Backup created: {backup_path}")

def connect(db_path):
    """Connect to SQLite database."""
    if not os.path.exists(db_path):
        logging.error(f"Database file not found: {db_path}")
        return None
    return sqlite3.connect(db_path)

def list_tables(conn):
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    return [r[0] for r in cur.fetchall()]

def has_column(conn, table, colname):
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info('{table}');")
    return any(row[1].lower() == colname.lower() for row in cur.fetchall())

def detect_ticker_col(conn, table):
    for name in ("Symbol", "Ticker"):
        if has_column(conn, table, name):
            return name
    return None

def detect_date_col(conn, table):
    for name in ("Date", "ReportDate", "AsOfDate"):
        if has_column(conn, table, name):
            return name
    return None

def preview_rows(conn, table, ticker_col, date_col, ticker, label):
    cur = conn.cursor()
    order = f'ORDER BY "{date_col}" DESC' if date_col else ""
    cur.execute(
        f'SELECT {date_col + "," if date_col else ""} EPS FROM "{table}" '
        f'WHERE "{ticker_col}"=? AND EPS IS NOT NULL AND TRIM(EPS)<>"" {order} LIMIT 3;',
        (ticker,)
    )
    rows = cur.fetchall()
    if rows:
        logging.info(f"[{table}] {label} sample EPS rows for {ticker}: {rows}")

def update_eps(conn, table, ticker_col, ticker, ratio):
    cur = conn.cursor()
    cur.execute(f"""
        UPDATE "{table}"
        SET EPS = CAST(EPS AS REAL) / ?
        WHERE "{ticker_col}" = ?
          AND EPS IS NOT NULL
          AND TRIM(EPS) <> ''
    """, (ratio, ticker))
    return cur.rowcount or 0

def main():
    if SPLIT_RATIO == 0:
        logging.error("Split ratio cannot be zero.")
        return

    conn = connect(DB_PATH)
    if conn is None:
        return

    try:
        backup_db(DB_PATH)
        tables = list_tables(conn)
        total = 0

        # Preview BEFORE
        for tbl in tables:
            if has_column(conn, tbl, "EPS"):
                tcol = detect_ticker_col(conn, tbl)
                if tcol:
                    preview_rows(conn, tbl, tcol, detect_date_col(conn, tbl), TICKER, "BEFORE")

        # Update
        with conn:
            for tbl in tables:
                if has_column(conn, tbl, "EPS"):
                    tcol = detect_ticker_col(conn, tbl)
                    if not tcol:
                        continue
                    changed = update_eps(conn, tbl, tcol, TICKER, SPLIT_RATIO)
                    if changed:
                        logging.info(f"[{tbl}] EPS adjusted for {changed} row(s).")
                        total += changed

        if total == 0:
            logging.warning("No rows updated. Check ticker symbol or EPS location.")
        else:
            logging.info(f"Total rows updated: {total}")

        # Preview AFTER
        for tbl in tables:
            if has_column(conn, tbl, "EPS"):
                tcol = detect_ticker_col(conn, tbl)
                if tcol:
                    preview_rows(conn, tbl, tcol, detect_date_col(conn, tbl), TICKER, "AFTER")

    finally:
        conn.close()

if __name__ == "__main__":
    main()

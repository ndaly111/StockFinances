# scripts/load_sp500_index_series.py
# Purpose: One-time loader to insert SPY P/E (TTM) and implied growth into SQLite,
# matching tables Index_PE_History and Index_Growth_History.

import argparse
import csv
import os
import sqlite3
from datetime import datetime

# -----------------------------
# CONFIG — EDIT THESE IF NEEDED
# -----------------------------
DB_PATH = os.environ.get("INDEX_DB_PATH", "Stock Data.db")
PE_CSV_PATH = os.environ.get("SP500_PE_CSV", "data/sp500_daily_pe_filled.csv")
GROWTH_CSV_PATH = os.environ.get("SP500_GROWTH_CSV", "data/sp500_implied_growth.csv")

TICKER = "SPY"
PE_TYPE = "TTM"
GROWTH_TYPE = "TTM"

# -----------------------------
# Helpers
# -----------------------------
def ymd(date_str):
    """Normalize date to YYYY-MM-DD."""
    # Accept common formats; default to as-is if already correct.
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    # Fallback: try pandas-ish date parts (e.g., 2015-10-05 00:00:00)
    try:
        return datetime.fromisoformat(date_str.replace("Z", "")).strftime("%Y-%m-%d")
    except Exception:
        # Last resort: trust incoming value (lets INSERT OR REPLACE dedupe dates it can)
        return date_str

def ensure_tables(conn):
    """
    Create the tables if they don't exist yet, matching your schema.
    Composite PKs ensure idempotent UPSERT via INSERT OR REPLACE.
    """
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS Index_PE_History (
            Date      TEXT NOT NULL,
            Ticker    TEXT NOT NULL,
            PE_Type   TEXT NOT NULL,
            PE_Ratio  REAL,
            PRIMARY KEY (Date, Ticker, PE_Type)
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS Index_Growth_History (
            Date           TEXT NOT NULL,
            Ticker         TEXT NOT NULL,
            Growth_Type    TEXT NOT NULL,
            Implied_Growth REAL,
            PRIMARY KEY (Date, Ticker, Growth_Type)
        );
    """)
    conn.commit()

def load_pe_rows(pe_csv_path):
    """
    Expecting columns: DATE, PE
    Returns list of tuples: (Date, Ticker, PE_Type, PE_Ratio)
    """
    rows = []
    with open(pe_csv_path, newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        need = {"DATE", "PE"}
        missing = need - set(r.fieldnames or [])
        if missing:
            raise ValueError(f"{pe_csv_path} missing columns: {sorted(missing)}")
        for rec in r:
            d = ymd(rec["DATE"])
            pe = rec["PE"].strip() if rec["PE"] is not None else ""
            if pe == "" or pe.lower() == "nan":
                continue  # skip blanks
            try:
                pe_val = float(pe)
            except ValueError:
                continue
            rows.append((d, TICKER, PE_TYPE, pe_val))
    return rows

def load_growth_rows(growth_csv_path):
    """
    Expecting columns: DATE, Implied_Growth
    Returns list of tuples: (Date, Ticker, Growth_Type, Implied_Growth)
    """
    rows = []
    with open(growth_csv_path, newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        need = {"DATE", "Implied_Growth"}
        missing = need - set(r.fieldnames or [])
        if missing:
            raise ValueError(f"{growth_csv_path} missing columns: {sorted(missing)}")
        for rec in r:
            d = ymd(rec["DATE"])
            g = rec["Implied_Growth"].strip() if rec["Implied_Growth"] is not None else ""
            if g == "" or g.lower() == "nan":
                continue
            try:
                g_val = float(g)
            except ValueError:
                continue
            rows.append((d, TICKER, GROWTH_TYPE, g_val))
    return rows

def insert_pe(conn, rows, dry_run=False):
    if not rows:
        return 0
    if dry_run:
        return len(rows)
    cur = conn.cursor()
    cur.executemany("""
        INSERT OR REPLACE INTO Index_PE_History (Date, Ticker, PE_Type, PE_Ratio)
        VALUES (?, ?, ?, ?);
    """, rows)
    conn.commit()
    return len(rows)

def insert_growth(conn, rows, dry_run=False):
    if not rows:
        return 0
    if dry_run:
        return len(rows)
    cur = conn.cursor()
    cur.executemany("""
        INSERT OR REPLACE INTO Index_Growth_History (Date, Ticker, Growth_Type, Implied_Growth)
        VALUES (?, ?, ?, ?);
    """, rows)
    conn.commit()
    return len(rows)

def backup_db(db_path):
    if not os.path.exists(db_path):
        return None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{db_path}.pre_index_load_backup_{ts}"
    with open(db_path, "rb") as src, open(backup_path, "wb") as dst:
        dst.write(src.read())
    return backup_path

def main():
    parser = argparse.ArgumentParser(description="One-time loader for SPY P/E and implied growth")
    parser.add_argument("--db", default=DB_PATH, help="Path to SQLite DB (default: Stock Data.db)")
    parser.add_argument("--pe_csv", default=PE_CSV_PATH, help="Path to sp500_daily_pe_filled.csv")
    parser.add_argument("--growth_csv", default=GROWTH_CSV_PATH, help="Path to sp500_implied_growth.csv")
    parser.add_argument("--dry-run", action="store_true", help="Validate and show counts without writing")
    args = parser.parse_args()

    if not os.path.exists(args.pe_csv):
        raise FileNotFoundError(f"PE CSV not found: {args.pe_csv}")
    if not os.path.exists(args.growth_csv):
        raise FileNotFoundError(f"Growth CSV not found: {args.growth_csv}")

    pe_rows = load_pe_rows(args.pe_csv)
    growth_rows = load_growth_rows(args.growth_csv)

    print(f"[INFO] Prepared {len(pe_rows)} P/E rows and {len(growth_rows)} growth rows.")
    if args.dry_run:
        print("[DRY RUN] No DB changes will be made.")
        return

    # Backup DB once before writing
    bkp = backup_db(args.db)
    if bkp:
        print(f"[INFO] DB backup created: {bkp}")

    conn = sqlite3.connect(args.db, timeout=30)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        ensure_tables(conn)

        n1 = insert_pe(conn, pe_rows, dry_run=False)
        n2 = insert_growth(conn, growth_rows, dry_run=False)
        print(f"[DONE] Inserted/updated {n1} P/E rows, {n2} growth rows into '{args.db}'.")
    finally:
        conn.close()

if __name__ == "__main__":
    main()

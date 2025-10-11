# scripts/load_sp500_index_series.py
# Purpose: One-time loader to insert SPY P/E (TTM) and implied growth into SQLite,
# matching tables Index_PE_History and Index_Growth_History.  The script reads a
# daily S&P 500 P/E series and a 10 year treasury yield series, aligns them by
# date, derives the implied growth via
#
#     growth = ((PE / 10) ** 0.1) + treasury_yield - 1
#
# and persists both the raw P/E values and the calculated growth series.

import argparse
import csv
import os
import sqlite3
from datetime import datetime
from typing import Dict, Iterable, Tuple

# -----------------------------
# CONFIG — EDIT THESE IF NEEDED
# -----------------------------
DB_PATH = os.environ.get("INDEX_DB_PATH", "Stock Data.db")
PE_CSV_PATH = os.environ.get("SP500_PE_CSV", "data/sp500_daily_pe_filled.csv")
YIELD_CSV_PATH = os.environ.get("TREASURY_YIELD_CSV", "data/treasury_10y_yield.csv")

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

def _norm(name: str) -> str:
    """Normalise a column heading for loose matching."""

    return "".join(ch for ch in name.upper() if ch.isalnum())


def _resolve_column(fieldnames: Iterable[str], *candidates: str) -> str:
    mapping: Dict[str, str] = {_norm(name): name for name in fieldnames if name}
    for candidate in candidates:
        key = _norm(candidate)
        if key in mapping:
            return mapping[key]
    raise ValueError(
        f"Unable to locate any of {candidates} in columns {list(fieldnames)}"
    )


def load_pe_rows(pe_csv_path):
    """
    Expecting columns: DATE, PE
    Returns list of tuples: (Date, Ticker, PE_Type, PE_Ratio)
    """
    rows = []
    with open(pe_csv_path, newline="", encoding="utf-8-sig") as f:
        # Read using DictReader, but if it doesn't find a header, derive it from the first non-blank line.
        r = csv.DictReader(f, skipinitialspace=True)
        if not r.fieldnames:
            f.seek(0)
            header = None
            for line in f:
                if line.strip():  # find first non-empty line
                    header = line.strip().lstrip("\ufeff")  # remove BOM if present
                    break
            if header is None:
                return []  # empty file
            fieldnames = [c.strip() for c in header.split(",")]
            f.seek(0)
            r = csv.DictReader(f, fieldnames=fieldnames, skipinitialspace=True)
            next(r, None)  # skip header row

        date_col = _resolve_column(
            r.fieldnames,
            "DATE",
            "Date",
            "observation_date",
            "Observation Date",
        )
        pe_col = _resolve_column(
            r.fieldnames,
            "PE",
            "P/E",
            "PE Ratio",
            "PE_Ratio",
            "PE_RATIO",
        )
        for rec in r:
            d = ymd(rec[date_col])
            pe_raw = rec.get(pe_col, "")
            pe = pe_raw.strip() if pe_raw is not None else ""
            if pe == "" or pe.lower() == "nan":
                continue  # skip blanks
            try:
                pe_val = float(pe)
            except ValueError:
                continue
            rows.append((d, TICKER, PE_TYPE, pe_val))
    return rows

def load_yield_map(yield_csv_path: str) -> Dict[str, float]:
    """Return mapping of YYYY-MM-DD ➜ 10 year yield (decimal)."""

    yields: Dict[str, float] = {}
    with open(yield_csv_path, newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f, skipinitialspace=True)
        if not r.fieldnames:
            f.seek(0)
            header = None
            for line in f:
                if line.strip():
                    header = line.strip().lstrip("\ufeff")
                    break
            if header is None:
                return {}
            fieldnames = [c.strip() for c in header.split(",")]
            f.seek(0)
            r = csv.DictReader(f, fieldnames=fieldnames, skipinitialspace=True)
            next(r, None)

        date_col = _resolve_column(
            r.fieldnames,
            "DATE",
            "Date",
            "observation_date",
            "Observation Date",
        )
        yield_col = _resolve_column(
            r.fieldnames,
            "Yield",
            "Rate",
            "10Y",
            "TenYear",
            "DGS10",
            "Treasury_Yield",
        )

        for rec in r:
            d = ymd(rec[date_col])
            y_raw = rec.get(yield_col, "")
            y = y_raw.strip() if y_raw is not None else ""
            if not y:
                continue
            y = y.replace("%", "")
            try:
                y_val = float(y)
            except ValueError:
                continue
            if y_val > 1:
                y_val /= 100.0
            yields[d] = y_val
    return yields


def compute_growth_rows(
    pe_rows: Iterable[Tuple[str, str, str, float]],
    yield_map: Dict[str, float],
) -> Tuple[list[Tuple[str, str, str, float]], int]:
    """Return (growth_rows, missing_count) from *pe_rows* and *yield_map*."""

    rows: list[Tuple[str, str, str, float]] = []
    missing = 0
    for d, ticker, gtype, pe_val in pe_rows:
        y = yield_map.get(d)
        if y is None:
            missing += 1
            continue
        growth = ((pe_val / 10.0) ** 0.1) + y - 1.0
        rows.append((d, ticker, gtype, growth))
    return rows, missing

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
    parser.add_argument(
        "--yield_csv",
        default=YIELD_CSV_PATH,
        help="Path to 10 year treasury yield CSV (default: data/treasury_10y_yield.csv)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and show counts without writing")
    args = parser.parse_args()

    if not os.path.exists(args.pe_csv):
        raise FileNotFoundError(f"PE CSV not found: {args.pe_csv}")
    if not os.path.exists(args.yield_csv):
        raise FileNotFoundError(f"Treasury yield CSV not found: {args.yield_csv}")

    pe_rows = load_pe_rows(args.pe_csv)
    yield_map = load_yield_map(args.yield_csv)
    growth_rows, missing = compute_growth_rows(pe_rows, yield_map)

    if missing:
        print(
            f"[WARN] Skipped {missing} P/E rows without matching 10y yield entries."
        )

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

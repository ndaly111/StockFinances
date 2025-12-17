#!/usr/bin/env python3
"""
Apply a historical stock split to existing database records.

This script adjusts split-sensitive fields (EPS and Shares_Outstanding) for rows
before a provided effective date, records the event in the Splits table, and
logs before/after samples to make verification easy.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sqlite3
from datetime import date, datetime
from typing import Iterable, List, Sequence

from split_utils import ensure_splits_table

DEFAULT_DB_PATH = "Stock Data.db"
DATE_CANDIDATES = ("Date", "Quarter", "ReportDate", "AsOfDate")
TICKER_CANDIDATES = ("Symbol", "Ticker")
EPS_COLUMNS = {"EPS", "TTM_EPS"}
SHARES_COLUMNS = {"Shares_Outstanding"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manually apply a stock split to stored data.")
    parser.add_argument("ticker", help="Ticker symbol to adjust (e.g., AAPL)")
    parser.add_argument("ratio", type=float, help="Split ratio (e.g., 4 for a 4-for-1 split)")
    parser.add_argument(
        "effective_date",
        type=str,
        help="Effective date for the split (YYYY-MM-DD). Rows on/after this date are left untouched.",
    )
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH, help=f"SQLite database path (default: {DEFAULT_DB_PATH})")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing them.")
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt.")
    parser.add_argument(
        "--tables",
        nargs="+",
        help="Optional allowlist of tables to update. When omitted, all tables are inspected.",
    )
    parser.add_argument("--no-backup", action="store_true", help="Skip creating an automatic pre-change backup.")
    return parser.parse_args()


def _validate_ratio(ratio: float) -> float:
    if ratio <= 0:
        raise ValueError("Split ratio must be greater than zero.")
    return ratio


def _parse_date(raw: str) -> date:
    try:
        return datetime.fromisoformat(raw).date()
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"Invalid date '{raw}'. Expected YYYY-MM-DD.") from exc


def _confirm(args: argparse.Namespace, ratio: float, split_date: date) -> None:
    if args.yes:
        return
    prompt = (
        f"Apply a {ratio:.4g}-for-1 split for {args.ticker.upper()} effective {split_date.isoformat()} "
        f"to database '{args.db_path}'? Type 'yes' to continue: "
    )
    if input(prompt).strip().lower() != "yes":
        raise SystemExit("Aborted by user.")


def _get_tables(cur: sqlite3.Cursor, allowlist: Iterable[str] | None) -> List[str]:
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    all_tables = [row[0] for row in cur.fetchall()]
    if allowlist:
        allowed = set(allowlist)
        return [t for t in all_tables if t in allowed]
    return all_tables


def _columns(cur: sqlite3.Cursor, table: str) -> List[str]:
    cur.execute(f'PRAGMA table_info("{table}");')
    return [row[1] for row in cur.fetchall()]


def _first_present(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _sample_rows(
    cur: sqlite3.Cursor,
    table: str,
    ticker_col: str,
    date_col: str,
    ticker: str,
    split_date: date,
    fields: Sequence[str],
    label: str,
) -> None:
    select_cols = [date_col] + [c for c in fields if c != date_col]
    col_expr = ", ".join(f'"{col}"' for col in select_cols)
    cur.execute(
        f'SELECT {col_expr} FROM "{table}" WHERE "{ticker_col}"=? AND "{date_col}" < ? ORDER BY "{date_col}" DESC LIMIT 5;',
        (ticker, split_date.isoformat()),
    )
    rows = cur.fetchall()
    if not rows:
        logging.info("[%s] %s sample: no matching rows.", table, label)
        return
    header = " | ".join(select_cols)
    logging.info("[%s] %s sample (%s):", table, label, header)
    for row in rows:
        logging.info("    %s", " | ".join(str(item) for item in row))


def _eligible_rowcount(
    cur: sqlite3.Cursor,
    table: str,
    ticker_col: str,
    date_col: str,
    ticker: str,
    split_date: date,
    target_columns: Sequence[str],
) -> int:
    predicates = [f'"{col}" IS NOT NULL' for col in target_columns]
    extra_clause = f" AND ({' OR '.join(predicates)})" if predicates else ""
    cur.execute(
        f'SELECT COUNT(*) FROM "{table}" '
        f'WHERE "{ticker_col}"=? AND "{date_col}" < ?{extra_clause};',
        (ticker, split_date.isoformat()),
    )
    return int(cur.fetchone()[0] or 0)


def _update_table(
    cur: sqlite3.Cursor,
    table: str,
    ticker_col: str,
    date_col: str,
    ticker: str,
    split_date: date,
    split_ratio: float,
    eps_cols: Sequence[str],
    shares_cols: Sequence[str],
) -> int:
    set_clauses: list[str] = []
    params: list[float | str] = []

    for col in eps_cols:
        set_clauses.append(f'"{col}" = CASE WHEN "{col}" IS NOT NULL THEN CAST("{col}" AS REAL) / ? ELSE "{col}" END')
        params.append(split_ratio)
    for col in shares_cols:
        set_clauses.append(
            f'"{col}" = CASE WHEN "{col}" IS NOT NULL THEN CAST("{col}" AS REAL) * ? ELSE "{col}" END'
        )
        params.append(split_ratio)

    columns = _columns(cur, table)
    if "Last_Updated" in columns:
        set_clauses.append('"Last_Updated" = CURRENT_TIMESTAMP')

    metrics_predicates = [f'"{col}" IS NOT NULL' for col in [*eps_cols, *shares_cols]]
    where_predicate = f'WHERE "{ticker_col}"=? AND "{date_col}" < ?'
    if metrics_predicates:
        where_predicate += f" AND ({' OR '.join(metrics_predicates)})"

    sql = f'UPDATE "{table}" SET {", ".join(set_clauses)} {where_predicate};'
    params.extend([ticker, split_date.isoformat()])
    cur.execute(sql, tuple(params))
    return cur.rowcount or 0


def _backup_database(db_path: str, ticker: str, split_date: date) -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = f"{db_path}.pre_split_{ticker}_{split_date.isoformat()}_{ts}.bak"
    shutil.copy2(db_path, backup_path)
    logging.info("Backup created: %s", backup_path)
    return backup_path


def _record_split_event(cur: sqlite3.Cursor, ticker: str, split_date: date, ratio: float) -> None:
    ensure_splits_table(cur)
    now_iso = datetime.utcnow().isoformat(sep=" ", timespec="seconds")
    cur.execute(
        """
        INSERT INTO Splits(Symbol, Date, Ratio, Source, Last_Checked)
        VALUES(?,?,?,?,?)
        ON CONFLICT(Symbol, Date) DO UPDATE SET
            Ratio=excluded.Ratio,
            Source=excluded.Source,
            Last_Checked=excluded.Last_Checked;
        """,
        (ticker, split_date.isoformat(), ratio, "manual", now_iso),
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    ratio = _validate_ratio(args.ratio)
    split_date = _parse_date(args.effective_date)
    ticker = args.ticker.upper()
    _confirm(args, ratio, split_date)

    if not os.path.exists(args.db_path):
        raise SystemExit(f"Database file not found: {args.db_path}")

    conn = sqlite3.connect(args.db_path)
    try:
        cur = conn.cursor()
        tables = _get_tables(cur, args.tables)
        if not tables:
            raise SystemExit("No tables found to inspect.")

        if not args.dry_run and not args.no_backup:
            _backup_database(args.db_path, ticker, split_date)

        if args.dry_run:
            conn.execute("BEGIN")

        total_rows = 0
        for table in tables:
            cols = _columns(cur, table)
            ticker_col = _first_present(cols, TICKER_CANDIDATES)
            date_col = _first_present(cols, DATE_CANDIDATES)
            eps_cols = [c for c in cols if c in EPS_COLUMNS]
            shares_cols = [c for c in cols if c in SHARES_COLUMNS]

            if not ticker_col or not date_col:
                logging.debug("[%s] Skipping: missing ticker/date column.", table)
                continue
            if not eps_cols and not shares_cols:
                logging.debug("[%s] Skipping: no EPS or Shares_Outstanding columns.", table)
                continue

            target_columns = [*eps_cols, *shares_cols]
            eligible = _eligible_rowcount(cur, table, ticker_col, date_col, ticker, split_date, target_columns)
            logging.info("[%s] Eligible rows before %s: %d", table, split_date.isoformat(), eligible)
            if eligible == 0:
                continue

            _sample_rows(cur, table, ticker_col, date_col, ticker, split_date, target_columns, "Before")

            updated = _update_table(
                cur,
                table,
                ticker_col,
                date_col,
                ticker,
                split_date,
                ratio,
                eps_cols,
                shares_cols,
            )
            total_rows += updated
            logging.info("[%s] Rows updated: %d", table, updated)
            _sample_rows(cur, table, ticker_col, date_col, ticker, split_date, target_columns, "After")

        if not args.dry_run:
            _record_split_event(cur, ticker, split_date, ratio)
            conn.commit()
            logging.info("Split recorded in Splits table.")
            logging.info("Completed with %d total row(s) updated.", total_rows)
        else:
            conn.rollback()
            logging.info("Dry run complete. No changes were written.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()

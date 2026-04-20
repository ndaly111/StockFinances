#!/usr/bin/env python3
"""Backfill up to 10 years of annual Revenue, Net Income, and EPS from SEC EDGAR.

Uses the XBRL CompanyFacts API (public, no key required) to fill gaps in the
Annual_Data table.  Only inserts rows where data is missing — existing complete
rows from FMP or other licensed sources are never overwritten.

Can be run standalone or called from the main pipeline via
``maybe_backfill_edgar_financials()``.
"""

import logging
import sqlite3
import time
from pathlib import Path
from typing import List

from data_providers.edgar import EdgarDataProvider, DataProviderError

logger = logging.getLogger(__name__)

DB_PATH = "Stock Data.db"
# SEC asks for ≤10 requests/sec; we stay well under that.
_SEC_DELAY = 0.15  # seconds between tickers


def _existing_complete_dates(ticker: str, cursor: sqlite3.Cursor) -> set:
    """Return the set of Date strings that already have all three fields populated."""
    cursor.execute(
        """
        SELECT Date FROM Annual_Data
        WHERE Symbol = ?
          AND Revenue IS NOT NULL
          AND Net_Income IS NOT NULL
          AND EPS IS NOT NULL
        """,
        (ticker,),
    )
    return {row[0] for row in cursor.fetchall()}


def backfill_ticker(ticker: str, cursor: sqlite3.Cursor, provider: EdgarDataProvider) -> int:
    """Backfill a single ticker. Returns number of rows upserted."""
    try:
        records = provider.fetch_annual_financials(ticker)
    except DataProviderError as exc:
        logger.warning("[%s] EDGAR backfill skipped: %s", ticker, exc)
        return 0
    except Exception as exc:
        logger.warning("[%s] EDGAR backfill error: %s", ticker, exc)
        return 0

    if not records:
        return 0

    existing = _existing_complete_dates(ticker, cursor)
    upserted = 0

    for rec in records:
        date = rec["Date"]

        # Never overwrite a row that already has all three fields from a licensed source
        if date in existing:
            continue

        # Check if there's an existing partial row — only fill in NULLs
        cursor.execute(
            "SELECT Revenue, Net_Income, EPS FROM Annual_Data WHERE Symbol = ? AND Date = ?",
            (ticker, date),
        )
        existing_row = cursor.fetchone()

        if existing_row:
            # Partial row exists — only update NULL columns
            updates = []
            params = []
            if existing_row[0] is None and rec.get("Revenue") is not None:
                updates.append("Revenue = ?")
                params.append(rec["Revenue"])
            if existing_row[1] is None and rec.get("Net_Income") is not None:
                updates.append("Net_Income = ?")
                params.append(rec["Net_Income"])
            if existing_row[2] is None and rec.get("EPS") is not None:
                updates.append("EPS = ?")
                params.append(rec["EPS"])

            if updates:
                params.extend([ticker, date])
                cursor.execute(
                    f"UPDATE Annual_Data SET {', '.join(updates)}, Last_Updated = CURRENT_TIMESTAMP "
                    f"WHERE Symbol = ? AND Date = ?",
                    params,
                )
                upserted += 1
        else:
            # No row exists — insert new
            cursor.execute(
                """
                INSERT INTO Annual_Data (Symbol, Date, Revenue, Net_Income, EPS, Last_Updated)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (ticker, date, rec.get("Revenue"), rec.get("Net_Income"), rec.get("EPS")),
            )
            upserted += 1

    return upserted


def backfill_all(
    tickers: List[str],
    db_path: str = DB_PATH,
    years: int = 10,
) -> dict:
    """Backfill EDGAR financials for all tickers.

    Returns a summary dict with counts.
    """
    provider = EdgarDataProvider(years=years)
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        total_upserted = 0
        success = 0
        skipped = 0

        for i, ticker in enumerate(tickers):
            try:
                count = backfill_ticker(ticker, cursor, provider)
                if count > 0:
                    conn.commit()
                    total_upserted += count
                    logger.info("[%s] EDGAR backfilled %d rows", ticker, count)
                    success += 1
                else:
                    skipped += 1
            except Exception as exc:
                conn.rollback()
                logger.warning("[%s] EDGAR backfill failed: %s", ticker, exc)
                skipped += 1

            # Rate-limit SEC requests
            if i < len(tickers) - 1:
                time.sleep(_SEC_DELAY)

    summary = {
        "tickers_processed": len(tickers),
        "tickers_with_new_data": success,
        "tickers_skipped": skipped,
        "total_rows_upserted": total_upserted,
    }
    logger.info("[EDGAR backfill] Complete: %s", summary)
    return summary


def maybe_backfill_edgar_financials(
    tickers: List[str],
    db_path: str = DB_PATH,
    years: int = 10,
) -> None:
    """Run the EDGAR backfill only when historical gaps exist.

    Checks whether tickers are missing annual data older than 4 years
    (i.e., beyond what FMP typically provides).  If gaps are found, runs
    the backfill for ALL tickers to ensure consistency.
    """
    from datetime import datetime

    cutoff_year = datetime.now().year - 4  # FMP covers ~4 years

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # Check if we have any data older than the FMP window
            cursor.execute(
                """
                SELECT COUNT(DISTINCT Symbol) FROM Annual_Data
                WHERE Date < ? AND Revenue IS NOT NULL
                """,
                (f"{cutoff_year}-01-01",),
            )
            tickers_with_history = cursor.fetchone()[0]

        # If most tickers already have deep history, skip
        if tickers_with_history >= len(tickers) * 0.8:
            print(f"[EDGAR backfill] {tickers_with_history}/{len(tickers)} tickers "
                  f"already have pre-{cutoff_year} data; skipping backfill.")
            return

    except sqlite3.Error as exc:
        print(f"[EDGAR backfill] Unable to check history: {exc}")

    print(f"[EDGAR backfill] Backfilling up to {years} years of financials for {len(tickers)} tickers...")
    summary = backfill_all(tickers, db_path, years=years)
    print(f"[EDGAR backfill] Done: {summary['total_rows_upserted']} rows added "
          f"across {summary['tickers_with_new_data']} tickers")


# ── Standalone usage ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import csv
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    tickers_file = sys.argv[1] if len(sys.argv) > 1 else "tickers.csv"
    with open(tickers_file) as f:
        reader = csv.reader(f)
        tickers = sorted({row[0].strip().upper() for row in reader if row and row[0].strip()})

    print(f"Backfilling EDGAR financials for {len(tickers)} tickers...")
    summary = backfill_all(tickers)
    print(f"Done: {summary}")

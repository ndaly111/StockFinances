#!/usr/bin/env python3
"""
backfill_index_growth.py — use existing FRED DGS10 in economic_data to fill yields
and recompute historical implied growth for SPY/QQQ.

This script performs two major tasks:

1. Normalize and backfill the Treasury 10 ‑year yield series.  The
   `economic_data` table stores daily DGS10 values in percent units.
   These values are divided by 100 to convert them into decimal form
   (e.g., 4.25 → 0.0425) and upserted into the `Treasury_Yield_History`
   table.  Existing rows are updated on conflict, ensuring that
   `Treasury_Yield_History` always contains clean, decimal values.

2. Recompute implied growth for every entry in `Index_PE_History`.
   Implied growth is calculated as
   `((PE_Ratio / 10) ** 0.1) + yield - 1`, where `yield` is the most
   recent 10 ‑year Treasury yield on or prior to the P/E date.  If no
   historical yield is available for a particular date, a fallback
   yield is used.  Results are written back into `Index_Growth_History`,
   keyed by date, ticker and growth type (TTM or Forward).

Run this script as part of your build pipeline after refreshing P/E
history but before generating charts or HTML pages.  You can also run
it as a one ‑off from the command line:

    python backfill_index_growth.py

It is safe to run multiple times—the upsert and recompute queries
ensure idempotence.
"""

import sqlite3
from typing import Optional


# Database configuration
DB_PATH = "Stock Data.db"
# Fallback yield to use when no historical yield is available (decimal)
FALLBACK_YIELD = 0.045


# SQL statement to upsert normalized yields into Treasury_Yield_History
UPSERT_YIELDS_SQL = """
INSERT INTO Treasury_Yield_History(Date, TenYr)
SELECT date, value / 100.0
  FROM economic_data
 WHERE indicator = 'DGS10' AND value IS NOT NULL
ON CONFLICT(Date) DO UPDATE SET TenYr = excluded.TenYr;
"""


# SQL statement to recompute implied growth
RECOMPUTE_IG_SQL = """
INSERT OR REPLACE INTO Index_Growth_History(Date, Ticker, Growth_Type, Implied_Growth)
SELECT
    p.Date,
    p.Ticker,
    p.PE_Type AS Growth_Type,
    POWER(p.PE_Ratio / 10.0, 0.1) +
    COALESCE(
        -- use the latest yield on or before p.Date
        (SELECT TenYr
           FROM Treasury_Yield_History t
          WHERE t.Date <= p.Date
          ORDER BY t.Date DESC
          LIMIT 1),
        ?
    ) - 1.0 AS Implied_Growth
FROM Index_PE_History p
WHERE p.PE_Ratio IS NOT NULL AND p.PE_Ratio > 0;
"""



def backfill_index_growth(
    db_path: str = DB_PATH,
    fallback_yield: float = FALLBACK_YIELD,
) -> None:
    """Backfill the 10 ‑year yield and recompute implied growth.

    This function connects to the SQLite database specified by
    `db_path`, inserts normalized 10 ‑year Treasury yields into
    `Treasury_Yield_History` from the `economic_data` table, and then
    recomputes implied growth for all rows in `Index_PE_History`.

    Args:
        db_path: Path to the SQLite database file.
        fallback_yield: Yield value (decimal) to use when no yield
            exists for a given P/E date.
    """
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        # Create the Treasury_Yield_History table if it doesn't exist
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS Treasury_Yield_History (
                Date  TEXT PRIMARY KEY,
                TenYr REAL
            );
            """
        )
        # Upsert the yield values from economic_data
        cur.execute(UPSERT_YIELDS_SQL)
        # Recompute implied growth using nearest yields
        cur.execute(RECOMPUTE_IG_SQL, (float(fallback_yield),))
        conn.commit()
        print(
            f"[backfill_index_growth] Completed yield upsert and implied growth recompute using fallback {fallback_yield:.4f}."
        )
    finally:
        conn.close()


# If executed as a script, perform the backfill using default settings
if __name__ == "__main__":
    backfill_index_growth()

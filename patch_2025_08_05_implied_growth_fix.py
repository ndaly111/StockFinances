#!/usr/bin/env python3
"""
One-off patch:
  • Recompute TTM implied-growth for SPY & QQQ on 2025-08-05
  • Then print the full Index_Growth_History table for those two tickers

Formula
-------
    g = (PE / 10) ** 0.1 + y − 1

where
  • y  = 10-year Treasury yield on DATE_FIX, converted to decimal
  • PE = latest TTM P/E available on or before DATE_FIX
"""

import sqlite3
import sys
from pathlib import Path

# ── configuration ────────────────────────────────────────────────────────────
DB_PATH  = Path("Stock Data.db")            # adjust if your DB lives elsewhere
DATE_FIX = "2025-08-05"
TICKERS  = ("SPY", "QQQ")                   # extend if needed
# ----------------------------------------------------------------------------

def to_decimal(v: float) -> float:
    """Convert 4.2 or 42 ➜ 0.042  ; leave 0.042 as-is."""
    if v < 0.5:  return v
    if v < 20:   return v / 100
    return v / 1000

def latest_pe(cur: sqlite3.Cursor, tk: str, cutoff: str) -> float | None:
    row = cur.execute(
        """
        SELECT PE_Ratio
        FROM   Index_PE_History
        WHERE  Ticker = ?
          AND  PE_Type = 'TTM'
          AND  Date <= ?
        ORDER BY Date DESC
        LIMIT 1
        """,
        (tk, cutoff)
    ).fetchone()
    return row[0] if row else None

def main() -> None:
    if not DB_PATH.exists():
        sys.exit(f"ERROR: database not found → {DB_PATH}")

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        # ── fetch 10-yr yield ────────────────────────────────────────────────
        row = cur.execute(
            "SELECT TenYr FROM Treasury_Yield_History WHERE Date = ?",
            (DATE_FIX,)
        ).fetchone()
        if not row:
            sys.exit(
                f"ERROR: No 10-yr yield stored for {DATE_FIX}. "
                "Run your normal job first, then re-run this patch."
            )
        y = to_decimal(float(row[0]))
        print(f"Using 10-yr yield y = {y:.6f}")

        # ── recompute & overwrite implied growth ────────────────────────────
        for tk in TICKERS:
            pe = latest_pe(cur, tk, DATE_FIX)
            if pe is None:
                print(f"[{tk}] skipped – no TTM P/E ≤ {DATE_FIX}")
                continue

            g = (pe / 10) ** 0.1 + y - 1
            cur.execute(
                """
                INSERT OR REPLACE INTO Index_Growth_History
                VALUES (?, ?, 'TTM', ?)
                """,
                (DATE_FIX, tk, g)
            )
            print(f"[{tk}] PE={pe:.2f} → new g={g:.6%}")

        conn.commit()
        print(f"✓  Patched Implied_Growth (TTM) rows for {DATE_FIX}\n")

        # ── print the entire table for SPY & QQQ ────────────────────────────
        rows = cur.execute(
            """
            SELECT Date, Ticker, Growth_Type, Implied_Growth
            FROM   Index_Growth_History
            WHERE  Ticker IN (?, ?)
            ORDER  BY Date, Ticker, Growth_Type
            """,
            TICKERS
        ).fetchall()

    if not rows:
        print("No rows found in Index_Growth_History for SPY/QQQ.")
        return

    # pretty header
    header = f"{'Date':<12} {'Ticker':<5} {'Type':<8} {'Implied_Growth':>14}"
    print(header)
    print("-" * len(header))

    # print every record
    for r in rows:
        print(
            f"{r['Date']:<12} {r['Ticker']:<5} {r['Growth_Type']:<8} "
            f"{r['Implied_Growth']:>13.6%}"
        )

if __name__ == "__main__":
    main()

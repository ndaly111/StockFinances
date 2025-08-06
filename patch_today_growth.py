#!/usr/bin/env python3
"""
Re-compute TODAY’S implied-growth (TTM) values for the indexes in IDXES,
using the yield that is already stored for today in Treasury_Yield_History.

    g = ((PE / 10) ** 0.1) + y  – 1

where y is taken *as recorded* today, but auto-converted to decimal
(0.042) if it was stored as 4.2 or 42.
"""
import sqlite3, math, pandas as pd
from datetime import datetime

DB_PATH = "Stock Data.db"
IDXES   = ["SPY", "QQQ"]          # extend if needed
TODAY   = datetime.today().strftime("%Y-%m-%d")

def _to_decimal(v: float) -> float:
    """Return decimal yield (0.042)."""
    if v < 0.5:  return v          # already decimal
    if v < 20:   return v / 100    # stored as percent
    return v / 1000                # stored as ^TNX quote

def _latest_pe(conn, tk):
    row = conn.execute(
        "SELECT PE_Ratio FROM Index_PE_History "
        "WHERE Ticker=? AND PE_Type='TTM' "
        "ORDER BY Date DESC LIMIT 1", (tk,)
    ).fetchone()
    return row[0] if row else None

with sqlite3.connect(DB_PATH) as conn:
    cur = conn.cursor()

    # 1. Fetch today’s yield
    y_row = cur.execute(
        "SELECT TenYr FROM Treasury_Yield_History WHERE Date=?",
        (TODAY,)
    ).fetchone()
    if not y_row:
        raise SystemExit(f"No TenYr yield recorded for {TODAY}. Run the normal job first.")
    y = _to_decimal(float(y_row[0]))
    print(f"Using yield y = {y:.4f}")

    # 2. Re-compute g for each ticker
    for tk in IDXES:
        pe = _latest_pe(conn, tk)
        if pe is None:
            print(f"[{tk}] skipped – no P/E in DB yet.")
            continue

        g = (pe / 10) ** 0.1 + y - 1
        print(f"[{tk}]  PE={pe:.2f}  →  g={g:.4%}")

        cur.execute("""
            INSERT OR REPLACE INTO Index_Growth_History
            VALUES (?,?, 'TTM', ?)
        """, (TODAY, tk, g))

    conn.commit()
    print("✓ Today’s implied-growth rows updated.")

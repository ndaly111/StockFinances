#!/usr/bin/env python3
"""
One-off patch: recompute **TTM implied-growth** for SPY & QQQ
on 2025-08-05 with the correct formula

    g = ((PE / 10) ** 0.1) + y - 1

• y  = Ten-year yield stored for 2025-08-05  
       (4.22  ➜  0.0422   or   42.2  ➜  0.0422)
• PE = latest TTM P/E ≤ 2025-08-05
-----------------------------------------------------------"""

import sqlite3, sys

DB        = "Stock Data.db"
DATE_FIX  = "2025-08-05"
IDXES     = ["SPY", "QQQ"]          # add more if ever needed

def to_decimal(v: float) -> float:
    """Convert 4.2 or 42 to 0.042, leave 0.042 as-is."""
    if v < 0.5:  return v
    if v < 20:   return v / 100
    return v / 1000

def latest_pe(cur, tk: str):
    row = cur.execute(
        """SELECT PE_Ratio FROM Index_PE_History
           WHERE Ticker=? AND PE_Type='TTM' AND Date<=?
           ORDER BY Date DESC LIMIT 1""",
        (tk, DATE_FIX)
    ).fetchone()
    return row[0] if row else None

with sqlite3.connect(DB) as conn:
    cur = conn.cursor()

    # --- fetch the correct yield ------------------------------------------------
    row = cur.execute(
        "SELECT TenYr FROM Treasury_Yield_History WHERE Date=?",
        (DATE_FIX,)
    ).fetchone()
    if not row:
        sys.exit(f"ERROR: No 10-yr yield stored for {DATE_FIX}.  "
                 "Run your normal job first, then re-run this patch.")
    y = to_decimal(float(row[0]))
    print(f"Using 10-yr yield y = {y:.6f}")

    # --- recompute & overwrite Implied_Growth (TTM) -----------------------------
    for tk in IDXES:
        pe = latest_pe(cur, tk)
        if pe is None:
            print(f"[{tk}] skipped – no TTM P/E ≤ {DATE_FIX}")
            continue
        g = (pe / 10)**0.1 + y - 1
        cur.execute(
            """INSERT OR REPLACE INTO Index_Growth_History
               VALUES (?,?, 'TTM', ?)""",
            (DATE_FIX, tk, g)
        )
        print(f"[{tk}] PE={pe:.2f} → new g={g:.6%}")

    conn.commit()
    print("✓  Patched Implied_Growth (TTM) rows for", DATE_FIX)

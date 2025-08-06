#!/usr/bin/env python3
"""
Overwrite the Implied_Growth (TTM) row dated 2025-08-05
for each ticker in IDXES with:

    g = ((PE / 10)**0.1) + y - 1

• y is pulled FROM THE DB for 2025-08-05 and auto-converted
  to decimal if it’s stored as 4.2 or 42.
• PE is the latest recorded TTM P/E (any date ≤ patch date).

Run once, commit the DB, delete the script.
"""

import sqlite3, sys
from datetime import datetime

DB      = "Stock Data.db"
DATE_FIX = "2025-08-05"
IDXES   = ["SPY", "QQQ"]          # add if needed

def to_decimal(v: float) -> float:     # 4.2 → 0.042
    if v < 0.5:  return v
    if v < 20:   return v / 100
    return v / 1000

with sqlite3.connect(DB) as conn:
    cur = conn.cursor()

    # --- 1. fetch yield for 2025-08-05 --------------------
    row = cur.execute(
        "SELECT TenYr FROM Treasury_Yield_History WHERE Date=?",
        (DATE_FIX,)
    ).fetchone()
    if not row:
        sys.exit(f"✗  No yield stored for {DATE_FIX}. Run your main job first.")
    y = to_decimal(float(row[0]))
    print(f"Using yield y = {y:.4f} for {DATE_FIX}")

    # helper: latest PE on/before fix date
    def latest_pe(tk):
        r = cur.execute(
            "SELECT PE_Ratio FROM Index_PE_History "
            "WHERE Ticker=? AND PE_Type='TTM' AND Date<=? "
            "ORDER BY Date DESC LIMIT 1", (tk, DATE_FIX)
        ).fetchone()
        return r[0] if r else None

    # --- 2. recompute & overwrite row ---------------------
    for tk in IDXES:
        pe = latest_pe(tk)
        if pe is None:
            print(f"[{tk}] skipped – no PE available.")
            continue
        g = (pe / 10) ** 0.1 + y - 1
        print(f"[{tk}] PE={pe:.2f}  →  new g={g:.4%}")

        cur.execute("""
            INSERT OR REPLACE INTO Index_Growth_History
            VALUES (?,?, 'TTM', ?)""", (DATE_FIX, tk, g))

    conn.commit()
    print("✓  Patched Implied_Growth rows for", DATE_FIX)

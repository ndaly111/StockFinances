#!/usr/bin/env python3
"""
Re-compute Implied-Growth (TTM) for SPY & QQQ only on 2025-08-05.

Formula:
    g = ((PE / 10)**0.1) + y – 1

• PE  = latest stored TTM P/E *as of when this script runs*
• y   = yield already stored for 2025-08-05 in Treasury_Yield_History
        (auto-converted to decimal if recorded as 4.2 or 42).
"""

import sqlite3
from datetime import datetime

DB_PATH  = "Stock Data.db"
DATE_FIX = "2025-08-05"
TICKERS  = ["SPY", "QQQ"]

def to_decimal(v: float) -> float:
    """4.2 → 0.042; 42 → 0.042; 0.042 stays 0.042."""
    if v < 0.5:  return v
    if v < 20:   return v / 100
    return v / 1000

def latest_pe(conn, tk):
    row = conn.execute(
        "SELECT PE_Ratio FROM Index_PE_History "
        "WHERE Ticker=? AND PE_Type='TTM' "
        "ORDER BY Date DESC LIMIT 1",
        (tk,)
    ).fetchone()
    return row[0] if row else None

with sqlite3.connect(DB_PATH) as conn:
    cur = conn.cursor()

    # ---- fetch yield for 5 Aug 2025 -----------------------
    y_row = cur.execute(
        "SELECT TenYr FROM Treasury_Yield_History WHERE Date=?",
        (DATE_FIX,)
    ).fetchone()
    if not y_row:
        raise SystemExit(f"No yield stored for {DATE_FIX}.")
    y = to_decimal(float(y_row[0]))
    print(f"Yield on {DATE_FIX} ≈ {y:.4f}")

    # ---- recalc & overwrite growth ------------------------
    for tk in TICKERS:
        pe = latest_pe(conn, tk)
        if pe is None:
            print(f"[{tk}] skipped – no P/E available.")
            continue

        g = (pe / 10) ** 0.1 + y - 1
        print(f"[{tk}] PE={pe:.2f}  →  g={g:.4%}")

        cur.execute("""
            INSERT OR REPLACE INTO Index_Growth_History
            VALUES (?,?, 'TTM', ?)
        """, (DATE_FIX, tk, g))

    conn.commit()
    print("✓ Implied-Growth rows for 2025-08-05 updated.")

#!/usr/bin/env python3
"""One-off backfill for SPY Dec 11-23 2025 PE/EPS gap.

The daily PE/EPS collector did not run on these 7 business days. Price
history was captured by another job. This script derives ETF-level
trailing EPS via linear interpolation across the gap, then computes
PE = Close / EPS using existing Index_Price_History rows. Implied growth
is recomputed afterward by re-running backfill_index_growth.py.
"""

import sqlite3
from datetime import date

DB = "Stock Data.db"

GAP_DATES = [
    "2025-12-11", "2025-12-12",
    "2025-12-15", "2025-12-16", "2025-12-17",
    "2025-12-22", "2025-12-23",
]


def _interp(d_target: date, d_lo: date, eps_lo: float, d_hi: date, eps_hi: float) -> float:
    span = (d_hi - d_lo).days
    frac = (d_target - d_lo).days / span
    return eps_lo + (eps_hi - eps_lo) * frac


def main(db_path: str = DB) -> int:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute(
        "SELECT Date, EPS FROM Index_EPS_History "
        "WHERE Ticker='SPY' AND EPS_Type='TTM' "
        "AND Date IN ('2025-12-10','2025-12-18','2025-12-19','2025-12-24')"
    )
    anchors = {r[0]: float(r[1]) for r in cur.fetchall()}
    expected = {"2025-12-10", "2025-12-18", "2025-12-19", "2025-12-24"}
    missing = expected - anchors.keys()
    if missing:
        print(f"Missing anchor rows: {sorted(missing)}")
        return 1

    cur.execute(
        "SELECT Date, Close FROM Index_Price_History "
        "WHERE Ticker='SPY' AND Date IN (%s)"
        % ",".join("?" * len(GAP_DATES)),
        GAP_DATES,
    )
    prices = {r[0]: float(r[1]) for r in cur.fetchall()}
    missing_px = set(GAP_DATES) - prices.keys()
    if missing_px:
        print(f"Missing Index_Price_History rows for: {sorted(missing_px)}")
        return 1

    rows_pe, rows_eps = [], []
    for ds in GAP_DATES:
        d = date.fromisoformat(ds)
        if d <= date(2025, 12, 18):
            lo, hi = "2025-12-10", "2025-12-18"
        else:
            lo, hi = "2025-12-19", "2025-12-24"
        eps = _interp(d, date.fromisoformat(lo), anchors[lo], date.fromisoformat(hi), anchors[hi])
        pe = prices[ds] / eps
        rows_pe.append((ds, "SPY", "TTM", pe))
        rows_eps.append((ds, "SPY", "TTM", eps))
        print(f"  {ds}  close={prices[ds]:.2f}  EPS={eps:.4f}  PE={pe:.4f}")

    cur.executemany(
        "INSERT OR REPLACE INTO Index_PE_History(Date,Ticker,PE_Type,PE_Ratio) VALUES (?,?,?,?)",
        rows_pe,
    )
    cur.executemany(
        "INSERT OR REPLACE INTO Index_EPS_History(Date,Ticker,EPS_Type,EPS) VALUES (?,?,?,?)",
        rows_eps,
    )
    conn.commit()
    conn.close()
    print(f"\nInserted {len(rows_pe)} PE rows and {len(rows_eps)} EPS rows.")
    print("Now run: python backfill_index_growth.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

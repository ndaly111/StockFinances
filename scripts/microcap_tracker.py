"""Track when each microcap candidate first appeared and what its price
has done since.

The screener already writes data/microcap_candidates.csv on each run. This
module post-processes that file and maintains a persistent ledger in
Stock Data.db (table Microcap_Appearances) so we can see, over time:

  - When was this ticker first surfaced by the screener?
  - What was its price then?
  - What's its price today?
  - What's the peak since first appearance?
  - Is it still on the current list, or did it drop off?

Run after the screener:
    python scripts/microcap_tracker.py

Also pulls a current price for tickers that are NOT on the current list
but were tracked previously — so the "current return" stays accurate even
for past picks that have dropped off (e.g. graduated above the $1B cap,
or D/E deteriorated). yfinance .info per ticker, batched in a small thread
pool to stay polite.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("microcap_tracker")

REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = REPO_ROOT / "Stock Data.db"
DEFAULT_IN = REPO_ROOT / "data" / "microcap_candidates.csv"

SCHEMA = """
CREATE TABLE IF NOT EXISTS Microcap_Appearances (
    ticker            TEXT PRIMARY KEY,
    first_seen_date   TEXT NOT NULL,
    first_seen_price  REAL NOT NULL,
    last_seen_date    TEXT NOT NULL,
    last_seen_price   REAL NOT NULL,
    max_price         REAL NOT NULL,
    max_price_date    TEXT NOT NULL,
    on_current_list   INTEGER NOT NULL DEFAULT 1
);
"""


def _safe_float(v) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        f = float(v)
        if f != f:  # NaN
            return None
        return f
    except (TypeError, ValueError):
        return None


def _fetch_price(ticker: str) -> Optional[float]:
    """Try to fetch the current price for a ticker. Used for previously-
    tracked candidates that aren't in today's candidates CSV (so they don't
    have a fresh price)."""
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        return None
    return (
        _safe_float(info.get("currentPrice"))
        or _safe_float(info.get("regularMarketPrice"))
    )


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute(SCHEMA)
    conn.commit()


def update_from_csv(csv_path: Path, db_path: Path = DB_PATH) -> dict:
    """Upsert each candidate from the screener CSV. Returns counts.

    Logic:
      - New ticker → INSERT with today as first/last/max.
      - Existing ticker on current list → UPDATE last_seen, max_price if
        higher, on_current_list = 1.
    """
    if not csv_path.exists():
        log.error(f"Candidates CSV not found: {csv_path}")
        return {"new": 0, "updated": 0, "skipped": 0}

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    new_count = 0
    updated_count = 0
    skipped_count = 0
    current_tickers: set[str] = set()

    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        _ensure_schema(conn)
        cur = conn.cursor()
        with open(csv_path, newline="", encoding="utf-8") as f:
            text = "\n".join(ln for ln in f.read().splitlines() if not ln.startswith("#"))
        if not text.strip():
            log.info("Candidates CSV is empty; nothing to do.")
            return {"new": 0, "updated": 0, "skipped": 0}

        for row in csv.DictReader(text.splitlines()):
            ticker = (row.get("ticker") or "").strip().upper()
            price = _safe_float(row.get("current_price"))
            if not ticker or price is None or price <= 0:
                skipped_count += 1
                continue
            current_tickers.add(ticker)

            cur.execute(
                "SELECT first_seen_date, first_seen_price, max_price FROM Microcap_Appearances WHERE ticker = ?",
                (ticker,),
            )
            existing = cur.fetchone()
            if existing is None:
                cur.execute(
                    """
                    INSERT INTO Microcap_Appearances
                      (ticker, first_seen_date, first_seen_price,
                       last_seen_date, last_seen_price,
                       max_price, max_price_date, on_current_list)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 1)
                    """,
                    (ticker, today, price, today, price, price, today),
                )
                new_count += 1
            else:
                _, _, max_p = existing
                if price > max_p:
                    cur.execute(
                        """
                        UPDATE Microcap_Appearances
                           SET last_seen_date  = ?,
                               last_seen_price = ?,
                               max_price       = ?,
                               max_price_date  = ?,
                               on_current_list = 1
                         WHERE ticker = ?
                        """,
                        (today, price, price, today, ticker),
                    )
                else:
                    cur.execute(
                        """
                        UPDATE Microcap_Appearances
                           SET last_seen_date  = ?,
                               last_seen_price = ?,
                               on_current_list = 1
                         WHERE ticker = ?
                        """,
                        (today, price, ticker),
                    )
                updated_count += 1

        # Mark previously-tracked tickers that are NOT in today's CSV.
        cur.execute("SELECT ticker FROM Microcap_Appearances WHERE on_current_list = 1")
        prev_tickers = {r[0] for r in cur.fetchall()}
        dropped = prev_tickers - current_tickers
        if dropped:
            cur.executemany(
                "UPDATE Microcap_Appearances SET on_current_list = 0 WHERE ticker = ?",
                [(t,) for t in dropped],
            )
            log.info(f"  marked {len(dropped)} tickers as dropped off the current list")

        # Refresh current price for dropped-off tickers so peak/current return
        # stays meaningful. Skip if they have no rows here.
        all_dropped = [t for (t,) in cur.execute(
            "SELECT ticker FROM Microcap_Appearances WHERE on_current_list = 0"
        ).fetchall()]
        if all_dropped:
            log.info(f"  refreshing current price for {len(all_dropped)} historically-tracked tickers")
            refreshed = 0
            t0 = time.time()
            results: dict[str, float] = {}
            # Polite worker count — these are not on the current list so
            # there's no rush to be fast.
            with ThreadPoolExecutor(max_workers=4) as ex:
                futs = {ex.submit(_fetch_price, t): t for t in all_dropped}
                for fut in as_completed(futs):
                    t = futs[fut]
                    p = fut.result()
                    if p is not None and p > 0:
                        results[t] = p
            for t, p in results.items():
                cur.execute(
                    "SELECT max_price FROM Microcap_Appearances WHERE ticker = ?",
                    (t,),
                )
                mp_row = cur.fetchone()
                if mp_row is None:
                    continue
                max_p = mp_row[0]
                if p > max_p:
                    cur.execute(
                        """
                        UPDATE Microcap_Appearances
                           SET last_seen_date  = ?,
                               last_seen_price = ?,
                               max_price       = ?,
                               max_price_date  = ?
                         WHERE ticker = ?
                        """,
                        (today, p, p, today, t),
                    )
                else:
                    cur.execute(
                        """
                        UPDATE Microcap_Appearances
                           SET last_seen_date  = ?,
                               last_seen_price = ?
                         WHERE ticker = ?
                        """,
                        (today, p, t),
                    )
                refreshed += 1
            log.info(f"  refreshed {refreshed} prices in {time.time()-t0:.0f}s")

        conn.commit()
    finally:
        conn.close()

    log.info(
        f"Microcap_Appearances: {new_count} new, {updated_count} updated, "
        f"{skipped_count} skipped"
    )
    return {"new": new_count, "updated": updated_count, "skipped": skipped_count}


def load_all_appearances(db_path: Path = DB_PATH) -> list[dict]:
    """Return every row in Microcap_Appearances as dicts with derived fields."""
    if not db_path.exists():
        return []
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        _ensure_schema(conn)
        cur = conn.cursor()
        cur.execute("""
            SELECT ticker, first_seen_date, first_seen_price,
                   last_seen_date, last_seen_price,
                   max_price, max_price_date, on_current_list
              FROM Microcap_Appearances
        """)
        rows = []
        today = datetime.now(timezone.utc).date()
        for r in cur.fetchall():
            (ticker, first_date, first_price, last_date, last_price,
             max_p, max_date, on_list) = r
            try:
                d0 = datetime.strptime(first_date, "%Y-%m-%d").date()
                days_tracked = (today - d0).days
            except ValueError:
                days_tracked = 0
            current_return = (
                (last_price / first_price - 1.0) if first_price > 0 else None
            )
            peak_return = (
                (max_p / first_price - 1.0) if first_price > 0 else None
            )
            rows.append({
                "ticker": ticker,
                "first_seen_date": first_date,
                "first_seen_price": first_price,
                "last_seen_date": last_date,
                "last_seen_price": last_price,
                "max_price": max_p,
                "max_price_date": max_date,
                "on_current_list": bool(on_list),
                "days_tracked": days_tracked,
                "current_return": current_return,
                "peak_return": peak_return,
            })
        return rows
    finally:
        conn.close()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--csv", type=Path, default=DEFAULT_IN)
    p.add_argument("--db", type=Path, default=DB_PATH)
    args = p.parse_args()
    update_from_csv(args.csv, args.db)
    return 0


if __name__ == "__main__":
    sys.exit(main())

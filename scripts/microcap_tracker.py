"""Track when each microcap candidate first appeared and what its price
has done since.

State lives in a JSON file at data/microcap_appearances.json. We
originally put this in Stock Data.db, but that DB is touched by other
workflows (daily_index_data, etc.) on overlapping schedules — SQLite is
binary, so any concurrent push causes an unmergeable git conflict.
JSON is a text file that can rebase + auto-merge fine (and even if not,
the ledger is just a flat dict per ticker).

Schema (one entry per ticker we've ever surfaced):
  {
    "AAPL": {
      "first_seen_date":  "2026-05-22",
      "first_seen_price": 195.27,
      "last_seen_date":   "2026-06-15",
      "last_seen_price":  201.40,
      "max_price":        205.10,
      "max_price_date":   "2026-06-10",
      "on_current_list":  true
    },
    ...
  }

Plus a top-level "updated_at" ISO timestamp.

Logic per run:
  - For each ticker in current candidates CSV: upsert (capture first-seen
    on first appearance, refresh last_seen / max_price always).
  - For tickers in the ledger NOT on the current list: mark on_current_list
    = false AND fetch a current price via yfinance so peak/current return
    stays accurate for past picks that have dropped off.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
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
DEFAULT_IN = REPO_ROOT / "data" / "microcap_candidates.csv"
DEFAULT_LEDGER = REPO_ROOT / "data" / "microcap_appearances.json"


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
    tracked candidates that aren't in today's candidates CSV."""
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        return None
    return (
        _safe_float(info.get("currentPrice"))
        or _safe_float(info.get("regularMarketPrice"))
    )


def _load_ledger(path: Path) -> dict:
    if not path.exists():
        return {"updated_at": None, "tickers": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        log.warning(f"Ledger {path} unreadable; starting fresh")
        return {"updated_at": None, "tickers": {}}


def _save_ledger(path: Path, ledger: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ledger["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    path.write_text(json.dumps(ledger, indent=2, sort_keys=True), encoding="utf-8")


def update_from_csv(csv_path: Path, ledger_path: Path = DEFAULT_LEDGER) -> dict:
    """Upsert each candidate from the screener CSV into the JSON ledger.
    Returns counts."""
    if not csv_path.exists():
        log.error(f"Candidates CSV not found: {csv_path}")
        return {"new": 0, "updated": 0, "skipped": 0}

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    ledger = _load_ledger(ledger_path)
    tickers_map: dict = ledger.setdefault("tickers", {})

    new_count = 0
    updated_count = 0
    skipped_count = 0
    current_tickers: set[str] = set()

    with open(csv_path, newline="", encoding="utf-8") as f:
        text = "\n".join(ln for ln in f.read().splitlines() if not ln.startswith("#"))
    if not text.strip():
        log.info("Candidates CSV is empty; nothing to upsert.")
    else:
        for row in csv.DictReader(text.splitlines()):
            ticker = (row.get("ticker") or "").strip().upper()
            price = _safe_float(row.get("current_price"))
            if not ticker or price is None or price <= 0:
                skipped_count += 1
                continue
            current_tickers.add(ticker)

            entry = tickers_map.get(ticker)
            if entry is None:
                tickers_map[ticker] = {
                    "first_seen_date":  today,
                    "first_seen_price": price,
                    "last_seen_date":   today,
                    "last_seen_price":  price,
                    "max_price":        price,
                    "max_price_date":   today,
                    "on_current_list":  True,
                }
                new_count += 1
            else:
                entry["last_seen_date"] = today
                entry["last_seen_price"] = price
                entry["on_current_list"] = True
                if price > entry.get("max_price", 0):
                    entry["max_price"] = price
                    entry["max_price_date"] = today
                updated_count += 1

    # Mark previously-tracked tickers that are NOT in today's CSV as dropped.
    dropped = [
        t for t, entry in tickers_map.items()
        if entry.get("on_current_list") and t not in current_tickers
    ]
    if dropped:
        log.info(f"  marking {len(dropped)} tickers as dropped off the current list")
        for t in dropped:
            tickers_map[t]["on_current_list"] = False

    # Refresh prices for all on_current_list=False tickers so peak/current
    # return stays meaningful for past picks. yfinance .info per ticker, 4
    # workers to stay polite.
    inactive = [t for t, entry in tickers_map.items() if not entry.get("on_current_list")]
    if inactive:
        log.info(f"  refreshing current price for {len(inactive)} historically-tracked tickers")
        t0 = time.time()
        results: dict[str, float] = {}
        with ThreadPoolExecutor(max_workers=4) as ex:
            futs = {ex.submit(_fetch_price, t): t for t in inactive}
            for fut in as_completed(futs):
                t = futs[fut]
                p = fut.result()
                if p is not None and p > 0:
                    results[t] = p
        refreshed = 0
        for t, p in results.items():
            entry = tickers_map[t]
            entry["last_seen_date"] = today
            entry["last_seen_price"] = p
            if p > entry.get("max_price", 0):
                entry["max_price"] = p
                entry["max_price_date"] = today
            refreshed += 1
        log.info(f"  refreshed {refreshed} prices in {time.time()-t0:.0f}s")

    _save_ledger(ledger_path, ledger)
    log.info(
        f"Microcap appearances ledger: {new_count} new, {updated_count} updated, "
        f"{skipped_count} CSV rows skipped — {len(tickers_map)} tickers tracked total"
    )
    return {"new": new_count, "updated": updated_count, "skipped": skipped_count}


def load_all_appearances(ledger_path: Path = DEFAULT_LEDGER) -> list[dict]:
    """Return every entry from the JSON ledger as dicts with derived fields."""
    ledger = _load_ledger(ledger_path)
    rows: list[dict] = []
    today = datetime.now(timezone.utc).date()
    for ticker, entry in ledger.get("tickers", {}).items():
        first_price = entry.get("first_seen_price") or 0.0
        last_price = entry.get("last_seen_price") or 0.0
        max_p = entry.get("max_price") or 0.0
        try:
            d0 = datetime.strptime(entry["first_seen_date"], "%Y-%m-%d").date()
            days_tracked = (today - d0).days
        except (KeyError, ValueError):
            days_tracked = 0
        current_return = (last_price / first_price - 1.0) if first_price > 0 else None
        peak_return = (max_p / first_price - 1.0) if first_price > 0 else None
        rows.append({
            "ticker":           ticker,
            "first_seen_date":  entry.get("first_seen_date"),
            "first_seen_price": first_price,
            "last_seen_date":   entry.get("last_seen_date"),
            "last_seen_price":  last_price,
            "max_price":        max_p,
            "max_price_date":   entry.get("max_price_date"),
            "on_current_list":  bool(entry.get("on_current_list")),
            "days_tracked":     days_tracked,
            "current_return":   current_return,
            "peak_return":      peak_return,
        })
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--csv", type=Path, default=DEFAULT_IN)
    p.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    args = p.parse_args()
    update_from_csv(args.csv, args.ledger)
    return 0


if __name__ == "__main__":
    sys.exit(main())

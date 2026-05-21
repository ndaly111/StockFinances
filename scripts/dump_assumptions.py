"""Dump current valuation assumptions to data/assumptions.json.

Reads Tickers_Info from Stock Data.db (the canonical store of per-ticker
nicks_growth_rate and projected_profit_margin) and writes a small JSON
file at data/assumptions.json:

    {
      "updated_at": "2026-05-21T01:30:00Z",
      "tickers": {
        "AAPL": {"growth_rate": 12.0, "profit_margin": 27.5},
        ...
      }
    }

The unlisted update_form.html fetches this file so the editor can show
the current value next to each input — so you can see what you're
overwriting before submitting.

Run after every assumption update + as part of the normal site refresh.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = REPO_ROOT / "Stock Data.db"
OUT_PATH = REPO_ROOT / "data" / "assumptions.json"


def _safe_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> int:
    if not DB_PATH.exists():
        print(f"ERROR: DB not found at {DB_PATH}", file=sys.stderr)
        return 1

    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row
    cur = con.execute(
        "SELECT ticker, nicks_growth_rate, projected_profit_margin "
        "FROM Tickers_Info ORDER BY ticker"
    )
    tickers = {}
    for row in cur.fetchall():
        t = row["ticker"]
        if not t:
            continue
        tickers[t] = {
            "growth_rate": _safe_float(row["nicks_growth_rate"]),
            "profit_margin": _safe_float(row["projected_profit_margin"]),
        }
    con.close()

    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tickers": tickers,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_PATH} ({len(tickers)} tickers)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

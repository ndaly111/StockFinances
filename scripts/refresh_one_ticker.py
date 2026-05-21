"""Targeted refresh: recompute valuation + tables + ticker page for ONE ticker.

Used after an assumption change so we don't pay the full 25-minute site refresh
for what is a one-cell update. Skips everything that doesn't depend on
nicks_growth_rate / projected_profit_margin:

  - raw financial fetches (annual / quarterly / EDGAR backfill)
  - forecast EPS/revenue chart regeneration
  - forward EPS/revenue history snapshots (those track upstream estimates,
    not our assumptions)
  - balance sheet, segments, expense reports
  - index growth pages, dashboard, market summary

Re-runs only:
  - valuation_update: recomputes fair-value PE/PS, writes the valuation chart
    PNG and the two valuation tables (HTML files in charts/).
  - prepare_and_generate_ticker_pages([ticker]): re-renders only this ticker's
    pages/{TICKER}_page.html so it picks up the new chart + tables.

Usage:
    python scripts/refresh_one_ticker.py AAPL
"""

from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# All the existing pipeline modules expect to run from the repo root (they
# hardcode 'Stock Data.db' and 'charts/' as relative paths). Make that the cwd
# regardless of where this script is invoked from.
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yfinance as yf  # noqa: E402

from ticker_info import prepare_data_for_display, generate_html_table  # noqa: E402
from valuation_update import valuation_update  # noqa: E402
from html_generator2 import prepare_and_generate_ticker_pages  # noqa: E402

DB_PATH = "Stock Data.db"


def fetch_10_year_treasury_yield():
    """Same helper main_remote.py uses — duplicated here to avoid importing
    main_remote (which runs heavy module-level setup)."""
    try:
        return yf.Ticker("^TNX").info.get("regularMarketPrice")
    except Exception as exc:
        print(f"[refresh_one_ticker] treasury fetch failed: {exc}")
        return None


def refresh(ticker: str) -> int:
    ticker = ticker.strip().upper()
    if not ticker:
        print("ERROR: ticker is required", file=sys.stderr)
        return 2

    if not os.path.exists(DB_PATH):
        print(f"ERROR: {DB_PATH} not found in cwd={os.getcwd()}", file=sys.stderr)
        return 1

    treasury = fetch_10_year_treasury_yield()
    if treasury is None:
        # Valuation_update tolerates None but the chart's "treasury overlay"
        # will be missing. Better than failing the whole refresh.
        print("[refresh_one_ticker] WARN: treasury yield unavailable, continuing")

    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    cursor = conn.cursor()

    try:
        # 1) Pull the assembled per-ticker data + market cap. This also writes
        # the financial table HTML used by the page template.
        prepared, mktcap = prepare_data_for_display(
            ticker, treasury, conn=conn, cursor=cursor, commit=False
        )
        generate_html_table(prepared, ticker)

        # Flush pending writes before valuation_update — that function opens
        # its own secondary connection for Splits and would deadlock otherwise
        # (we already hit this bug once in main_remote.py).
        conn.commit()

        # 2) Recompute valuation. This is the cell that actually depends on
        # nicks_growth_rate / projected_profit_margin in Tickers_Info.
        dashboard_data: list = []
        valuation_update(ticker, cursor, treasury, mktcap, dashboard_data)
        conn.commit()
    finally:
        try:
            plt.close("all")
        except Exception:
            pass
        conn.close()

    # 3) Re-render just this ticker's page. html_generator2 reads from the
    # files we just wrote in charts/ and pages/, so this picks up the new
    # valuation chart + tables.
    prepare_and_generate_ticker_pages([ticker])

    print(f"[refresh_one_ticker] {ticker} refreshed")
    return 0


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python scripts/refresh_one_ticker.py TICKER", file=sys.stderr)
        return 2
    return refresh(sys.argv[1])


if __name__ == "__main__":
    sys.exit(main())

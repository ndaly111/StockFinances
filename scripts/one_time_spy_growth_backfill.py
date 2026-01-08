#!/usr/bin/env python3
"""
One-time workflow to backfill SPY monthly prices, recompute PE/EPS, and regenerate growth pages.

Steps:
1) Import monthly SPY prices from CSV into Index_Price_History (older than existing cutoff).
2) Backfill Index_PE_History + Index_EPS_History for SPY using Trendonify + yfinance.
3) Recompute Index_Growth_History using available yields.
4) Regenerate spy_growth.html (and qqq_growth.html) pages.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backfill_index_growth import backfill_index_growth
from index_growth_charts import render_index_growth_charts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-time SPY growth backfill workflow")
    parser.add_argument("--db", default="Stock Data.db", help="Path to SQLite DB (default: Stock Data.db)")
    parser.add_argument(
        "--csv",
        default="spy_price_history_monthly_1993_present.csv",
        help="Path to SPY monthly price CSV committed in repo",
    )
    parser.add_argument("--ticker", default="SPY", help="Ticker to backfill (default: SPY)")
    parser.add_argument(
        "--table",
        default="Index_Price_History",
        help="Target price table name (default: Index_Price_History)",
    )
    parser.add_argument(
        "--create-table",
        action="store_true",
        help="Create Index_Price_History if it does not exist",
    )
    parser.add_argument(
        "--simulate-pre-etf",
        action="store_true",
        help="Simulate pre-ETF history in PE/EPS backfill using index proxies",
    )
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd()
    db_path = repo_root / args.db
    csv_path = repo_root / args.csv

    if not db_path.exists():
        raise SystemExit(f"DB not found: {db_path}")
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    import_cmd = [
        sys.executable,
        str(repo_root / "scripts" / "import_spy_monthly_prices_to_db.py"),
        "--db",
        str(db_path),
        "--csv",
        str(csv_path),
        "--ticker",
        args.ticker,
        "--table",
        args.table,
    ]
    if args.create_table:
        import_cmd.append("--create-table")
    run(import_cmd)

    pe_eps_cmd = [
        sys.executable,
        str(repo_root / "backfill_index_pe_eps.py"),
        "--db",
        str(db_path),
        "--tickers",
        args.ticker,
    ]
    if args.simulate_pre_etf:
        pe_eps_cmd.append("--simulate-pre-etf")
    run(pe_eps_cmd)

    print("[run] backfill_index_growth")
    backfill_index_growth(db_path=str(db_path))

    print("[run] render_index_growth_charts")
    # regenerate the Bokeh assets that spy_growth.html / qqq_growth.html load
    for tk in ("SPY", "QQQ"):
        render_index_growth_charts(tk)


if __name__ == "__main__":
    main()

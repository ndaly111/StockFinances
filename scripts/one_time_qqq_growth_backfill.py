#!/usr/bin/env python3
"""
One-time workflow to backfill QQQ monthly prices, PE/EPS, and regenerate growth pages.

Steps:
1) Generate QQQ monthly price CSV from yfinance (if it doesn't already exist).
2) Import monthly QQQ prices from CSV into Index_Price_History.
3) Backfill Index_PE_History + Index_EPS_History for QQQ using Trendonify + yfinance.
4) Recompute Index_Growth_History using available yields.
5) Regenerate spy_growth.html and qqq_growth.html pages.

This mirrors one_time_spy_growth_backfill.py but is tailored for QQQ, which
lacks pre-existing historical CSV files and instead relies on yfinance for
price data and Trendonify for P/E data.
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

DEFAULT_QQQ_CSV = "data/qqq_price_history_monthly_1999_present.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-time QQQ growth backfill workflow")
    parser.add_argument("--db", default="Stock Data.db", help="Path to SQLite DB")
    parser.add_argument(
        "--csv",
        default=DEFAULT_QQQ_CSV,
        help="Path to QQQ monthly price CSV (generated if missing)",
    )
    parser.add_argument("--ticker", default="QQQ", help="Ticker to backfill (default: QQQ)")
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
        help="Simulate pre-ETF history using ^NDX index proxy",
    )
    parser.add_argument(
        "--skip-csv-generation",
        action="store_true",
        help="Skip CSV generation step (use existing CSV only)",
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

    # Step 1: Generate QQQ monthly price CSV if it doesn't exist
    if not args.skip_csv_generation and not csv_path.exists():
        print(f"[step 1] Generating {csv_path} from yfinance...")
        run([
            sys.executable,
            str(repo_root / "scripts" / "generate_index_price_csv.py"),
            "--ticker",
            args.ticker,
            "--output",
            str(csv_path),
        ])
    elif csv_path.exists():
        print(f"[step 1] CSV already exists: {csv_path}")
    else:
        raise SystemExit(
            f"CSV not found: {csv_path}. "
            "Remove --skip-csv-generation to auto-generate it."
        )

    # Step 2: Import monthly prices into DB
    print("[step 2] Importing monthly prices into DB...")
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

    # Also load into Index_Price_History_Monthly for derive_monthly_pe compatibility
    print("[step 2b] Loading into Index_Price_History_Monthly...")
    run([
        sys.executable,
        str(repo_root / "scripts" / "load_index_price_csv.py"),
        "--db",
        str(db_path),
        "--csv",
        str(csv_path),
        "--ticker",
        args.ticker,
        "--close-column",
        "Close",
    ])

    # Step 3: Backfill PE/EPS from Trendonify + yfinance
    print("[step 3] Backfilling PE/EPS for QQQ...")
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

    # Step 4: Recompute implied growth
    print("[step 4] Recomputing implied growth...")
    backfill_index_growth(db_path=str(db_path))

    # Step 5: Regenerate charts and pages
    print("[step 5] Regenerating charts and pages...")
    for tk in ("SPY", "QQQ"):
        render_index_growth_charts(tk)


if __name__ == "__main__":
    main()

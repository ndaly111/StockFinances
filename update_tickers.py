"""Add / remove / update tickers in tickers.csv + Tickers_Info table.

tickers.csv carries ONLY the list of tickers (column: Ticker). The DB's
Tickers_Info table owns nicks_growth_rate and projected_profit_margin —
the CSV used to mirror those but the columns were dropped, and the old
write paths here pre-date that move. This file now keeps the two in
sync correctly: list-level operations (add/remove) touch the CSV;
assumption updates (update) only touch the DB.

Usage:
    python update_tickers.py add TICKER [growth] [margin]
    python update_tickers.py remove TICKER
    python update_tickers.py update TICKER GROWTH [MARGIN]
"""

from __future__ import annotations

import sqlite3
import sys

import pandas as pd

CSV_PATH = "tickers.csv"
DB_PATH = "Stock Data.db"
TICKER_COL = "Ticker"  # canonical column name in tickers.csv


def _normalize(t: str) -> str:
    return (t or "").strip().upper()


def update_csv(ticker: str, action: str) -> None:
    """List-level ops only (add / remove). 'update' is a no-op here because
    the growth/margin columns live in the DB, not the CSV."""
    ticker = _normalize(ticker)
    if action == "update":
        return  # nothing to do at the CSV level

    df = pd.read_csv(CSV_PATH)
    if TICKER_COL not in df.columns:
        # Be defensive: some older copies may have lowercase 'ticker'.
        lower = {c.lower(): c for c in df.columns}
        if "ticker" in lower:
            df = df.rename(columns={lower["ticker"]: TICKER_COL})
        else:
            raise SystemExit(
                f"{CSV_PATH} is missing the {TICKER_COL!r} column "
                f"(got: {list(df.columns)})"
            )

    existing = df[TICKER_COL].astype(str).str.upper().tolist()
    if action == "add":
        if ticker in existing:
            print(f"[update_tickers] {ticker} already in {CSV_PATH}; no add")
            return
        df = pd.concat(
            [df, pd.DataFrame([{TICKER_COL: ticker}])], ignore_index=True
        )
    elif action == "remove":
        if ticker not in existing:
            print(f"[update_tickers] {ticker} not in {CSV_PATH}; no remove")
            return
        df = df[df[TICKER_COL].astype(str).str.upper() != ticker]
    else:
        raise SystemExit(f"unknown action for CSV: {action!r}")

    df.to_csv(CSV_PATH, index=False)


def update_database(
    ticker: str,
    action: str,
    growth_rate: str | None = None,
    profit_margin: str | None = None,
) -> None:
    ticker = _normalize(ticker)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA busy_timeout=30000")
    cur = conn.cursor()

    if action == "add":
        # Insert if missing, otherwise just set the assumption values.
        cur.execute(
            "INSERT OR IGNORE INTO Tickers_Info (ticker) VALUES (?)",
            (ticker,),
        )
        cur.execute(
            "UPDATE Tickers_Info "
            "SET nicks_growth_rate = COALESCE(?, nicks_growth_rate), "
            "    projected_profit_margin = COALESCE(?, projected_profit_margin) "
            "WHERE ticker = ?",
            (growth_rate, profit_margin, ticker),
        )
    elif action == "remove":
        cur.execute("DELETE FROM Tickers_Info WHERE ticker = ?", (ticker,))
    elif action == "update":
        # Margin is optional — only overwrite when supplied.
        if profit_margin is not None and profit_margin != "":
            cur.execute(
                "UPDATE Tickers_Info "
                "SET nicks_growth_rate = ?, projected_profit_margin = ? "
                "WHERE ticker = ?",
                (growth_rate, profit_margin, ticker),
            )
        else:
            cur.execute(
                "UPDATE Tickers_Info SET nicks_growth_rate = ? WHERE ticker = ?",
                (growth_rate, ticker),
            )
        if cur.rowcount == 0:
            raise SystemExit(
                f"ticker {ticker!r} not found in Tickers_Info — "
                "use 'add' to create it first"
            )
    else:
        raise SystemExit(f"unknown action: {action!r}")

    conn.commit()
    conn.close()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    action = sys.argv[1]
    ticker = sys.argv[2]
    growth_rate = sys.argv[3] if len(sys.argv) > 3 else None
    profit_margin = sys.argv[4] if len(sys.argv) > 4 else None

    update_csv(ticker, action)
    update_database(ticker, action, growth_rate, profit_margin)
    print(f"[update_tickers] {action} {ticker} OK")

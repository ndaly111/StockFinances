#!/usr/bin/env python3
"""Generate a monthly price history CSV for a given ticker using yfinance.

Produces a CSV with columns: Date, Open, High, Low, Close, Volume, ChangePct
in the same format as the existing spy_price_history_monthly_1993_present.csv.

Usage:
    python scripts/generate_index_price_csv.py --ticker QQQ --output data/qqq_price_history_monthly_1999_present.csv
    python scripts/generate_index_price_csv.py --ticker SPY --output data/spy_price_history_monthly_1993_present.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate monthly price history CSV for a ticker",
    )
    parser.add_argument(
        "--ticker",
        required=True,
        help="Ticker symbol (e.g., QQQ, SPY)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV path (e.g., data/qqq_price_history_monthly_1999_present.csv)",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Start date (YYYY-MM-DD). Defaults to earliest available.",
    )
    return parser.parse_args()


def generate_index_price_csv(
    ticker: str,
    output_path: str,
    start: str | None = None,
) -> int:
    """Download monthly price history and save to CSV. Returns row count."""

    print(f"[generate_index_price_csv] Downloading {ticker} monthly history...")

    tk = yf.Ticker(ticker)
    hist = tk.history(
        period="max" if start is None else None,
        start=start,
        interval="1mo",
        auto_adjust=False,
        actions=False,
    )

    if hist.empty:
        print(f"[generate_index_price_csv] No data returned for {ticker}")
        return 0

    # Normalize index
    if getattr(hist.index, "tz", None) is not None:
        hist.index = hist.index.tz_localize(None)

    # Resample to month-start dates for consistency
    hist = hist.sort_index()
    hist.index = hist.index.to_period("M").to_timestamp()

    # Drop duplicates (keep last per month)
    hist = hist[~hist.index.duplicated(keep="last")]

    # Compute change percent
    hist["ChangePct"] = hist["Close"].pct_change()

    # Drop the first row if ChangePct is NaN (first month has no prior)
    hist = hist.dropna(subset=["Close"])

    # Format output
    df = pd.DataFrame(
        {
            "Date": hist.index.strftime("%Y-%m-%d"),
            "Open": hist["Open"].round(2),
            "High": hist["High"].round(2),
            "Low": hist["Low"].round(2),
            "Close": hist["Close"].round(2),
            "Volume": hist["Volume"].astype(float),
            "ChangePct": hist["ChangePct"],
        }
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(
        f"[generate_index_price_csv] Wrote {len(df)} rows to {out} "
        f"({df['Date'].iloc[0]} to {df['Date'].iloc[-1]})"
    )
    return len(df)


def main() -> None:
    args = _parse_args()
    count = generate_index_price_csv(
        ticker=args.ticker,
        output_path=args.output,
        start=args.start,
    )
    if count == 0:
        raise SystemExit("No data generated")


if __name__ == "__main__":
    main()

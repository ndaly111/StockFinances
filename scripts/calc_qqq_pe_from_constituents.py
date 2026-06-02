#!/usr/bin/env python3
"""Calculate QQQ trailing P/E from NDX-100 constituent earnings.

Method:
1. Load current QQQ holdings + weights (from a CSV of (ticker, weight_pct)).
2. For each constituent, pull annual diluted EPS (yfinance income_stmt, 4-5
   fiscal years) and daily close history.
3. Build a per-stock TTM EPS daily series by forward-filling the most recent
   reported annual EPS until the next fiscal year-end.
4. Aggregate to NDX-weighted P/E via earnings-yield averaging:
   weighted_yield(t) = Σ w_i × EPS_i(t) / Price_i(t)
   NDX_PE(t) = 1 / weighted_yield(t)
5. Upsert into Index_PE_History (QQQ, TTM). Implied growth must be recomputed
   afterward by running backfill_index_growth.py.

Caveats:
  - Uses CURRENT constituents and weights (survivorship bias). Companies that
    left QQQ are not represented; weights drift in reality.
  - Annual EPS, not quarterly, so the series steps once per fiscal year per
    stock. With ~100 stocks staggered, the aggregate is reasonably smooth.
  - Free yfinance gives at most ~5 fiscal years per stock. Older dates have
    no calculation possible; this script only writes rows where we have
    sufficient coverage (default: >=80% of index weight covered).
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


DB_PATH = "Stock Data.db"
TICKER = "QQQ"
HOLDINGS_CSV_DEFAULT = "data/ndx_holdings.csv"
MIN_COVERAGE = 0.80


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default=DB_PATH)
    p.add_argument("--holdings", default=HOLDINGS_CSV_DEFAULT)
    p.add_argument("--start", default="2020-01-01", help="Earliest date to consider.")
    p.add_argument("--end", default=None, help="Latest date (default: today).")
    p.add_argument("--min-coverage", type=float, default=MIN_COVERAGE,
                   help="Minimum fraction of index weight that must have data on a date to write it.")
    p.add_argument("--dry-run", action="store_true", help="Print result, don't write to DB.")
    p.add_argument("--delay", type=float, default=0.3, help="Seconds between yfinance calls.")
    return p.parse_args()


def _load_holdings(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "ticker" not in df or "weight_pct" not in df:
        raise SystemExit(f"{path} must have columns ticker,weight_pct")
    df["weight"] = df["weight_pct"].astype(float) / 100.0
    return df[["ticker", "weight"]].copy()


def _pull_ticker(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame | None:
    """Return DataFrame indexed by date with columns Price, TTM_EPS."""
    try:
        tk = yf.Ticker(symbol)
        ist = tk.income_stmt
        hist = tk.history(start=start, end=end + pd.Timedelta(days=1), auto_adjust=False, actions=False)
    except Exception as e:
        print(f"  [{symbol}] fetch err: {e}")
        return None
    if hist is None or hist.empty:
        return None
    if getattr(hist.index, "tz", None) is not None:
        hist.index = hist.index.tz_localize(None)
    price = hist["Close"].astype(float)
    price.index = pd.DatetimeIndex(price.index.date)

    # EPS from annual income statement.
    if ist is None or ist.empty or "Diluted EPS" not in ist.index:
        return None
    eps_row = ist.loc["Diluted EPS"].dropna()
    if eps_row.empty:
        return None
    # Index is fiscal year-end Timestamps. Sort ascending.
    eps_row.index = pd.DatetimeIndex([pd.Timestamp(d).normalize() for d in eps_row.index])
    eps_row = eps_row.sort_index()
    # For each date in price.index, find the most recent fiscal year-end whose
    # report would be public (assume reports become available ~60 days after FY end).
    report_lag = pd.Timedelta(days=60)
    eps_series = pd.Series(index=price.index, dtype=float)
    for ts in price.index:
        avail = eps_row[eps_row.index + report_lag <= ts]
        eps_series.loc[ts] = avail.iloc[-1] if not avail.empty else np.nan
    return pd.DataFrame({"Price": price, "TTM_EPS": eps_series}).dropna()


def main() -> int:
    args = _parse_args()
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end) if args.end else pd.Timestamp.today().normalize()
    print(f"Window: {start.date()} → {end.date()}")

    holdings = _load_holdings(args.holdings)
    holdings = holdings.sort_values("weight", ascending=False).reset_index(drop=True)
    print(f"Loaded {len(holdings)} holdings; total weight = {holdings['weight'].sum():.4f}")

    full_idx = pd.date_range(start=start, end=end, freq="B")
    yield_acc = pd.Series(0.0, index=full_idx)
    weight_acc = pd.Series(0.0, index=full_idx)

    for i, row in holdings.iterrows():
        sym, w = row["ticker"], float(row["weight"])
        print(f"[{i+1:>3}/{len(holdings)}] {sym:<8} w={w:.4f}")
        df = _pull_ticker(sym, start, end)
        time.sleep(args.delay)
        if df is None or df.empty:
            print(f"   skipped (no data)")
            continue
        df = df.reindex(full_idx).ffill()
        df = df.dropna()
        if df.empty:
            print(f"   skipped (no data after reindex)")
            continue
        # Earnings yield contribution: w_i × EPS_i / Price_i, only where EPS > 0
        valid = df[df["TTM_EPS"] > 0]
        contrib_yield = (w * valid["TTM_EPS"] / valid["Price"]).reindex(full_idx).fillna(0.0)
        contrib_weight = pd.Series(w, index=valid.index).reindex(full_idx).fillna(0.0)
        yield_acc = yield_acc.add(contrib_yield, fill_value=0.0)
        weight_acc = weight_acc.add(contrib_weight, fill_value=0.0)

    # NDX P/E only valid where coverage is above threshold
    ndx_pe = pd.Series(index=full_idx, dtype=float)
    sufficient = weight_acc >= args.min_coverage
    ndx_pe[sufficient] = weight_acc[sufficient] / yield_acc[sufficient]   # = 1 / (yield / weight) renormalized

    valid = ndx_pe.dropna()
    print(f"\nComputed P/E rows: {len(valid)}")
    if not valid.empty:
        print(f"  Date range:  {valid.index.min().date()} → {valid.index.max().date()}")
        print(f"  PE min/avg/max: {valid.min():.2f} / {valid.mean():.2f} / {valid.max():.2f}")
        print(f"  First 5: {valid.head().round(2).to_dict()}")
        print(f"  Last 5:  {valid.tail().round(2).to_dict()}")

    if args.dry_run or valid.empty:
        return 0

    rows = [(d.strftime("%Y-%m-%d"), TICKER, "TTM", float(v)) for d, v in valid.items()]
    with sqlite3.connect(args.db) as conn:
        cur = conn.cursor()
        cur.executemany(
            "INSERT OR REPLACE INTO Index_PE_History(Date,Ticker,PE_Type,PE_Ratio) VALUES (?,?,?,?)",
            rows,
        )
        conn.commit()
    print(f"\nWrote {len(rows)} P/E rows. Run backfill_index_growth.py next.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Calculate QQQ trailing P/E and EPS from NDX-100 constituent earnings.

Method:
1. Fetch current NDX-100 holdings + weights live from slickcharts.com.
2. For each constituent, pull annual diluted EPS (yfinance income_stmt, 4-5
   fiscal years) and daily close history.
3. Build a per-stock TTM EPS daily series by carrying forward the most recent
   reported annual EPS once it would be publicly available (FY end + 60 days).
4. Aggregate to NDX-weighted trailing P/E via earnings-yield averaging:
       weighted_yield(t) = Σ w_i × EPS_i(t) / Price_i(t)
       NDX_PE(t)         = 1 / weighted_yield(t)
5. Upsert Index_PE_History (QQQ, TTM) and Index_EPS_History
   (QQQ, IMPLIED_FROM_PE = QQQ_close / NDX_PE). Implied growth must be
   recomputed afterward by running backfill_index_growth.py.

Caveats:
  - Uses CURRENT constituents and weights (survivorship bias).
  - Annual EPS, not quarterly: per-stock series steps once per fiscal year.
    With ~100 staggered FY ends, the aggregate is reasonably smooth.
  - Free yfinance gives at most ~5 fiscal years per stock. Older dates with
    <80% weight coverage are skipped.

No CSV dependency: holdings are fetched fresh on each run.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
import time

import numpy as np
import pandas as pd
import requests
import yfinance as yf


DB_PATH = "Stock Data.db"
TICKER = "QQQ"
MIN_COVERAGE = 0.80
SLICKCHARTS_URL = "https://www.slickcharts.com/nasdaq100"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default=DB_PATH)
    p.add_argument("--start", default="2021-01-01", help="Earliest date to consider.")
    p.add_argument("--end", default=None, help="Latest date (default: today).")
    p.add_argument("--min-coverage", type=float, default=MIN_COVERAGE,
                   help="Minimum fraction of index weight required on a date to write it.")
    p.add_argument("--dry-run", action="store_true", help="Print result without writing to DB.")
    p.add_argument("--delay", type=float, default=0.2, help="Seconds between yfinance calls.")
    return p.parse_args()


def _fetch_holdings() -> pd.DataFrame:
    """Return DataFrame (ticker, weight) from slickcharts NDX-100 page."""
    resp = requests.get(SLICKCHARTS_URL, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    rows = re.findall(
        r'<td>(\d+)</td>.*?/symbol/([A-Z\.]+)".*?<td>([0-9.]+)%</td>',
        resp.text,
        re.DOTALL,
    )
    seen, out = set(), []
    for _rank, tk, wt in rows:
        if tk in seen:
            continue
        seen.add(tk)
        out.append((tk, float(wt) / 100.0))
    if not out:
        raise RuntimeError("Failed to parse slickcharts NDX-100 page.")
    df = pd.DataFrame(out, columns=["ticker", "weight"])
    return df.sort_values("weight", ascending=False).reset_index(drop=True)


def _pull_ticker(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame | None:
    """Return DataFrame indexed by date with columns Price, TTM_EPS for `symbol`."""
    try:
        tk = yf.Ticker(symbol)
        ist = tk.income_stmt
        hist = tk.history(start=start, end=end + pd.Timedelta(days=1), auto_adjust=False, actions=False)
    except Exception as exc:
        print(f"  [{symbol}] fetch err: {exc}")
        return None
    if hist is None or hist.empty:
        return None
    if getattr(hist.index, "tz", None) is not None:
        hist.index = hist.index.tz_localize(None)
    price = hist["Close"].astype(float)
    price.index = pd.DatetimeIndex(price.index.date)

    if ist is None or ist.empty or "Diluted EPS" not in ist.index:
        return None
    eps_row = ist.loc["Diluted EPS"].dropna()
    if eps_row.empty:
        return None
    eps_row.index = pd.DatetimeIndex([pd.Timestamp(d).normalize() for d in eps_row.index])
    eps_row = eps_row.sort_index()

    report_lag = pd.Timedelta(days=60)
    eps_series = pd.Series(index=price.index, dtype=float)
    for ts in price.index:
        avail = eps_row[eps_row.index + report_lag <= ts]
        eps_series.loc[ts] = avail.iloc[-1] if not avail.empty else np.nan
    return pd.DataFrame({"Price": price, "TTM_EPS": eps_series}).dropna()


def _qqq_prices(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Return QQQ daily close, used to derive ETF-level implied EPS."""
    hist = yf.Ticker("QQQ").history(start=start, end=end + pd.Timedelta(days=1), auto_adjust=False, actions=False)
    if getattr(hist.index, "tz", None) is not None:
        hist.index = hist.index.tz_localize(None)
    s = hist["Close"].astype(float)
    s.index = pd.DatetimeIndex(s.index.date)
    return s[~s.index.duplicated(keep="last")].sort_index()


def main() -> int:
    args = _parse_args()
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end) if args.end else pd.Timestamp.today().normalize()
    print(f"Window: {start.date()} → {end.date()}")

    holdings = _fetch_holdings()
    print(f"Fetched {len(holdings)} NDX-100 holdings (total weight {holdings['weight'].sum():.4f}) from slickcharts.")

    full_idx = pd.date_range(start=start, end=end, freq="B")
    yield_acc = pd.Series(0.0, index=full_idx)
    weight_acc = pd.Series(0.0, index=full_idx)

    for i, row in holdings.iterrows():
        sym, w = row["ticker"], float(row["weight"])
        print(f"[{i+1:>3}/{len(holdings)}] {sym:<8} w={w:.4f}")
        df = _pull_ticker(sym, start, end)
        time.sleep(args.delay)
        if df is None or df.empty:
            print("   skipped (no data)")
            continue
        df = df.reindex(full_idx).ffill().dropna()
        if df.empty:
            print("   skipped (no data after reindex)")
            continue
        valid = df[df["TTM_EPS"] > 0]
        if valid.empty:
            continue
        contrib_yield = (w * valid["TTM_EPS"] / valid["Price"]).reindex(full_idx).fillna(0.0)
        contrib_weight = pd.Series(w, index=valid.index).reindex(full_idx).fillna(0.0)
        yield_acc = yield_acc.add(contrib_yield, fill_value=0.0)
        weight_acc = weight_acc.add(contrib_weight, fill_value=0.0)

    sufficient = weight_acc >= args.min_coverage
    ndx_pe = pd.Series(index=full_idx, dtype=float)
    ndx_pe[sufficient] = weight_acc[sufficient] / yield_acc[sufficient]
    ndx_pe = ndx_pe.dropna()

    print(f"\nComputed P/E rows: {len(ndx_pe)}")
    if ndx_pe.empty:
        print("Nothing to write.")
        return 0
    print(f"  Date range: {ndx_pe.index.min().date()} → {ndx_pe.index.max().date()}")
    print(f"  PE min/avg/max: {ndx_pe.min():.2f} / {ndx_pe.mean():.2f} / {ndx_pe.max():.2f}")

    qqq_px = _qqq_prices(start, end)
    aligned = pd.DataFrame({"PE": ndx_pe, "Close": qqq_px}).dropna()
    aligned["EPS_ETF"] = aligned["Close"] / aligned["PE"]
    print(f"  EPS (ETF-level) min/avg/max: {aligned['EPS_ETF'].min():.2f} / {aligned['EPS_ETF'].mean():.2f} / {aligned['EPS_ETF'].max():.2f}")

    if args.dry_run:
        return 0

    pe_rows  = [(d.strftime("%Y-%m-%d"), TICKER, "TTM",                 float(r["PE"]))      for d, r in aligned.iterrows()]
    eps_rows = [(d.strftime("%Y-%m-%d"), TICKER, "IMPLIED_FROM_PE",     float(r["EPS_ETF"])) for d, r in aligned.iterrows()]

    with sqlite3.connect(args.db) as conn:
        cur = conn.cursor()
        cur.executemany(
            "INSERT OR REPLACE INTO Index_PE_History(Date,Ticker,PE_Type,PE_Ratio) VALUES (?,?,?,?)",
            pe_rows,
        )
        cur.executemany(
            "INSERT OR REPLACE INTO Index_EPS_History(Date,Ticker,EPS_Type,EPS) VALUES (?,?,?,?)",
            eps_rows,
        )
        conn.commit()
    print(f"\nWrote {len(pe_rows)} PE rows and {len(eps_rows)} EPS rows. Run backfill_index_growth.py next.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

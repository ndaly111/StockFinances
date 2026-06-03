#!/usr/bin/env python3
"""Compute QQQ trailing P/E and EPS from per-stock SEC EDGAR XBRL filings.

For each current NDX-100 constituent we fetch its diluted EPS as reported in
its 10-Q and 10-K filings (via the SEC XBRL company-concept API), reconstruct a
quarterly EPS time series (deriving Q4 from FY - Q1 - Q2 - Q3), then build a
trailing-12-month EPS series by summing the most recent four quarters whose
reports were public at each date (45-day reporting lag).

Daily prices come from yfinance. Per-stock daily P/E = price / TTM_EPS.
NDX-weighted P/E uses earnings-yield aggregation:
    yield(t) = Σ w_i × EPS_i(t) / Price_i(t)
    NDX_PE(t) = 1 / yield(t)

Writes:
    Index_PE_History  (QQQ, TTM, PE_Ratio)
    Index_EPS_History (QQQ, IMPLIED_FROM_PE, EPS = QQQ_close / NDX_PE)

Holdings come live from slickcharts. Constituents that don't file with the SEC
(e.g. foreign-listed ASML, CCEP, FER, TRI, etc.) are dropped from the
aggregation and reported in coverage stats.

Run:
    python scripts/calc_qqq_pe_from_edgar.py --start 2015-01-01
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sqlite3
import sys
import time
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf


DB_PATH = "Stock Data.db"
TICKER = "QQQ"
SLICKCHARTS_URL = "https://www.slickcharts.com/nasdaq100"
EDGAR_CIK_URL = "https://www.sec.gov/files/company_tickers.json"
EDGAR_CONCEPT_URL = (
    "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/{taxonomy}/{concept}.json"
)
USER_AGENT = "Nick Daly StockFinances ndaly111@gmail.com"
HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/json"}
REPORT_LAG_DAYS = 45
MIN_COVERAGE = 0.70


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default=DB_PATH)
    p.add_argument("--start", default="2016-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--min-coverage", type=float, default=MIN_COVERAGE)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--edgar-delay", type=float, default=0.15)
    p.add_argument("--yf-delay", type=float, default=0.15)
    return p.parse_args()


# ---------------------------------------------------------------- holdings --
def _fetch_holdings() -> pd.DataFrame:
    r = requests.get(SLICKCHARTS_URL, headers={"User-Agent": USER_AGENT}, timeout=30)
    r.raise_for_status()
    rows = re.findall(
        r'<td>(\d+)</td>.*?/symbol/([A-Z\.]+)".*?<td>([0-9.]+)%</td>',
        r.text, re.DOTALL,
    )
    seen, out = set(), []
    for _rank, tk, wt in rows:
        if tk in seen: continue
        seen.add(tk)
        out.append((tk, float(wt) / 100.0))
    if not out:
        raise RuntimeError("Failed to parse slickcharts holdings.")
    return pd.DataFrame(out, columns=["ticker", "weight"]).sort_values("weight", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------- EDGAR --
_CIK_MAP_CACHE: dict[str, str] = {}
def _load_cik_map() -> dict[str, str]:
    if _CIK_MAP_CACHE:
        return _CIK_MAP_CACHE
    r = requests.get(EDGAR_CIK_URL, headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    for _, row in data.items():
        tk = row.get("ticker")
        cik = row.get("cik_str")
        if tk and cik:
            _CIK_MAP_CACHE[tk.upper()] = f"{int(cik):010d}"
    return _CIK_MAP_CACHE


def _fetch_eps_concept(cik: str) -> Optional[list[dict]]:
    """Return list of XBRL EPS entries for this CIK (US-GAAP diluted preferred)."""
    candidates = [
        ("us-gaap", "EarningsPerShareDiluted"),
        ("us-gaap", "EarningsPerShareBasic"),
        ("ifrs-full", "DilutedEarningsLossPerShare"),
        ("ifrs-full", "BasicEarningsLossPerShare"),
    ]
    for tax, concept in candidates:
        url = EDGAR_CONCEPT_URL.format(cik=cik, taxonomy=tax, concept=concept)
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
        except Exception:
            continue
        if r.status_code != 200:
            continue
        try:
            data = r.json()
        except Exception:
            continue
        units = data.get("units", {})
        # Prefer USD/shares unit if multiple
        for unit_key in ("USD/shares", "EUR/shares", "GBP/shares", "shares"):
            if unit_key in units and units[unit_key]:
                return units[unit_key]
        # Fall back to whatever single unit it has
        for v in units.values():
            if v: return v
    return None


def _quarterly_series(entries: list[dict]) -> pd.Series:
    """Build a deduped quarterly EPS series from EDGAR XBRL entries.

    EDGAR's ``fy`` field labels the REPORTING year of the filing — when a 10-Q
    references its prior-year comparison data, the comparison gets stamped
    with the current fy. So (fy, fp) isn't unique per quarter. Use end_date
    as the canonical key instead.

    Approach:
      - Collect 90-day (single-quarter) entries: end_date → val.
      - Collect 363-day annual (10-K) entries: annual_end → val.
      - Each annual: derive Q4 = annual - (sum of last 3 quarterly EPS whose
        end < annual_end and within ~270 days prior).
    """
    quarterly: dict[pd.Timestamp, float] = {}
    annuals: dict[pd.Timestamp, float] = {}

    for v in entries:
        sd = v.get("start"); ed = v.get("end"); val = v.get("val")
        if sd is None or ed is None or val is None:
            continue
        try:
            d_start = dt.date.fromisoformat(sd)
            d_end = dt.date.fromisoformat(ed)
        except Exception:
            continue
        span = (d_end - d_start).days
        ts = pd.Timestamp(d_end).normalize()
        if 75 <= span <= 105:
            # Single-quarter EPS. First-seen wins (avoids weird re-statements).
            quarterly.setdefault(ts, float(val))
        elif 340 <= span <= 380 and v.get("form") == "10-K":
            annuals.setdefault(ts, float(val))

    pts: dict[pd.Timestamp, float] = dict(quarterly)
    q_sorted = sorted(quarterly.items())
    for ann_end, ann_val in annuals.items():
        # Find the three single-quarter entries whose end falls strictly within
        # the trailing 270 days before this annual end (and on or after start
        # of the annual fiscal year ≈ ann_end - 365d).
        window_start = ann_end - pd.Timedelta(days=365)
        prior_qs = [v for (qe, v) in q_sorted if window_start < qe < ann_end]
        if len(prior_qs) >= 3:
            q4 = ann_val - sum(prior_qs[-3:])
            pts[ann_end] = q4   # overrides if a 90-day Q4 entry also existed

    if not pts:
        return pd.Series(dtype=float)
    s = pd.Series(pts).sort_index()
    return s


def _ttm_series(quarterly_eps: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    """For each date in idx, return TTM EPS = sum of 4 most recent quarters
    whose end + report_lag <= date.
    """
    if quarterly_eps.empty:
        return pd.Series(np.nan, index=idx)
    q = quarterly_eps.sort_index()
    lag = pd.Timedelta(days=REPORT_LAG_DAYS)
    out_vals = np.full(len(idx), np.nan)
    # For efficiency, iterate idx as monotonic and maintain a sliding pointer
    avail_dates = q.index + lag  # when each quarter becomes "public"
    avail_dates = avail_dates.to_numpy()
    q_vals = q.values
    q_ends = q.index.to_numpy()
    idx_arr = idx.to_numpy()

    for i, d in enumerate(idx_arr):
        # Quarters whose report is public at date d
        public_mask = avail_dates <= d
        if not public_mask.any():
            continue
        # Pick last 4 quarters by end date
        public_idx = np.where(public_mask)[0]
        last4 = public_idx[-4:]
        if len(last4) < 4:
            continue  # Need full TTM (4 quarters)
        out_vals[i] = float(np.sum(q_vals[last4]))
    return pd.Series(out_vals, index=idx)


def _yf_close_and_splits(symbol: str, start: pd.Timestamp, end: pd.Timestamp):
    """Return (close_series, splits_series). Close is split-adjusted via
    yfinance auto_adjust=True (back-adjusted to current share basis).
    splits_series has split events keyed by date.
    """
    try:
        tk = yf.Ticker(symbol)
        h = tk.history(start=start, end=end + pd.Timedelta(days=1), auto_adjust=True, actions=True)
        splits = tk.splits
    except Exception as e:
        print(f"   [{symbol}] yfinance err: {e}")
        return pd.Series(dtype=float), pd.Series(dtype=float)
    if h is None or h.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    if getattr(h.index, "tz", None) is not None:
        h.index = h.index.tz_localize(None)
    s = h["Close"].astype(float)
    s.index = pd.DatetimeIndex(s.index.date)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    if splits is None or splits.empty:
        splits = pd.Series(dtype=float)
    else:
        splits = splits.copy()
        if getattr(splits.index, "tz", None) is not None:
            splits.index = splits.index.tz_localize(None)
        splits.index = pd.DatetimeIndex(splits.index.date)
        splits = splits[splits > 0].sort_index()
    return s, splits


def _split_adjust(quarterly: pd.Series, splits: pd.Series) -> pd.Series:
    """Adjust historical EPS values to current share basis.

    For each EPS reported on quarter end E with value V, the
    post-adjustment value is V / Π(split_ratio_k) over all split events
    occurring after E. yfinance's split column already gives the ratio
    (e.g. 10.0 for a 10-for-1 split).
    """
    if quarterly.empty or splits.empty:
        return quarterly
    out = quarterly.copy()
    # Sort splits ascending
    splits_sorted = splits.sort_index()
    for end_ts, val in quarterly.items():
        factor = 1.0
        for sd, ratio in splits_sorted.items():
            if sd > end_ts:
                factor *= float(ratio)
        out.loc[end_ts] = float(val) / factor
    return out


# ------------------------------------------------------------------ main --
def main() -> int:
    args = _parse_args()
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end) if args.end else pd.Timestamp.today().normalize()
    print(f"Window: {start.date()} → {end.date()}")

    holdings = _fetch_holdings()
    print(f"Fetched {len(holdings)} NDX-100 constituents (total weight {holdings['weight'].sum():.4f})")
    cik_map = _load_cik_map()
    print(f"Loaded EDGAR ticker→CIK map: {len(cik_map)} entries")

    full_idx = pd.date_range(start=start, end=end, freq="B")
    yield_acc = pd.Series(0.0, index=full_idx)
    weight_acc = pd.Series(0.0, index=full_idx)

    no_cik = no_eps = ok = 0
    for i, row in holdings.iterrows():
        sym, w = row["ticker"], float(row["weight"])
        cik = cik_map.get(sym.upper())
        if not cik:
            print(f"[{i+1:>3}/{len(holdings)}] {sym:<8} w={w:.4f}  no CIK in EDGAR")
            no_cik += 1
            continue

        entries = _fetch_eps_concept(cik)
        time.sleep(args.edgar_delay)
        if not entries:
            print(f"[{i+1:>3}/{len(holdings)}] {sym:<8} w={w:.4f}  CIK={cik} no EPS concept")
            no_eps += 1
            continue
        q_eps = _quarterly_series(entries)
        if q_eps.empty or len(q_eps) < 4:
            print(f"[{i+1:>3}/{len(holdings)}] {sym:<8} w={w:.4f}  insufficient quarters ({len(q_eps)})")
            no_eps += 1
            continue

        px, splits = _yf_close_and_splits(sym, start, end)
        time.sleep(args.yf_delay)
        if px.empty:
            print(f"[{i+1:>3}/{len(holdings)}] {sym:<8} w={w:.4f}  no price")
            continue

        # Split-adjust EPS to current share basis so it matches yfinance
        # auto-adjusted prices.
        q_eps_adj = _split_adjust(q_eps, splits)

        # Align to business-day full_idx
        px_aligned = px.reindex(full_idx).ffill()
        ttm = _ttm_series(q_eps_adj, full_idx)
        df = pd.DataFrame({"Price": px_aligned, "TTM_EPS": ttm}).dropna()
        df = df[df["TTM_EPS"] > 0]
        if df.empty:
            print(f"[{i+1:>3}/{len(holdings)}] {sym:<8} w={w:.4f}  no valid TTM")
            continue

        n_first = df.index.min().date()
        n_last = df.index.max().date()
        print(f"[{i+1:>3}/{len(holdings)}] {sym:<8} w={w:.4f}  q={len(q_eps):>3}  ttm-coverage {n_first} → {n_last}")
        ok += 1

        yield_acc = yield_acc.add(
            (w * df["TTM_EPS"] / df["Price"]).reindex(full_idx).fillna(0.0),
            fill_value=0.0,
        )
        weight_acc = weight_acc.add(
            pd.Series(w, index=df.index).reindex(full_idx).fillna(0.0),
            fill_value=0.0,
        )

    print(f"\nConstituents: ok={ok}  no_cik={no_cik}  no_eps={no_eps}")
    sufficient = weight_acc >= args.min_coverage
    print(f"Dates with coverage >= {args.min_coverage:.0%}: {int(sufficient.sum())} / {len(full_idx)}")

    ndx_pe = pd.Series(index=full_idx, dtype=float)
    ndx_pe[sufficient] = weight_acc[sufficient] / yield_acc[sufficient]
    ndx_pe = ndx_pe.dropna()
    if ndx_pe.empty:
        print("Nothing to write.")
        return 0
    print(f"\nNDX PE range: {ndx_pe.index.min().date()} → {ndx_pe.index.max().date()}  ({len(ndx_pe)} rows)")
    print(f"  PE min/avg/max: {ndx_pe.min():.2f} / {ndx_pe.mean():.2f} / {ndx_pe.max():.2f}")

    qqq_px, _qqq_splits = _yf_close_and_splits("QQQ", start, end)
    merged = pd.DataFrame({"PE": ndx_pe, "Close": qqq_px}).dropna()
    merged["EPS_ETF"] = merged["Close"] / merged["PE"]

    if args.dry_run:
        print("(dry-run) Not writing to DB.")
        # Sample
        for ts in [pd.Timestamp("2016-06-01"), pd.Timestamp("2018-01-02"),
                   pd.Timestamp("2020-03-23"), pd.Timestamp("2022-06-15"),
                   pd.Timestamp("2024-12-31"), merged.index.max()]:
            if ts in merged.index:
                r = merged.loc[ts]
                print(f"  {ts.date()}: PE={r['PE']:.2f}  EPS_ETF={r['EPS_ETF']:.2f}  EPS_display={r['EPS_ETF']*4:.2f}")
        return 0

    pe_rows = [(d.strftime("%Y-%m-%d"), TICKER, "TTM", float(r["PE"])) for d, r in merged.iterrows()]
    eps_rows = [(d.strftime("%Y-%m-%d"), TICKER, "IMPLIED_FROM_PE", float(r["EPS_ETF"])) for d, r in merged.iterrows()]
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
    print(f"Wrote {len(pe_rows)} PE rows and {len(eps_rows)} EPS rows. Run backfill_index_growth.py next.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

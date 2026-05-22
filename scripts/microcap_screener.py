"""Microcap value screener — find undervalued small companies.

Thesis: microcaps below ~$500M get little institutional attention. Some are
priced as if their growth will slow dramatically when they've been growing
consistently for years. The screener surfaces companies where:

  * Historical 5-yr EPS CAGR (or revenue CAGR if not profitable 5 yrs ago)
    is materially higher than the implied growth rate baked into the current
    price.
  * Debt/Equity < 0.5 (lower is better).
  * Growth is reasonably consistent (we don't filter, but we report
    "positive years out of 5" and the YoY sequence so chaotic patterns are
    visible at a glance).

This module is the analytic core — given a ticker, it returns a dict with
all the screening metrics. The universe-acquisition layer (NYSE + NASDAQ
listed, market cap < $500M) sits in a separate script that calls into this
one. For now the entry point at the bottom runs the analysis against any
list of tickers (CSV or argv) and writes a ranked CSV.

Output columns:
  Ticker, Name, Sector, MarketCap, MetricUsed, CAGR5yr, YearsPositive,
  YoYSequence, ImpliedGrowth, Gap, DebtEquity, BusinessSummary
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

# Import the EDGAR provider from the existing pipeline so we share the same
# code path the watchlist uses for historical financials.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_providers.edgar import EdgarDataProvider, DataProviderError  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("microcap_screener")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "data" / "microcap_candidates.csv"
DEFAULT_LIST = REPO_ROOT / "tickers.csv"

# Cap that lets in true microcaps but skips nano/penny territory.
# Bumped 2026-05-22: $500M missed MAMA at its current $598M; $1B gives more
# headroom for "still undercovered" names.
MARKET_CAP_CEILING = 1_000_000_000
# Lower bound (skip nano under this — illiquid / data gaps).
MARKET_CAP_FLOOR = 50_000_000
# Balance-sheet gate.
DEBT_EQUITY_CEILING = 0.5
# Treasury yield used in implied-growth formula. Pull dynamically when
# available; fall back to a reasonable static guess so screening still works
# offline.
DEFAULT_TREASURY_YIELD = 0.045

# One shared EDGAR provider per process. The class caches CIK lookups in
# memory; we pre-populate the cache once so each subsequent ticker lookup
# doesn't re-download company_tickers.json.
_EDGAR: Optional[EdgarDataProvider] = None


def _get_edgar() -> EdgarDataProvider:
    global _EDGAR
    if _EDGAR is not None:
        return _EDGAR
    prov = EdgarDataProvider()
    try:
        log.info("Pre-warming EDGAR ticker -> CIK cache (one company_tickers.json download)...")
        resp = prov.session.get(prov.TICKERS_URL, timeout=30)
        resp.raise_for_status()
        for rec in resp.json().values():
            t = str(rec.get("ticker", "")).upper()
            if t:
                prov._cik_cache[t] = str(rec["cik_str"]).zfill(10)
        log.info(f"  EDGAR cache: {len(prov._cik_cache)} ticker -> CIK mappings ready")
    except Exception as exc:
        log.warning(f"EDGAR cache pre-warm failed: {exc}; will lazy-resolve per ticker")
    _EDGAR = prov
    return _EDGAR


def _edgar_annual_series(ticker: str, metric: str) -> Optional[pd.Series]:
    """Pull a 5+ year annual series from EDGAR. metric is 'EPS' or 'Revenue'.

    Returns a pd.Series indexed by fiscal year-end date string, oldest-first.
    Returns None if EDGAR has no data for this ticker (foreign issuers,
    private companies that just IPO'd, tickers not in EDGAR).
    """
    prov = _get_edgar()
    try:
        records = prov.fetch_annual_financials(ticker)
    except DataProviderError:
        return None
    except Exception as exc:
        log.debug(f"{ticker}: EDGAR fetch failed: {exc}")
        return None

    if not records:
        return None

    field = "EPS" if metric == "EPS" else "Revenue"
    pairs: list[tuple[str, float]] = []
    for r in records:
        v = r.get(field)
        if v is None:
            continue
        try:
            pairs.append((str(r["Date"]), float(v)))
        except (TypeError, ValueError):
            continue

    if not pairs:
        return None

    pairs.sort(key=lambda x: x[0])  # oldest-first
    return pd.Series(
        data=[p[1] for p in pairs],
        index=[p[0] for p in pairs],
        name=metric,
    )


@dataclass
class Candidate:
    ticker: str
    name: str
    sector: str
    industry: str
    market_cap: float
    metric_used: str           # "EPS" or "Revenue"
    cagr_5yr: float            # decimal: 0.18 = 18%
    years_positive: str        # e.g. "4/5"
    yoy_sequence: str          # e.g. "+18,+22,-4,+15,+28"
    implied_growth: float      # decimal
    finviz_5yr: Optional[float] # decimal, if available
    gap: float                 # cagr_5yr - implied_growth
    debt_equity: float
    business_summary: str
    reasons_skipped: str       # diagnostic if this candidate was filtered out


def _safe_float(v) -> Optional[float]:
    if v is None or pd.isna(v):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _annual_series(yticker: yf.Ticker, metric: str) -> Optional[pd.Series]:
    """Pull an annual series (most recent first) from yfinance .financials.

    metric: 'EPS' or 'Revenue'. yfinance's .financials returns a DataFrame
    keyed by fiscal year. For EPS we'd ideally use 'Basic EPS' or
    'Diluted EPS'; for Revenue, 'Total Revenue'.
    """
    try:
        fin = yticker.financials
    except Exception as exc:
        log.debug(f"{yticker.ticker}: .financials failed: {exc}")
        return None
    if fin is None or fin.empty:
        return None

    row_candidates = {
        "EPS": ["Basic EPS", "Diluted EPS"],
        "Revenue": ["Total Revenue", "Revenue"],
    }.get(metric, [])

    for row_name in row_candidates:
        if row_name in fin.index:
            series = fin.loc[row_name].dropna()
            if not series.empty:
                # yfinance returns most-recent first; reverse to oldest-first
                # for CAGR math.
                series = series.iloc[::-1]
                return series
    return None


def _compute_5yr_cagr(series: pd.Series, window: int = 5) -> tuple[Optional[float], int, str]:
    """Return (cagr, years_positive_count, yoy_sequence_str) for a series.

    Series is oldest-first. We use the most recent (window+1) values so the
    CAGR is over exactly `window` YoY periods even when EDGAR gives us 10
    years of history. If the series is shorter than window+1, we use the
    full span (and the result is over fewer years — still reported, caller
    can see it via the YoY sequence).
    """
    if series is None or len(series) < 2:
        return None, 0, ""

    s = series.dropna().astype(float)
    if len(s) < 2:
        return None, 0, ""

    # Restrict to the most recent (window+1) annual data points so 5-yr CAGR
    # is actually 5 YoY periods, not 9 when EDGAR returns a deeper history.
    if len(s) > window + 1:
        s = s.iloc[-(window + 1):]

    first = float(s.iloc[0])
    last = float(s.iloc[-1])
    span = len(s) - 1  # number of year-over-year periods

    # CAGR is undefined when start <= 0; we let the fallback (revenue) handle
    # the unprofitable case at the caller layer.
    if first <= 0:
        return None, 0, ""
    if last <= 0:
        # finished worse than where we started — return a negative growth rate
        # so the caller can show it but the gap will likely sort low.
        cagr = None
    else:
        cagr = (last / first) ** (1.0 / span) - 1.0

    # YoY sequence + positive-years count
    yoy_pcts: list[float] = []
    for i in range(1, len(s)):
        prev = float(s.iloc[i - 1])
        cur = float(s.iloc[i])
        if prev <= 0:
            yoy_pcts.append(float("nan"))
            continue
        yoy_pcts.append((cur - prev) / prev)

    pos_years = sum(1 for x in yoy_pcts if not pd.isna(x) and x > 0)
    total_years = sum(1 for x in yoy_pcts if not pd.isna(x))
    seq = ",".join(
        f"{x*100:+.0f}" if not pd.isna(x) else "?" for x in yoy_pcts
    )
    pos_str = f"{pos_years}/{total_years}" if total_years else "—"
    return cagr, pos_years, f"{pos_str} ({seq})"


def _implied_growth(pe_ratio: Optional[float], treasury_yield: float) -> Optional[float]:
    """Recreate the formula the rest of StockFinances uses for index implied
    growth: g = (PE / 10) ^ (1/10) + r - 1. Works for any single stock as
    long as PE > 0."""
    if pe_ratio is None or pe_ratio <= 0:
        return None
    try:
        return (pe_ratio / 10.0) ** 0.1 + treasury_yield - 1.0
    except (ValueError, ZeroDivisionError):
        return None


def _debt_equity(info: dict) -> Optional[float]:
    """yfinance reports debt/equity as a number (sometimes 0–100, sometimes
    0–1). We standardize to a 0–1+ decimal ratio."""
    de = _safe_float(info.get("debtToEquity"))
    if de is None:
        return None
    # yfinance returns this as a percentage when > 5 (heuristic). Anything
    # below 5 is already a ratio. This is unfortunately how their schema
    # works in practice.
    if de > 5:
        de = de / 100.0
    return de


def _current_treasury_yield() -> float:
    """Try to pull the 10Y treasury yield via ^TNX; fall back to default."""
    try:
        v = yf.Ticker("^TNX").info.get("regularMarketPrice")
        if v is not None and 0 < v < 100:
            return float(v) / 100.0
    except Exception:
        pass
    return DEFAULT_TREASURY_YIELD


def screen_ticker(ticker: str, treasury_yield: float) -> Candidate:
    """Run the full analytic pipeline for one ticker. Always returns a
    Candidate; sets reasons_skipped if any disqualifying condition hit."""
    t = ticker.strip().upper()
    skipped: list[str] = []
    try:
        yt = yf.Ticker(t)
        info = yt.info or {}
    except Exception as exc:
        log.debug(f"{t}: .info failed: {exc}")
        info = {}
        skipped.append(f"info_fetch_failed:{exc}")

    market_cap = _safe_float(info.get("marketCap")) or 0.0
    if market_cap == 0:
        skipped.append("no_market_cap")
    elif market_cap > MARKET_CAP_CEILING:
        skipped.append("too_large")
    elif market_cap < MARKET_CAP_FLOOR:
        skipped.append("too_small")

    # Early out for size-filtered tickers: skip the financials HTTP round-trip
    # (that's the slow leg). Saves ~3-4 min when screening the full universe.
    size_filtered = any(s in skipped for s in ("too_large", "too_small", "no_market_cap"))

    de = _debt_equity(info)
    if de is None:
        skipped.append("no_debt_equity")
    elif de > DEBT_EQUITY_CEILING:
        skipped.append(f"de_too_high:{de:.2f}")

    metric_used = "EPS"
    cagr = None
    pos_years = 0
    yoy_str = ""
    if not size_filtered:
        # EDGAR first (10-year history for proper 5-yr CAGR), yfinance as
        # fallback for tickers EDGAR doesn't have (ADRs, foreign issuers,
        # recent IPOs not yet in XBRL).
        series = _edgar_annual_series(t, "EPS")
        if series is None or series.empty:
            series = _annual_series(yt, "EPS")
        cagr, pos_years, yoy_str = _compute_5yr_cagr(series)
        if cagr is None:
            metric_used = "Revenue"
            series = _edgar_annual_series(t, "Revenue")
            if series is None or series.empty:
                series = _annual_series(yt, "Revenue")
            cagr, pos_years, yoy_str = _compute_5yr_cagr(series)
        if cagr is None:
            skipped.append("no_growth_data")

    pe = _safe_float(info.get("trailingPE")) or _safe_float(info.get("forwardPE"))
    implied = _implied_growth(pe, treasury_yield)
    if implied is None and not size_filtered:
        skipped.append("no_implied_growth")

    gap = (
        cagr - implied
        if cagr is not None and implied is not None
        else float("nan")
    )

    return Candidate(
        ticker=t,
        name=info.get("shortName") or info.get("longName") or t,
        sector=info.get("sector") or "",
        industry=info.get("industry") or "",
        market_cap=market_cap,
        metric_used=metric_used,
        cagr_5yr=cagr if cagr is not None else float("nan"),
        years_positive=yoy_str.split(" ")[0] if yoy_str else "",
        yoy_sequence=yoy_str.split(" ", 1)[1] if " " in yoy_str else "",
        implied_growth=implied if implied is not None else float("nan"),
        finviz_5yr=None,  # populated later by the universe layer if useful
        gap=gap,
        debt_equity=de if de is not None else float("nan"),
        business_summary=(info.get("longBusinessSummary") or "")[:600],
        reasons_skipped=", ".join(skipped),
    )


def screen_many(tickers: list[str], max_workers: int = 10) -> list[Candidate]:
    treasury = _current_treasury_yield()
    log.info(f"Using treasury yield {treasury:.4f}")
    out: list[Candidate] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(screen_ticker, t, treasury): t for t in tickers}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                out.append(fut.result())
            except Exception as exc:
                log.warning(f"{futs[fut]}: failed — {exc}")
            if i % 25 == 0:
                log.info(f"  {i}/{len(tickers)} done in {time.time()-t0:.0f}s")
    log.info(f"Screened {len(out)} tickers in {time.time()-t0:.0f}s")
    return out


def write_csv(candidates: list[Candidate], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Keep only candidates that passed all the hard filters; rank by gap.
    passed = [c for c in candidates if not c.reasons_skipped]
    passed.sort(key=lambda c: c.gap, reverse=True)
    rows = [asdict(c) for c in passed]
    # Also include skipped ones in a sidecar so we can see what got rejected
    # and why; useful for tuning.
    skipped_path = out_path.with_name(out_path.stem + "_skipped.csv")
    skipped_rows = [asdict(c) for c in candidates if c.reasons_skipped]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        else:
            f.write("# No candidates passed the filters.\n")
    with open(skipped_path, "w", newline="", encoding="utf-8") as f:
        if skipped_rows:
            w = csv.DictWriter(f, fieldnames=list(skipped_rows[0].keys()))
            w.writeheader()
            w.writerows(skipped_rows)
    log.info(f"Wrote {len(rows)} candidates → {out_path}")
    log.info(f"Wrote {len(skipped_rows)} skipped → {skipped_path}")


def _load_tickers(source: Path) -> list[str]:
    """Read a ticker list from a CSV (with header 'Ticker' or 'ticker') or
    a plain newline-delimited file."""
    raw = source.read_text(encoding="utf-8").splitlines()
    if not raw:
        return []
    head = raw[0].strip().lower()
    if head in {"ticker"}:
        return [r.strip().upper() for r in raw[1:] if r.strip()]
    return [r.strip().upper() for r in raw if r.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--list", type=Path, default=DEFAULT_LIST,
                        help="Ticker list source (CSV with Ticker header or newline-delimited). Ignored when --universe is set.")
    parser.add_argument("--universe", action="store_true",
                        help="Screen the full NYSE + NASDAQ common-stock universe instead of --list.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=0,
                        help="Cap how many tickers to screen (0 = no cap)")
    parser.add_argument("--workers", type=int, default=15)
    args = parser.parse_args()

    if args.universe:
        from microcap_universe import fetch_universe
        tickers = fetch_universe()
    else:
        if not args.list.exists():
            print(f"ERROR: list file not found: {args.list}", file=sys.stderr)
            return 1
        tickers = _load_tickers(args.list)
    if args.limit:
        tickers = tickers[: args.limit]
    source_desc = "the NYSE + NASDAQ universe" if args.universe else str(args.list)
    log.info(f"Screening {len(tickers)} tickers from {source_desc}")

    candidates = screen_many(tickers, max_workers=args.workers)
    write_csv(candidates, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

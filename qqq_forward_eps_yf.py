"""Bottom-up QQQ forward EPS from yfinance analyst estimates (consistent basis).

QQQ has no free authoritative index-level consensus (FactSet covers only the
S&P 500), so we aggregate constituent analyst estimates ourselves and snapshot
the result into Index_Forward_EPS_History — the QQQ counterpart of
factset_sp500_eps.snapshot_factset_spy.

Accuracy notes (2026-07-10):
  - Numerator and denominator use the SAME basis: yfinance earnings_estimate
    ``avg`` (this-FY consensus) over ``yearAgoEps`` (the year-ago figure on the
    same adjusted basis). Mixing adjusted forward estimates with GAAP trailing
    EPS (TTM_Data) badly overstated per-name growth (e.g. AMGN +58% vs the
    true +2.3%) — never reintroduce that mix.
  - Cross-validated against independent market-implied growth: Siblis Research
    (Jul 1 2026) trailing P/E 35.24 / forward P/E 25.17 -> +40.0%, and their
    indexed EPS 216.28/154.48 -> +40.0%; this module's consistent-basis
    aggregate gave +37.8% on ~87% of index weight. Within a few points —
    the bottom-up is honest, not inflated.
  - yfinance ``info['forwardEps']`` is UNRELIABLE (inconsistent with the same
    ticker's forwardPE); only ``earnings_estimate`` rows are used here.
"""
from __future__ import annotations

import logging
import math
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)

_TARGET_PCT = 95.0     # take top-weight names to this cumulative weight; a few
                       # always come back empty, so aim above the 85% display gate
_MAX_WORKERS = 6


def _num(x) -> Optional[float]:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def fetch_constituent_estimate(ticker: str, info: Optional[dict] = None) -> Optional[dict]:
    """One constituent's consistent-basis estimate set from yfinance:
    {this_fy, year_ago, next_fy, shares} — or None if anything essential is
    missing. ``info`` may be a prefetched .info dict (avoids a slow call)."""
    import yfinance as yf
    try:
        tk = yf.Ticker(ticker)
        ee = tk.earnings_estimate
    except Exception as e:
        logger.warning("[%s] earnings_estimate failed: %s", ticker, e)
        return None
    if ee is None or getattr(ee, "empty", True) or "0y" not in ee.index:
        return None
    row0 = ee.loc["0y"]
    this_fy = _num(row0.get("avg"))
    year_ago = _num(row0.get("yearAgoEps"))
    next_fy = _num(ee.loc["+1y"].get("avg")) if "+1y" in ee.index else None

    shares = _num(info.get("sharesOutstanding")) if info else None
    if not shares:
        try:
            shares = _num(tk.info.get("sharesOutstanding"))
        except Exception:
            shares = None
    if this_fy is None or year_ago is None or not shares or shares <= 0:
        return None
    return {"this_fy": this_fy, "year_ago": year_ago,
            "next_fy": next_fy, "shares": shares}


def _cached_info(ticker: str) -> Optional[dict]:
    """Reuse the per-run prefetched .info from main_remote when available."""
    try:
        from forecasted_earnings_chart import get_cached_yf_info
        return get_cached_yf_info(ticker)
    except Exception:
        return None


def compute_qqq_bottom_up(holdings: List[Tuple[str, float]],
                          fetch: Callable = fetch_constituent_estimate,
                          target_pct: float = _TARGET_PCT) -> dict:
    """Aggregate constituent estimates into index-level growth.

    growth_this_fy = Σ(thisFY_eps × shares) / Σ(yearAgo_eps × shares) − 1, over
    the top-weight universe (to *target_pct* cumulative weight). growth_next_fy
    is computed the same way restricted to names that carry a next-FY estimate.
    coverage_weight is relative to the FULL index."""
    total_w = sum(w for _, w in holdings) or 1.0
    universe: List[Tuple[str, float]] = []
    cum = 0.0
    for tk, w in sorted(holdings, key=lambda x: x[1], reverse=True):
        universe.append((tk, w))
        cum += w
        if cum / total_w * 100.0 >= target_pct:
            break

    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
        fetched = list(ex.map(lambda tw: fetch(tw[0], info=_cached_info(tw[0])),
                              universe))

    num_this = den = 0.0
    nxt_num = nxt_den = 0.0
    cov_w = 0.0
    for (tk, w), d in zip(universe, fetched):
        if not d:
            continue
        num_this += d["this_fy"] * d["shares"]
        den += d["year_ago"] * d["shares"]
        cov_w += w
        if d.get("next_fy") is not None:
            nxt_num += d["next_fy"] * d["shares"]
            nxt_den += d["this_fy"] * d["shares"]

    if den <= 0 or cov_w <= 0:
        return {"growth_this_fy": None, "growth_next_fy": None,
                "coverage_weight": cov_w / total_w}
    return {
        "growth_this_fy": num_this / den - 1.0,
        "growth_next_fy": (nxt_num / nxt_den - 1.0) if nxt_den > 0 else None,
        "coverage_weight": cov_w / total_w,
    }


def snapshot_qqq_yf(conn: sqlite3.Connection, today: date = None,
                    result: dict = None) -> int:
    """Compute (or take precomputed) QQQ bottom-up growth and upsert its forecast
    row, scaled onto the chart's index-level EPS via the latest historical EPS —
    same pattern as the legacy snapshot. Row is written even when the display
    gate withholds it (displayable=0) so the history is auditable. Returns 1 if
    a row was written, 0 otherwise. Never raises for data problems."""
    import index_forward_eps as ife
    import forward_eps_validate as v
    ife.ensure_forward_eps_table(conn)
    today = today or date.today()

    if result is None:
        try:
            import index_holdings as ih
            holdings = ih.fetch_holdings("QQQ")
        except Exception as e:
            logger.warning("[qqq-yf] holdings fetch failed: %s", e)
            return 0
        result = compute_qqq_bottom_up(holdings)

    g1 = result.get("growth_this_fy")
    if g1 is None:
        logger.warning("[qqq-yf] no usable bottom-up growth; skip")
        return 0
    cov = result.get("coverage_weight")
    latest = ife._latest_hist_eps(conn, "QQQ")
    if latest is None or latest <= 0:
        logger.warning("[qqq-yf] no historical index EPS to scale onto; skip")
        return 0

    displayable = bool(v.is_displayable("QQQ", g1, cov))
    conn.execute(
        """INSERT OR REPLACE INTO Index_Forward_EPS_History
             (date_recorded, ticker, forward_eps_etf, forward_eps_index, forward_pe,
              horizon_date, source, coverage_weight, growth_this_fy, growth_next_fy,
              method, displayable)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (today.isoformat(), "QQQ", None, latest * (1.0 + g1), None,
         f"{today.year}-12-31", "yahoo_bottom_up", cov, g1,
         result.get("growth_next_fy"), "bottom_up_yf", int(displayable)))
    conn.commit()
    logger.info("[qqq-yf] QQQ growth=%+.1f%% (next %s) coverage=%.0f%% displayable=%s",
                g1 * 100,
                f"{result['growth_next_fy']*100:+.1f}%" if result.get("growth_next_fy") is not None else "n/a",
                (cov or 0) * 100, displayable)
    return 1

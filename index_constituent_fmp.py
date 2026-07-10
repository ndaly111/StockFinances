"""FMP-backed enrichment of index constituents for the bottom-up forward-EPS
aggregation.

The bottom-up index forward EPS (see forward_eps_bottom_up.aggregate) needs, per
constituent: forward EPS (This/Next FY), trailing EPS and shares outstanding. We
already have those for the ~100 site tickers, but only ~57% of SPY / ~72% of QQQ
by weight — below the 85% display gate. This module fills the gap for the
remaining high-weight constituents straight from FMP (already licensed) and writes
into the SAME two tables the aggregator reads, so no aggregation logic changes:

  - Forward_EPS_FY_History : /v3/analyst-estimates (estimatedEpsAvg, nearest 2 FYs)
  - TTM_Data               : /v3/income-statement?period=quarter (last 4 qtrs)

If the analyst-estimates endpoint is not on the account's FMP tier it returns an
error/empty; the ticker is skipped and coverage simply stays low (the display
gate then withholds the forecast — no regression, no bad data shown).
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import date, datetime, timezone
from typing import List, Optional

import requests

from config import get_fmp_api_key

logger = logging.getLogger(__name__)

_BASE = "https://financialmodelingprep.com/api/v3"
_TIMEOUT = 20


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _parse_date(raw) -> Optional[date]:
    if not raw:
        return None
    try:
        return datetime.strptime(str(raw)[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _get_json(url: str, params: dict, session=None):
    s = session or requests
    r = s.get(url, params=params, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


def fetch_forward_eps(ticker: str, api_key: str, today: date = None, session=None) -> Optional[dict]:
    """Nearest two *future* fiscal-year EPS estimates from FMP annual analyst
    estimates. Returns {this_fy, next_fy, this_end, next_end} or None on failure /
    no forward year available."""
    today = today or date.today()
    try:
        data = _get_json(f"{_BASE}/analyst-estimates/{ticker}",
                         {"period": "annual", "limit": 12, "apikey": api_key}, session)
    except Exception as e:                       # tier / network / parse
        logger.warning("[%s] analyst-estimates failed: %s", ticker, e)
        return None
    if not isinstance(data, list) or not data:
        return None
    rows = []
    for d in data:
        fy_end, eps = _parse_date(d.get("date")), d.get("estimatedEpsAvg")
        if fy_end is None or eps is None:
            continue
        try:
            rows.append((fy_end, float(eps)))
        except (TypeError, ValueError):
            continue
    future = sorted(r for r in rows if r[0] >= today)
    if not future:
        return None
    this_fy = future[0]
    next_fy = future[1] if len(future) >= 2 else None
    return {
        "this_fy": this_fy[1], "this_end": this_fy[0].isoformat(),
        "next_fy": next_fy[1] if next_fy else None,
        "next_end": next_fy[0].isoformat() if next_fy else None,
    }


def fetch_ttm_eps_shares(ticker: str, api_key: str, session=None) -> Optional[dict]:
    """Trailing-twelve-month net income (sum of last 4 quarters), shares outstanding
    (latest quarter diluted), and TTM EPS = TTM_net_income / shares so that
    eps * shares == net income exactly (the aggregator relies on that identity).
    Returns None if fewer than 4 quarters or missing fields."""
    try:
        data = _get_json(f"{_BASE}/income-statement/{ticker}",
                         {"period": "quarter", "limit": 4, "apikey": api_key}, session)
    except Exception as e:
        logger.warning("[%s] income-statement failed: %s", ticker, e)
        return None
    if not isinstance(data, list) or len(data) < 4:
        return None
    net_incomes, shares = [], None
    for i, q in enumerate(data[:4]):
        ni = q.get("netIncome")
        if ni is None:
            return None
        net_incomes.append(float(ni))
        if i == 0:
            shares = q.get("weightedAverageShsOutDil") or q.get("weightedAverageShsOut")
    if not shares:
        return None
    shares = float(shares)
    if shares <= 0:
        return None
    ttm_ni = sum(net_incomes)
    return {"ttm_net_income": ttm_ni, "shares": shares,
            "ttm_eps": ttm_ni / shares, "quarter": data[0].get("date")}


def _upsert_forward(conn, today_s, ticker, label, period_end, eps) -> None:
    conn.execute(
        """INSERT INTO Forward_EPS_FY_History
             (date_recorded, ticker, period_end, period_label, forward_eps, source)
           VALUES (?,?,?,?,?,?)
           ON CONFLICT(date_recorded, ticker, period_end) DO UPDATE SET
             period_label=excluded.period_label, forward_eps=excluded.forward_eps,
             source=excluded.source""",
        (today_s, ticker, period_end, label, eps, "fmp.analyst-estimates"))


def _upsert_ttm(conn, ticker, fin) -> None:
    conn.execute(
        """INSERT INTO TTM_Data
             (Symbol, TTM_Revenue, TTM_Net_Income, TTM_EPS, Shares_Outstanding,
              Quarter, Last_Updated)
           VALUES (?,?,?,?,?,?,?)
           ON CONFLICT(Symbol) DO UPDATE SET
             TTM_Net_Income=excluded.TTM_Net_Income, TTM_EPS=excluded.TTM_EPS,
             Shares_Outstanding=excluded.Shares_Outstanding, Quarter=excluded.Quarter,
             Last_Updated=excluded.Last_Updated""",
        (ticker, None, fin["ttm_net_income"], fin["ttm_eps"], fin["shares"],
         fin["quarter"], _now_utc()))


def enrich_ttm_shares_fmp(conn: sqlite3.Connection, tickers, session=None,
                          api_key: str = None) -> int:
    """Fill ONLY TTM EPS + shares (from FMP income-statement, a proven endpoint) into
    TTM_Data for the given constituents. Forward EPS for these names is sourced
    elsewhere — the Zacks scrape (scrape_forward_data_batch) — because FMP's
    analyst-estimates endpoint is not on this account's tier. Returns count written.

    Used for QQQ's high-weight uncovered constituents so the bottom-up aggregation
    (which needs TTM_EPS, shares, and This-FY forward EPS per name) can cover them."""
    tickers = list(tickers)
    if not tickers:
        return 0
    api_key = api_key or get_fmp_api_key()
    n_ok = 0
    for tk in tickers:
        fin = fetch_ttm_eps_shares(tk, api_key, session=session)
        if not fin:
            continue
        _upsert_ttm(conn, tk, fin)
        n_ok += 1
    conn.commit()
    logger.info("[fmp-ttm] wrote TTM_Data for %d/%d constituents", n_ok, len(tickers))
    return n_ok


def enrich_constituents_fmp(conn: sqlite3.Connection, tickers, today: date = None,
                            session=None, api_key: str = None) -> int:
    """Fetch forward EPS + TTM EPS/shares from FMP for each ticker and upsert into
    Forward_EPS_FY_History + TTM_Data. A ticker is written only if BOTH fetches
    succeed (no partial rows). Returns the number of fully-enriched tickers."""
    tickers = list(tickers)
    if not tickers:
        return 0
    api_key = api_key or get_fmp_api_key()
    today = today or date.today()
    today_s = today.isoformat()
    n_ok = 0
    for tk in tickers:
        fwd = fetch_forward_eps(tk, api_key, today=today, session=session)
        if not fwd or fwd.get("this_fy") is None:
            continue
        fin = fetch_ttm_eps_shares(tk, api_key, session=session)
        if not fin:
            continue
        _upsert_forward(conn, today_s, tk, "This FY", fwd["this_end"], fwd["this_fy"])
        if fwd.get("next_fy") is not None:
            _upsert_forward(conn, today_s, tk, "Next FY", fwd["next_end"], fwd["next_fy"])
        _upsert_ttm(conn, tk, fin)
        n_ok += 1
    conn.commit()
    logger.info("[fmp-enrich] enriched %d/%d constituents", n_ok, len(tickers))
    return n_ok

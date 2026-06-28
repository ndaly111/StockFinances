"""Consensus forward EPS for SPY/QQQ (index level), snapshotted daily.

Primary source: stockanalysis.com ETF pages (holdings-weighted forward P/E +
forward EPS). Fallback: yfinance ETF forwardPE/forwardEps. Values are scaled
from ETF level to index level so they line up with index_growth_charts EPS.
"""
from __future__ import annotations

import io
import logging
import re
import sqlite3
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import requests
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = "Stock Data.db"
TABLE = "Index_Forward_EPS_History"
IDXES = ["SPY", "QQQ"]

# Mirror of index_growth_charts._INDEX_EPS_DIVISOR. A drift-guard test in
# Test/test_index_forward_eps.py asserts these stay equal.
INDEX_EPS_DIVISOR = {"SPY": 10.0, "QQQ": 4.0}


def ensure_forward_eps_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE} (
            date_recorded     TEXT NOT NULL,
            ticker            TEXT NOT NULL,
            forward_eps_etf   REAL,
            forward_eps_index REAL,
            forward_pe        REAL,
            horizon_date      TEXT,
            source            TEXT,
            PRIMARY KEY (date_recorded, ticker)
        )
        """
    )
    conn.commit()


@dataclass
class ForwardEPS:
    ticker: str
    forward_pe: float
    forward_eps_etf: float
    forward_eps_index: float
    horizon_date: str
    source: str


def _default_horizon() -> str:
    """NTM estimates ~ 12 months out; used when the source gives no date."""
    return (date.today() + timedelta(days=365)).isoformat()


def _scale_index(tk: str, eps_etf: float) -> float:
    return eps_etf * INDEX_EPS_DIVISOR.get(tk.upper(), 1.0)


def _forward_from_yf(tk: str, info: dict) -> Optional[ForwardEPS]:
    pe = info.get("forwardPE")
    eps_etf = info.get("forwardEps")
    price = info.get("regularMarketPrice")
    try:
        pe = float(pe) if pe is not None else None
    except (TypeError, ValueError):
        pe = None
    if pe is None or pe <= 0:
        return None
    if eps_etf is None and price is not None:
        try:
            eps_etf = float(price) / pe
        except (TypeError, ValueError, ZeroDivisionError):
            eps_etf = None
    if eps_etf is None:
        return None
    eps_etf = float(eps_etf)
    return ForwardEPS(
        ticker=tk.upper(),
        forward_pe=pe,
        forward_eps_etf=eps_etf,
        forward_eps_index=_scale_index(tk, eps_etf),
        horizon_date=_default_horizon(),
        source="yfinance",
    )


# Sane growth band -50%..+80%. (A gross ETF/index scaling error shows up as
# absurd implied growth, so the growth band alone catches it.)
_GROWTH_MIN, _GROWTH_MAX = -0.50, 0.80


def _passes_sanity(forward_pe, forward_eps_index, latest_hist_eps) -> bool:
    try:
        forward_pe = float(forward_pe)
        forward_eps_index = float(forward_eps_index)
    except (TypeError, ValueError):
        return False
    if forward_pe <= 0 or forward_eps_index <= 0:
        return False
    if latest_hist_eps is None:
        return True
    try:
        latest = float(latest_hist_eps)
    except (TypeError, ValueError):
        return True
    if latest <= 0:
        return True
    growth = forward_eps_index / latest - 1.0
    if not (_GROWTH_MIN <= growth <= _GROWTH_MAX):
        return False
    return True


# ---------------------------------------------------------------------------
# stockanalysis.com scraper
# ---------------------------------------------------------------------------

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}
_FWD_PE_RE = re.compile(r"forward.*p/?e|p/?e.*forward", re.IGNORECASE)


def _parse_forward_pe(html: str) -> Optional[float]:
    """Find a 'Forward PE' label/value in any table on the page."""
    try:
        tables = pd.read_html(io.StringIO(html))
    except (ValueError, Exception):
        return None
    for tbl in tables:
        if tbl.shape[1] < 2:
            continue
        labels = tbl.iloc[:, 0].astype(str)
        mask = labels.str.contains(_FWD_PE_RE, na=False)
        if not mask.any():
            continue
        raw = str(tbl.loc[mask].iloc[0, 1])
        m = re.search(r"-?\d+(?:\.\d+)?", raw.replace(",", ""))
        if m:
            try:
                return float(m.group())
            except ValueError:
                return None
    return None


def _fetch_stockanalysis_pe(tk: str, session: requests.Session) -> Optional[float]:
    url = f"https://stockanalysis.com/etf/{tk.lower()}/"
    try:
        r = session.get(url, headers=_HEADERS, timeout=20)
        r.raise_for_status()
    except requests.RequestException as e:
        logger.warning("[%s] stockanalysis fetch failed: %s", tk, e)
        return None
    pe = _parse_forward_pe(r.text)
    if pe is None:
        logger.warning("[%s] stockanalysis: forward P/E not found (layout drift?)", tk)
    return pe


# ---------------------------------------------------------------------------
# yfinance wrapper + orchestration
# ---------------------------------------------------------------------------


def _yf_info(tk: str) -> dict:
    try:
        info = yf.Ticker(tk).info
        return info if isinstance(info, dict) else {}
    except Exception as e:        # noqa: BLE001 - yfinance raises many types
        logger.warning("[%s] yfinance info failed: %s", tk, e)
        return {}


def fetch_forward_eps(tk: str, session: requests.Session, latest_hist_eps: Optional[float]) -> Optional[ForwardEPS]:
    """Primary stockanalysis.com -> yfinance fallback -> sanity guard."""
    tk = tk.upper()
    info = _yf_info(tk)
    price = info.get("regularMarketPrice")

    # Primary: stockanalysis forward P/E + price for the EPS dollar value.
    pe = _fetch_stockanalysis_pe(tk, session)
    if pe is not None and pe > 0 and price:
        eps_etf = float(price) / pe
        cand = ForwardEPS(
            ticker=tk, forward_pe=pe, forward_eps_etf=eps_etf,
            forward_eps_index=_scale_index(tk, eps_etf),
            horizon_date=_default_horizon(), source="stockanalysis",
        )
        if _passes_sanity(cand.forward_pe, cand.forward_eps_index, latest_hist_eps):
            return cand
        logger.warning("[%s] stockanalysis value failed sanity; trying yfinance", tk)

    # Fallback: yfinance.
    cand = _forward_from_yf(tk, info)
    if cand and _passes_sanity(cand.forward_pe, cand.forward_eps_index, latest_hist_eps):
        return cand
    logger.warning("[%s] no usable forward EPS (primary+fallback failed/rejected)", tk)
    return None

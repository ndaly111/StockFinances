"""Consensus forward EPS for SPY/QQQ (index level), snapshotted daily.

Primary source: stockanalysis.com ETF pages (holdings-weighted forward P/E +
forward EPS). Fallback: yfinance ETF forwardPE/forwardEps. Values are scaled
from ETF level to index level so they line up with index_growth_charts EPS.
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

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
    if eps_etf is None and price:
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

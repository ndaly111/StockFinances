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

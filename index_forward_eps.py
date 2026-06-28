"""Consensus forward EPS for SPY/QQQ (index level), snapshotted daily.

Primary source: bottom-up constituent forward EPS aggregation.
Values are scaled from ETF level to index level so they line up with
index_growth_charts EPS.
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


def _add_columns(conn: sqlite3.Connection, table: str, cols: list) -> None:
    existing = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
    for name, decl in cols:
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")


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
    _add_columns(conn, TABLE, [
        ("coverage_weight", "REAL"), ("growth_this_fy", "REAL"),
        ("growth_next_fy", "REAL"), ("method", "TEXT"), ("displayable", "INTEGER")])
    conn.commit()


def ensure_constituents_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS index_constituents (
             date_recorded TEXT NOT NULL, index_name TEXT NOT NULL,
             ticker TEXT NOT NULL, weight REAL,
             PRIMARY KEY (date_recorded, index_name, ticker))""")
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


def _latest_hist_eps(conn: sqlite3.Connection, tk: str) -> Optional[float]:
    """Latest index-level historical EPS for scale/growth sanity checks.

    Mirrors index_growth_charts._series_eps source priority loosely: prefer
    TTM_REPORTED, else TTM_DAILY, else IMPLIED_FROM_PE*divisor.
    """
    try:
        row = conn.execute(
            """SELECT EPS, EPS_Type FROM Index_EPS_History
                WHERE Ticker=? AND EPS_Type IN
                      ('TTM_REPORTED','TTM_DAILY','IMPLIED_FROM_PE')
             ORDER BY CASE EPS_Type
                        WHEN 'TTM_REPORTED' THEN 0
                        WHEN 'TTM_DAILY'    THEN 1
                        ELSE 2 END, Date DESC
                LIMIT 1""",
            (tk,),
        ).fetchone()
    except sqlite3.Error:
        return None
    if not row or row[0] is None:
        return None
    eps, eps_type = float(row[0]), row[1]
    if eps_type == "IMPLIED_FROM_PE":
        eps *= INDEX_EPS_DIVISOR.get(tk.upper(), 1.0)
    return eps


def snapshot_forward_eps(conn: sqlite3.Connection, today=None) -> int:
    """Bottom-up aggregate constituent EPS → upsert today's forward EPS for each index.

    Returns number of rows written.
    """
    import index_holdings as ih
    import forward_eps_bottom_up as bu
    import forward_eps_validate as v
    ensure_forward_eps_table(conn)
    ensure_constituents_table(conn)
    today = today or date.today().isoformat()
    written = 0
    for idx in IDXES:                      # ["SPY","QQQ"]
        try:
            holdings = ih.fetch_holdings(idx)
        except Exception as e:
            logger.warning("[%s] holdings fetch failed: %s", idx, e); continue
        ih.persist_holdings(conn, idx, holdings, today=today)
        fin = bu.load_constituent_financials(conn, [tk for tk, _ in holdings])
        add = ih.uncovered_for_target(holdings, set(fin), target_pct=90.0)
        if add:
            logger.info("[%s] auto-extend scrape: %d names (%s...)", idx, len(add), add[:5])
        res = bu.aggregate(holdings, fin)
        if res["growth_this_fy"] is None:
            logger.warning("[%s] no usable bottom-up growth; skip", idx); continue
        latest_hist = _latest_hist_eps(conn, idx)
        # Forward index EPS point for the chart: scale the latest historical index EPS
        # by the computed this-FY growth so it sits on the same level as _series_eps.
        fwd_eps_index = (latest_hist * (1 + res["growth_this_fy"])) if latest_hist else None
        displayable = bool(v.is_displayable(idx, res["growth_this_fy"], res["coverage_weight"]))
        if fwd_eps_index is not None and not _passes_sanity(20.0, fwd_eps_index, latest_hist):
            displayable = False
        conn.execute(
            f"""INSERT OR REPLACE INTO {TABLE}
                (date_recorded, ticker, forward_eps_etf, forward_eps_index, forward_pe,
                 horizon_date, source, coverage_weight, growth_this_fy, growth_next_fy,
                 method, displayable)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (today, idx, None, fwd_eps_index, None, _default_horizon(), "bottom_up",
             res["coverage_weight"], res["growth_this_fy"], res["growth_next_fy"],
             "bottom_up", int(displayable)))
        written += 1
    conn.commit()
    logger.info("[forward-eps] snapshot wrote %d row(s) for %s", written, today)
    return written

"""Split adjustment helpers.

This module fetches split history from Yahoo Finance (and optionally FMP),
records it in the ``Splits`` table, and applies cumulative adjustments to
stored per-share metrics so downstream consumers (charts, valuations, etc.)
operate on split-adjusted data.
"""

from __future__ import annotations

import logging
import math
import os
import sqlite3
from datetime import datetime, date
from functools import lru_cache
from typing import Iterable, List, Tuple

import requests
import yfinance as yf

SPLIT_SOURCE_YF = "yfinance"
SPLIT_SOURCE_FMP = "fmp"


def ensure_splits_table(cur: sqlite3.Cursor) -> None:
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS Splits(
            Symbol TEXT,
            Date TEXT,
            Ratio REAL,
            Source TEXT,
            Last_Checked TEXT,
            PRIMARY KEY(Symbol, Date)
        );
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_splits_symbol_date ON Splits(Symbol, Date)")


def _coerce_ratio(raw_ratio: float | int | str | None) -> float | None:
    try:
        ratio = float(raw_ratio)
    except Exception:
        return None
    return ratio if ratio > 0 else None


def _from_yfinance(ticker: str) -> List[Tuple[date, float, str]]:
    tk = yf.Ticker(ticker)
    splits = tk.splits
    events: List[Tuple[date, float, str]] = []
    if splits is None or splits.empty:
        return events

    for idx, ratio in splits.items():
        coerced = _coerce_ratio(ratio)
        if coerced:
            events.append((idx.date(), coerced, SPLIT_SOURCE_YF))
    return events


def _from_fmp(ticker: str) -> List[Tuple[date, float, str]]:
    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        return []

    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/stock_split/{ticker}"
    try:
        resp = requests.get(url, params={"apikey": api_key}, timeout=15)
        resp.raise_for_status()
        payload = resp.json()
        rows: Iterable = payload.get("historical", []) if isinstance(payload, dict) else payload
    except Exception as exc:  # network errors should not abort the caller
        logging.warning("[%s] Unable to fetch FMP splits: %s", ticker, exc)
        return []

    events: List[Tuple[date, float, str]] = []
    for row in rows:
        try:
            event_date = datetime.fromisoformat(str(row.get("date"))).date()
        except Exception:
            continue

        numerator = _coerce_ratio(row.get("numerator"))
        denominator = _coerce_ratio(row.get("denominator"))
        if numerator and denominator:
            ratio = numerator / denominator
        else:
            ratio = _coerce_ratio(row.get("ratio"))

        if ratio:
            events.append((event_date, ratio, SPLIT_SOURCE_FMP))
    return events


@lru_cache(maxsize=64)
def fetch_split_history(ticker: str) -> List[Tuple[date, float, str]]:
    """Fetch split history from available providers and cache per ticker."""
    seen: dict[date, Tuple[float, str]] = {}
    for source_events in (_from_yfinance(ticker), _from_fmp(ticker)):
        for dt, ratio, source in source_events:
            if dt not in seen:
                seen[dt] = (ratio, source)
    return sorted([(dt, ratio, source) for dt, (ratio, source) in seen.items()], key=lambda r: r[0])


def _latest_financial_date(ticker: str, cur: sqlite3.Cursor) -> date | None:
    cur.execute("SELECT MAX(Date) FROM Annual_Data WHERE Symbol=?", (ticker,))
    annual_max = cur.fetchone()[0]

    cur.execute("SELECT Quarter FROM TTM_Data WHERE Symbol=?", (ticker,))
    ttm_row = cur.fetchone()
    ttm_date = ttm_row[0] if ttm_row else None

    candidates: List[date] = []
    for raw in (annual_max, ttm_date):
        try:
            candidates.append(datetime.fromisoformat(str(raw)).date())
        except Exception:
            continue

    if not candidates:
        return None
    return max(candidates)


def load_eps_series(
    ticker: str, cur: sqlite3.Cursor, min_years: int, include_ttm: bool = True
) -> List[Tuple[date, float, str]]:
    """
    Load a time-ordered EPS series from annual and optional TTM/quarterly data.

    Returns a list of tuples: (period_end_date, eps, label) where ``label`` is
    ``"annual"`` for fiscal-year figures and ``"ttm"`` for TTM/quarterly data.
    """

    def _coerce_date(raw: object) -> date | None:
        try:
            return datetime.fromisoformat(str(raw)).date()
        except Exception:
            return None

    def _coerce_eps(raw: object) -> float | None:
        try:
            value = float(raw)
        except Exception:
            return None
        if value == 0 or math.isnan(value):
            return None
        return value

    annual_rows: List[Tuple[date, float, str]] = []
    cur.execute("SELECT Date, EPS FROM Annual_Data WHERE Symbol=? ORDER BY Date DESC", (ticker,))
    for raw_date, raw_eps in cur.fetchall():
        eps_val = _coerce_eps(raw_eps)
        dt_val = _coerce_date(raw_date)
        if eps_val is None or dt_val is None:
            continue
        annual_rows.append((dt_val, eps_val, "annual"))
        if len(annual_rows) >= min_years:
            break

    if not annual_rows:
        logging.info("[%s] No annual EPS rows available for last %d years", ticker, min_years)
    elif len(annual_rows) < min_years:
        logging.info(
            "[%s] Only %d annual EPS rows found (requested %d)", ticker, len(annual_rows), min_years
        )

    ttm_rows: List[Tuple[date, float, str]] = []
    if include_ttm:
        cur.execute("SELECT TTM_EPS, Quarter FROM TTM_Data WHERE Symbol=?", (ticker,))
        ttm_row = cur.fetchone()
        if not ttm_row:
            logging.info("[%s] No TTM row present when loading EPS series", ticker)
        else:
            eps_val = _coerce_eps(ttm_row[0])
            dt_val = _coerce_date(ttm_row[1])
            if eps_val is None or dt_val is None:
                logging.info("[%s] TTM row present but missing EPS or date", ticker)
            else:
                ttm_rows.append((dt_val, eps_val, "ttm"))

    combined = sorted(annual_rows + ttm_rows, key=lambda r: r[0])
    if combined:
        logging.debug("[%s] Loaded %d EPS data points", ticker, len(combined))
    else:
        logging.info("[%s] No EPS data points available", ticker)
    return combined


def infer_split_candidates(
    series: List[Tuple[date, float, str]], tolerance: float = 0.03
) -> List[Tuple[date, float, Tuple[str, str]]]:
    """Detect likely split ratios by examining adjacent EPS values.

    The input ``series`` should contain (date, eps, label) tuples. Adjacent
    entries are compared in chronological order, considering only pairs where
    both EPS values are non-zero and share the same sign. When the absolute
    ratio between neighboring EPS values approximates a common split factor,
    a candidate is emitted with the later date in the pair.
    """

    if tolerance <= 0:
        raise ValueError("tolerance must be positive")

    common_ratios: List[float] = [2, 3, 4, 5, 10, 20, 25, 50]
    candidates: List[Tuple[date, float, Tuple[str, str]]] = []

    sorted_series = sorted(series, key=lambda r: r[0])
    for prev, nxt in zip(sorted_series, sorted_series[1:]):
        prev_date, prev_eps, prev_label = prev
        next_date, next_eps, next_label = nxt

        if prev_eps == 0 or next_eps == 0:
            continue

        if prev_eps * next_eps < 0:
            continue

        ratio = abs(prev_eps) / abs(next_eps)

        if abs(ratio - 1.0) <= tolerance:
            continue

        for common_ratio in common_ratios:
            if abs(ratio - common_ratio) <= tolerance:
                candidates.append((next_date, common_ratio, (prev_label, next_label)))
                break

    return candidates


def merge_candidate_events(
    candidates: List[Tuple[date, float, Tuple[str, str]]],
    merge_window_days: int = 180,
    tolerance: float = 0.03,
) -> List[Tuple[date, float, List[Tuple[str, str]]]]:
    """
    Merge nearby candidate split events that represent the same ratio.

    Adjacent candidate records with similar ratios (within ``tolerance``) whose
    boundary dates fall within ``merge_window_days`` are combined into a single
    event. The merged event's date is anchored to the earliest boundary after
    the jump, and all contributing period label pairs are retained for
    traceability.
    """

    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if merge_window_days < 0:
        raise ValueError("merge_window_days cannot be negative")

    if not candidates:
        return []

    def _ratios_match(a: float, b: float) -> bool:
        return abs(a - b) <= tolerance

    sorted_candidates = sorted(candidates, key=lambda r: r[0])
    merged: List[Tuple[date, float, List[Tuple[str, str]]]] = []

    anchor_date, current_ratio, first_periods = sorted_candidates[0]
    last_boundary = anchor_date
    period_pairs: List[Tuple[str, str]] = [first_periods]

    for cand_date, cand_ratio, cand_periods in sorted_candidates[1:]:
        within_window = (cand_date - last_boundary).days <= merge_window_days
        if within_window and _ratios_match(cand_ratio, current_ratio):
            if cand_periods not in period_pairs:
                period_pairs.append(cand_periods)
            last_boundary = cand_date
            continue

        merged.append((anchor_date, current_ratio, period_pairs))
        anchor_date, current_ratio = cand_date, cand_ratio
        last_boundary = cand_date
        period_pairs = [cand_periods]

    merged.append((anchor_date, current_ratio, period_pairs))
    return merged


def apply_split_adjustments(ticker: str, cur: sqlite3.Cursor) -> bool:
    """
    Detect new splits and adjust stored per-share values.

    Returns True when adjustments were applied.
    """

    ensure_splits_table(cur)
    latest_data_date = _latest_financial_date(ticker, cur)
    if not latest_data_date:
        return False

    split_history = fetch_split_history(ticker)
    if not split_history:
        return False

    cur.execute("SELECT Date FROM Splits WHERE Symbol=?", (ticker,))
    recorded_dates = {row[0] for row in cur.fetchall()}

    now_iso = datetime.utcnow().isoformat(sep=" ", timespec="seconds")
    # Upsert all known events with refreshed Last_Checked
    for dt, ratio, source in split_history:
        cur.execute(
            """
            INSERT INTO Splits(Symbol, Date, Ratio, Source, Last_Checked)
            VALUES(?,?,?,?,?)
            ON CONFLICT(Symbol, Date) DO UPDATE SET
                Ratio=excluded.Ratio,
                Source=excluded.Source,
                Last_Checked=excluded.Last_Checked;
            """,
            (ticker, dt.isoformat(), ratio, source, now_iso),
        )

    pending_events = [
        (dt, ratio, source)
        for dt, ratio, source in split_history
        if dt.isoformat() not in recorded_dates and dt > latest_data_date
    ]

    if not pending_events:
        cur.connection.commit()
        return False

    pending_events.sort(key=lambda r: r[0])
    cumulative_ratio = 1.0
    for _, ratio, _ in pending_events:
        if ratio:
            cumulative_ratio *= ratio

    earliest_split_date = pending_events[0][0].isoformat()

    cur.execute(
        """
        UPDATE Annual_Data
           SET EPS = EPS / ?,
               Last_Updated = CURRENT_TIMESTAMP
         WHERE Symbol=? AND Date < ?;
        """,
        (cumulative_ratio, ticker, earliest_split_date),
    )

    cur.execute(
        "SELECT TTM_EPS, Shares_Outstanding, Quarter, Last_Updated FROM TTM_Data WHERE Symbol=?",
        (ticker,),
    )
    ttm_row = cur.fetchone()
    if ttm_row and ttm_row[2]:
        try:
            ttm_quarter = datetime.fromisoformat(str(ttm_row[2])).date()
        except Exception:
            ttm_quarter = None
        try:
            ttm_last_updated = datetime.fromisoformat(str(ttm_row[3])).date() if ttm_row[3] else None
        except Exception:
            ttm_last_updated = None

        should_adjust_ttm = (
            ttm_quarter is not None
            and ttm_quarter < pending_events[0][0]
            and (ttm_last_updated is None or ttm_last_updated < pending_events[0][0])
        )

        if should_adjust_ttm:
            new_eps = ttm_row[0] / cumulative_ratio if ttm_row[0] is not None else None
            new_shares = ttm_row[1] * cumulative_ratio if ttm_row[1] is not None else None
            cur.execute(
                """
                UPDATE TTM_Data
                   SET TTM_EPS = ?,
                       Shares_Outstanding = ?,
                       Last_Updated = CURRENT_TIMESTAMP
                 WHERE Symbol=?;
                """,
                (new_eps, new_shares, ticker),
            )

    cur.connection.commit()
    logging.info("[%s] Applied cumulative split ratio %.4f", ticker, cumulative_ratio)
    return True

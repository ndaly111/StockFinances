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
from collections import defaultdict
from datetime import datetime, date
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Tuple

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


def reconcile_split_events(
    ticker: str,
    inferred_events: List[Tuple[date, float, List[Tuple[str, str]]]],
    tolerance: float = 0.03,
    date_window_days: int = 30,
) -> List[dict]:
    """
    Align inferred split events with provider history.

    Each inferred event is reconciled against provider-sourced history from
    :func:`fetch_split_history`. When a provider event falls within the
    ``date_window_days`` window of an inferred event and the ratios match within
    ``tolerance``, the provider's date and ratio are preferred and the event is
    marked ``provider_match``. Otherwise, the inferred event is retained and
    marked ``inferred_only``.
    """

    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if date_window_days < 0:
        raise ValueError("date_window_days cannot be negative")

    def _ratio_bucket(val: float) -> int:
        return int(round(val / tolerance))

    provider_events = fetch_split_history(ticker)
    provider_index: dict[int, List[Tuple[date, float, str]]] = defaultdict(list)
    for provider_event in provider_events:
        provider_index[_ratio_bucket(provider_event[1])].append(provider_event)

    def _ratios_match(candidate_ratio: float, provider_ratio: float) -> bool:
        return abs(candidate_ratio - provider_ratio) <= tolerance

    def _find_best_provider_match(
        inferred_date: date, inferred_ratio: float
    ) -> Tuple[date, float, str] | None:
        bucket = _ratio_bucket(inferred_ratio)
        best_match: Tuple[date, float, str] | None = None
        best_distance: int | None = None

        for bucket_key in (bucket - 1, bucket, bucket + 1):
            for provider_date, provider_ratio, provider_source in provider_index.get(
                bucket_key, []
            ):
                if not _ratios_match(inferred_ratio, provider_ratio):
                    continue
                day_distance = abs((provider_date - inferred_date).days)
                if day_distance > date_window_days:
                    continue
                if best_distance is None or day_distance < best_distance:
                    best_match = (provider_date, provider_ratio, provider_source)
                    best_distance = day_distance
        return best_match

    reconciled: List[dict] = []
    for inferred_date, inferred_ratio, inferred_periods in inferred_events:
        provider_match = _find_best_provider_match(inferred_date, inferred_ratio)

        if provider_match:
            provider_date, provider_ratio, provider_source = provider_match
            reconciled.append(
                {
                    "date": provider_date,
                    "ratio": provider_ratio,
                    "source": provider_source,
                    "status": "provider_match",
                    "inferred_date": inferred_date,
                    "inferred_ratio": inferred_ratio,
                    "inferred_periods": inferred_periods,
                    "provider_date": provider_date,
                    "provider_ratio": provider_ratio,
                    "provider_source": provider_source,
                }
            )
        else:
            reconciled.append(
                {
                    "date": inferred_date,
                    "ratio": inferred_ratio,
                    "source": "inferred",
                    "status": "inferred_only",
                    "inferred_date": inferred_date,
                    "inferred_ratio": inferred_ratio,
                    "inferred_periods": inferred_periods,
                    "provider_date": None,
                    "provider_ratio": None,
                    "provider_source": None,
                }
            )

    return sorted(reconciled, key=lambda r: r["date"])


def plan_split_adjustments(
    ticker: str,
    merged_events: List[Dict[str, object]],
    status_map: Dict[str, object],
) -> List[Dict[str, object]]:
    """
    Build a human-readable split-adjustment plan.

    ``merged_events`` should contain dictionaries with at least ``date``,
    ``ratio``, ``source``, and ``status`` keys (such as the output of
    :func:`reconcile_split_events`). ``status_map`` can carry:

    - ``annual_periods`` (or ``annual_dates``/``annual_period_ends``): iterable
      of fiscal year-end dates.
    - ``ttm_period`` (or ``ttm_date``): the latest TTM/quarterly period date.
    - ``event_status`` (or ``statuses``): optional overrides keyed by ISO date
      string.
    """

    status_map = status_map or {}

    def _coerce_date(raw: object) -> date | None:
        if isinstance(raw, date):
            return raw
        try:
            return datetime.fromisoformat(str(raw)).date()
        except Exception:
            return None

    def _coerce_ratio(raw: object) -> float | None:
        try:
            return float(raw)
        except Exception:
            return None

    annual_periods: List[date] = []
    for key in ("annual_periods", "annual_dates", "annual_period_ends"):
        for raw in status_map.get(key, []) or []:
            coerced = _coerce_date(raw)
            if coerced:
                annual_periods.append(coerced)
    annual_periods = sorted(set(annual_periods))

    ttm_period = _coerce_date(status_map.get("ttm_period") or status_map.get("ttm_date"))
    status_overrides: Dict[str, str] = status_map.get("event_status") or status_map.get("statuses") or {}

    def _first_post_split_period(event_date: date) -> date:
        candidates: List[date] = [period for period in annual_periods if period > event_date]
        if ttm_period and ttm_period > event_date:
            candidates.append(ttm_period)
        if candidates:
            return min(candidates)
        return event_date

    plan: List[Dict[str, object]] = []
    sorted_events = sorted(
        merged_events,
        key=lambda evt: _coerce_date(evt.get("date")) or date.min,
    )

    for event in sorted_events:
        event_date = _coerce_date(event.get("date"))
        if not event_date:
            logging.warning("[%s] Skipping split event without a valid date: %s", ticker, event)
            continue

        ratio_value = _coerce_ratio(event.get("ratio"))
        apply_before = _first_post_split_period(event_date)
        affected_years = sorted({period.year for period in annual_periods if period < apply_before})

        iso_date = event_date.isoformat()
        status_value = status_overrides.get(iso_date) or event.get("status") or "pending"
        latest_year = affected_years[-1] if affected_years else "n/a"
        logging.info(
            "[%s] apply ratio %s to rows before %s (affecting fiscal years ≤ %s)",
            ticker,
            ratio_value,
            apply_before.isoformat(),
            latest_year,
        )

        plan.append(
            {
                "date": iso_date,
                "ratio": ratio_value,
                "apply_before": apply_before.isoformat(),
                "affected_years": affected_years,
                "source": event.get("source") or "unknown",
                "status": status_value,
            }
        )

    return plan


def assess_split_adjustment_status(
    ticker: str,
    cur: sqlite3.Cursor,
    *,
    tolerance: float = 0.03,
    merge_window_days: int = 180,
    date_window_days: int = 45,
    min_years: int = 10,
) -> Tuple[str, List[Dict[str, object]]]:
    """
    Determine whether stored EPS figures are split-adjusted for a ticker.

    Returns a tuple of ``(status, evidence)`` where ``status`` is one of
    ``"adjusted"``, ``"unadjusted"``, or ``"mismatch/inconclusive"`` and
    ``evidence`` lists the neighboring EPS ratios inspected around each
    provider or inferred split date for auditability.
    """

    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if merge_window_days < 0:
        raise ValueError("merge_window_days cannot be negative")
    if date_window_days < 0:
        raise ValueError("date_window_days cannot be negative")

    eps_series = load_eps_series(ticker, cur, min_years=min_years)
    ordered_series = sorted(eps_series, key=lambda r: r[0])

    if len(ordered_series) < 2:
        return "mismatch/inconclusive", [
            {
                "issue": "insufficient_eps_points",
                "points_found": len(ordered_series),
                "note": "At least two EPS observations are required to compute ratios.",
            }
        ]

    def _neighbor_ratio(anchor: date) -> Tuple[Optional[float], Optional[Tuple[date, float, str]], Optional[Tuple[date, float, str]]]:
        before = None
        after = None
        for dt_val, eps_val, label in ordered_series:
            if dt_val <= anchor:
                before = (dt_val, eps_val, label)
            if dt_val > anchor:
                after = (dt_val, eps_val, label)
                break

        if not before or not after:
            return None, before, after

        prev_eps = before[1]
        next_eps = after[1]
        if prev_eps == 0 or next_eps == 0 or prev_eps * next_eps < 0:
            return None, before, after

        return abs(prev_eps) / abs(next_eps), before, after

    def _is_ratio_adjusted(ratio_value: Optional[float]) -> bool:
        return ratio_value is not None and abs(ratio_value - 1.0) <= tolerance

    inferred_events = merge_candidate_events(
        infer_split_candidates(ordered_series, tolerance=tolerance),
        merge_window_days=merge_window_days,
        tolerance=tolerance,
    )
    provider_events = fetch_split_history(ticker)

    matched_provider_indices: set[int] = set()
    evidence: List[Dict[str, object]] = []
    has_unmatched_inferred = False
    has_ratio_jump = False
    has_unknown = False

    for inferred_date, inferred_ratio, inferred_periods in inferred_events:
        provider_match_index: Optional[int] = None
        for idx, (provider_date, provider_ratio, _source) in enumerate(provider_events):
            if idx in matched_provider_indices:
                continue
            date_close = abs((provider_date - inferred_date).days) <= date_window_days
            ratio_close = abs(provider_ratio - inferred_ratio) <= tolerance
            if date_close and ratio_close:
                provider_match_index = idx
                break

        if provider_match_index is None:
            has_unmatched_inferred = True
            event_date = inferred_date
            provider_ratio = None
            provider_source = None
        else:
            matched_provider_indices.add(provider_match_index)
            event_date, provider_ratio, provider_source = provider_events[provider_match_index]

        neighbor_ratio, before, after = _neighbor_ratio(event_date)
        classification = "adjusted" if _is_ratio_adjusted(neighbor_ratio) else "jump" if neighbor_ratio else "unknown"

        if classification == "jump":
            has_ratio_jump = True
        elif classification == "unknown":
            has_unknown = True

        evidence.append(
            {
                "event_date": event_date,
                "event_kind": "inferred_only" if provider_match_index is None else "provider_match",
                "provider_ratio": provider_ratio,
                "provider_source": provider_source,
                "inferred_ratio": inferred_ratio,
                "inferred_periods": inferred_periods,
                "neighbor_ratio": neighbor_ratio,
                "neighbor_periods": {
                    "before": before,
                    "after": after,
                },
                "classification": classification,
            }
        )

    for idx, (provider_date, provider_ratio, provider_source) in enumerate(provider_events):
        if idx in matched_provider_indices:
            continue

        neighbor_ratio, before, after = _neighbor_ratio(provider_date)
        classification = "adjusted" if _is_ratio_adjusted(neighbor_ratio) else "jump" if neighbor_ratio else "unknown"
        if classification == "jump":
            has_ratio_jump = True
        elif classification == "unknown":
            has_unknown = True

        evidence.append(
            {
                "event_date": provider_date,
                "event_kind": "provider_only",
                "provider_ratio": provider_ratio,
                "provider_source": provider_source,
                "inferred_ratio": None,
                "inferred_periods": None,
                "neighbor_ratio": neighbor_ratio,
                "neighbor_periods": {
                    "before": before,
                    "after": after,
                },
                "classification": classification,
            }
        )

    status: str
    if has_unmatched_inferred or has_ratio_jump:
        status = "unadjusted"
    elif evidence and not has_unknown:
        status = "adjusted"
    else:
        status = "mismatch/inconclusive"

    return status, sorted(evidence, key=lambda row: row.get("event_date") or date.min)


def _load_fiscal_periods(ticker: str, cur: sqlite3.Cursor) -> Tuple[List[date], Optional[date]]:
    annual_periods: List[date] = []
    cur.execute("SELECT Date FROM Annual_Data WHERE Symbol=? ORDER BY Date", (ticker,))
    for (raw_date,) in cur.fetchall():
        try:
            annual_periods.append(datetime.fromisoformat(str(raw_date)).date())
        except Exception:
            continue

    ttm_period: Optional[date] = None
    cur.execute("SELECT Quarter FROM TTM_Data WHERE Symbol=?", (ticker,))
    row = cur.fetchone()
    if row:
        try:
            ttm_period = datetime.fromisoformat(str(row[0])).date()
        except Exception:
            ttm_period = None

    return annual_periods, ttm_period


def _eps_inference_plan(
    ticker: str,
    cur: sqlite3.Cursor,
    provider_history: Optional[List[Tuple[date, float, str]]],
    *,
    tolerance: float = 0.03,
    merge_window_days: int = 180,
    date_window_days: int = 45,
    min_years: int = 10,
) -> Dict[str, object]:
    provider_history = provider_history or fetch_split_history(ticker)

    annual_periods, ttm_period = _load_fiscal_periods(ticker, cur)

    eps_series = load_eps_series(ticker, cur, min_years=min_years)
    inferred_candidates = merge_candidate_events(
        infer_split_candidates(eps_series, tolerance=tolerance),
        merge_window_days=merge_window_days,
        tolerance=tolerance,
    )
    reconciled_events = reconcile_split_events(
        ticker,
        inferred_candidates,
        tolerance=tolerance,
        date_window_days=date_window_days,
    )

    status, evidence = assess_split_adjustment_status(
        ticker,
        cur,
        tolerance=tolerance,
        merge_window_days=merge_window_days,
        date_window_days=date_window_days,
        min_years=min_years,
    )

    status_overrides: Dict[str, str] = {}
    for ev in evidence:
        event_date = ev.get("event_date")
        classification = ev.get("classification")
        if not event_date or classification is None:
            continue
        if ev.get("event_kind") not in {"inferred_only", "provider_match"}:
            continue

        iso_date = event_date.isoformat()
        if classification == "jump":
            status_overrides[iso_date] = "unadjusted"
        else:
            status_overrides[iso_date] = classification

    plan = plan_split_adjustments(
        ticker,
        reconciled_events,
        status_map={
            "annual_periods": annual_periods,
            "ttm_period": ttm_period,
            "event_status": status_overrides,
        },
    )

    adjustable_events = [event for event in plan if event.get("status") == "unadjusted"]

    return {
        "status": status,
        "evidence": evidence,
        "reconciled_events": reconciled_events,
        "plan": plan,
        "adjustable_events": adjustable_events,
        "ttm_period": ttm_period,
    }


def _apply_inferred_adjustments(
    ticker: str,
    cur: sqlite3.Cursor,
    adjustable_events: List[Dict[str, object]],
    ttm_period: Optional[date],
) -> bool:
    if not adjustable_events:
        return False

    applied = False
    sorted_events = sorted(
        adjustable_events,
        key=lambda ev: datetime.fromisoformat(str(ev.get("apply_before"))) if ev.get("apply_before") else date.max,
    )

    for event in sorted_events:
        ratio = event.get("ratio")
        apply_before = event.get("apply_before")
        if ratio is None or apply_before is None:
            continue

        try:
            ratio_value = float(ratio)
            apply_before_date = datetime.fromisoformat(str(apply_before)).date()
        except Exception:
            continue

        if ratio_value <= 0:
            continue

        cur.execute(
            """
            UPDATE Annual_Data
               SET EPS = EPS / ?,
                   Last_Updated = CURRENT_TIMESTAMP
             WHERE Symbol=? AND Date < ?;
            """,
            (ratio_value, ticker, apply_before_date.isoformat()),
        )
        applied = True

    if ttm_period:
        ttm_adjust_ratio = 1.0
        earliest_event_date: Optional[date] = None
        for event in sorted_events:
            ratio = event.get("ratio")
            event_date_raw = event.get("date")
            try:
                event_date = datetime.fromisoformat(str(event_date_raw)).date()
                ratio_value = float(ratio) if ratio is not None else None
            except Exception:
                continue

            if ratio_value is None or ratio_value <= 0 or event_date >= ttm_period:
                continue

            ttm_adjust_ratio *= ratio_value
            if earliest_event_date is None or event_date < earliest_event_date:
                earliest_event_date = event_date

        if ttm_adjust_ratio != 1.0 and earliest_event_date:
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
                    ttm_last_updated = (
                        datetime.fromisoformat(str(ttm_row[3])).date() if ttm_row[3] else None
                    )
                except Exception:
                    ttm_last_updated = None

                should_adjust_ttm = (
                    ttm_quarter is not None
                    and ttm_quarter < earliest_event_date
                    and (ttm_last_updated is None or ttm_last_updated < earliest_event_date)
                )

                if should_adjust_ttm:
                    new_eps = ttm_row[0] / ttm_adjust_ratio if ttm_row[0] is not None else None
                    new_shares = ttm_row[1] * ttm_adjust_ratio if ttm_row[1] is not None else None
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
                    applied = True

    return applied


def apply_split_adjustments(
    ticker: str,
    cur: sqlite3.Cursor,
    *,
    use_eps_inference: bool = False,
    inference_options: Optional[Dict[str, object]] = None,
) -> bool:
    """
    Detect new splits and adjust stored per-share values.

    When ``use_eps_inference`` is enabled, EPS-based inference is consulted
    before provider updates to merge inferred/provider events and, when
    necessary, apply adjustments ahead of recording the split history.

    Returns True when adjustments were applied.
    """

    ensure_splits_table(cur)
    latest_data_date = _latest_financial_date(ticker, cur)
    if not latest_data_date:
        return False

    split_history = fetch_split_history(ticker)
    cur.execute("SELECT Date FROM Splits WHERE Symbol=?", (ticker,))
    recorded_dates = {row[0] for row in cur.fetchall()}

    now_iso = datetime.utcnow().isoformat(sep=" ", timespec="seconds")
    adjustments_applied = False
    inference_result: Optional[Dict[str, object]] = None
    inference_adjusted_dates: set[str] = set()

    if use_eps_inference:
        inference_kwargs = inference_options or {}
        inference_result = _eps_inference_plan(
            ticker,
            cur,
            split_history,
            tolerance=float(inference_kwargs.get("tolerance", 0.03)),
            merge_window_days=int(inference_kwargs.get("merge_window_days", 180)),
            date_window_days=int(inference_kwargs.get("date_window_days", 45)),
            min_years=int(inference_kwargs.get("min_years", 10)),
        )

        logging.info(
            "[%s] EPS inference status: %s (events=%d)",
            ticker,
            inference_result.get("status"),
            len(inference_result.get("reconciled_events") or []),
        )

        adjustable_events = inference_result.get("adjustable_events") or []
        if inference_result.get("status") == "unadjusted" and adjustable_events:
            applied = _apply_inferred_adjustments(
                ticker,
                cur,
                adjustable_events,
                inference_result.get("ttm_period"),
            )
            adjustments_applied = adjustments_applied or applied
            inference_adjusted_dates = {event.get("date") for event in adjustable_events if event.get("date")}
            logging.info(
                "[%s] EPS inference applied adjustments: %s (dates=%s)",
                ticker,
                applied,
                sorted(inference_adjusted_dates),
            )
        else:
            logging.info("[%s] EPS inference found no unadjusted events", ticker)

    events_to_record: Dict[date, Tuple[float, str]] = {dt: (ratio, source) for dt, ratio, source in split_history}
    if inference_result:
        for event in inference_result.get("reconciled_events") or []:
            event_date = event.get("date")
            try:
                ratio_value = float(event.get("ratio"))
            except Exception:
                continue

            source_value = event.get("source") or "inferred"
            status_value = event.get("status")
            if status_value == "provider_match" and source_value:
                source_value = f"{source_value}+inferred"

            if isinstance(event_date, date):
                events_to_record[event_date] = (ratio_value, source_value)

    for dt, (ratio, source) in events_to_record.items():
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
        if dt.isoformat() not in recorded_dates
        and dt.isoformat() not in inference_adjusted_dates
        and dt > latest_data_date
    ]

    if pending_events:
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

        logging.info("[%s] Applied cumulative split ratio %.4f", ticker, cumulative_ratio)
        adjustments_applied = True
    else:
        logging.info("[%s] No provider-based split adjustments pending", ticker)

    cur.connection.commit()
    return adjustments_applied

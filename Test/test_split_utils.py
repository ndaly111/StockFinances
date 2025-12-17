import os
import sqlite3
import sys
from datetime import date

import pytest


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from split_utils import (
    assess_split_adjustment_status,
    merge_candidate_events,
    plan_split_adjustments,
    reconcile_split_events,
)


def _setup_cursor(annual_rows, ttm_row=None):
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()
    cur.execute("CREATE TABLE Annual_Data(Symbol TEXT, Date TEXT, EPS REAL);")
    cur.executemany("INSERT INTO Annual_Data(Symbol, Date, EPS) VALUES(?,?,?)", annual_rows)
    cur.execute("CREATE TABLE TTM_Data(Symbol TEXT, TTM_EPS REAL, Quarter TEXT);")
    if ttm_row:
        cur.execute("INSERT INTO TTM_Data(Symbol, TTM_EPS, Quarter) VALUES(?,?,?)", ttm_row)
    return cur


def test_merge_candidates_within_window_and_matching_ratio():
    candidates = [
        (date(2023, 3, 1), 2.0, ("annual", "annual")),
        (date(2023, 3, 20), 2.02, ("ttm", "annual")),
        (date(2023, 9, 1), 3.0, ("annual", "annual")),
    ]

    merged = merge_candidate_events(candidates, merge_window_days=60, tolerance=0.05)

    assert merged == [
        (date(2023, 3, 1), 2.0, [("annual", "annual"), ("ttm", "annual")]),
        (date(2023, 9, 1), 3.0, [("annual", "annual")]),
    ]


def test_merge_candidates_outside_window_are_not_combined():
    candidates = [
        (date(2023, 1, 1), 2.0, ("annual", "annual")),
        (date(2023, 8, 1), 2.0, ("ttm", "annual")),
    ]

    merged = merge_candidate_events(candidates, merge_window_days=120)

    assert merged == [
        (date(2023, 1, 1), 2.0, [("annual", "annual")]),
        (date(2023, 8, 1), 2.0, [("ttm", "annual")]),
    ]


def test_merge_candidates_with_ratio_mismatch():
    candidates = [
        (date(2023, 1, 1), 2.0, ("annual", "annual")),
        (date(2023, 3, 1), 2.2, ("annual", "ttm")),
    ]

    merged = merge_candidate_events(candidates, merge_window_days=90, tolerance=0.1)

    assert merged == [
        (date(2023, 1, 1), 2.0, [("annual", "annual")]),
        (date(2023, 3, 1), 2.2, [("annual", "ttm")]),
    ]


def test_merge_candidate_events_validates_tolerance():
    candidates = [(date(2023, 1, 1), 2.0, ("annual", "annual"))]

    with pytest.raises(ValueError):
        merge_candidate_events(candidates, tolerance=0)


def test_reconcile_prefers_provider_event(monkeypatch):
    inferred = [(date(2024, 2, 15), 2.0, [("annual", "annual")])]
    provider_events = [
        (date(2024, 2, 10), 1.5, "yfinance"),
        (date(2024, 2, 20), 2.02, "fmp"),
    ]

    monkeypatch.setattr("split_utils.fetch_split_history", lambda ticker: provider_events)

    reconciled = reconcile_split_events("ABC", inferred, tolerance=0.05, date_window_days=30)

    assert reconciled == [
        {
            "date": date(2024, 2, 20),
            "ratio": 2.02,
            "source": "fmp",
            "status": "provider_match",
            "inferred_date": date(2024, 2, 15),
            "inferred_ratio": 2.0,
            "inferred_periods": [("annual", "annual")],
            "provider_date": date(2024, 2, 20),
            "provider_ratio": 2.02,
            "provider_source": "fmp",
        }
    ]


def test_reconcile_marks_inferred_when_no_provider_match(monkeypatch):
    inferred = [(date(2024, 6, 1), 3.0, [("ttm", "annual")])]
    provider_events = [(date(2024, 8, 1), 3.0, "yfinance")]

    monkeypatch.setattr("split_utils.fetch_split_history", lambda ticker: provider_events)

    reconciled = reconcile_split_events("XYZ", inferred, tolerance=0.05, date_window_days=30)

    assert reconciled == [
        {
            "date": date(2024, 6, 1),
            "ratio": 3.0,
            "source": "inferred",
            "status": "inferred_only",
            "inferred_date": date(2024, 6, 1),
            "inferred_ratio": 3.0,
            "inferred_periods": [("ttm", "annual")],
            "provider_date": None,
            "provider_ratio": None,
            "provider_source": None,
        }
    ]


def test_plan_split_adjustments_uses_fiscal_boundaries_and_overrides():
    merged_events = [
        {"date": date(2023, 7, 15), "ratio": 2.0, "source": "yfinance", "status": "provider_match"}
    ]
    status_map = {
        "annual_periods": [date(2021, 12, 31), date(2022, 12, 31), date(2023, 12, 31)],
        "ttm_period": date(2024, 3, 31),
        "event_status": {"2023-07-15": "pending_adjustment"},
    }

    plan = plan_split_adjustments("ABC", merged_events, status_map)

    assert plan == [
        {
            "date": "2023-07-15",
            "ratio": 2.0,
            "apply_before": "2023-12-31",
            "affected_years": [2021, 2022],
            "source": "yfinance",
            "status": "pending_adjustment",
        }
    ]


def test_plan_split_adjustments_includes_ttm_boundary():
    merged_events = [
        {"date": date(2024, 2, 1), "ratio": 3.0, "source": "fmp", "status": "provider_match"}
    ]
    status_map = {
        "annual_periods": [date(2021, 12, 31), date(2022, 12, 31), date(2023, 12, 31)],
        "ttm_period": date(2024, 3, 31),
    }

    plan = plan_split_adjustments("XYZ", merged_events, status_map)

    assert plan[0]["apply_before"] == "2024-03-31"
    assert plan[0]["affected_years"] == [2021, 2022, 2023]


def test_assess_status_adjusted(monkeypatch):
    cur = _setup_cursor(
        [
            ("ABC", "2021-12-31", 1.0),
            ("ABC", "2022-12-31", 1.02),
            ("ABC", "2023-12-31", 1.05),
        ]
    )

    monkeypatch.setattr(
        "split_utils.fetch_split_history",
        lambda ticker: [(date(2022, 6, 1), 2.0, "yfinance")],
    )

    status, evidence = assess_split_adjustment_status(
        "ABC", cur, tolerance=0.05, min_years=3, date_window_days=30
    )

    assert status == "adjusted"
    assert evidence[0]["classification"] == "adjusted"
    assert evidence[0]["neighbor_ratio"] is not None


def test_assess_status_unadjusted_when_inferred_missing_provider(monkeypatch):
    cur = _setup_cursor(
        [
            ("XYZ", "2021-12-31", 4.0),
            ("XYZ", "2022-12-31", 2.0),
            ("XYZ", "2023-12-31", 1.9),
        ]
    )
    monkeypatch.setattr("split_utils.fetch_split_history", lambda ticker: [])

    status, evidence = assess_split_adjustment_status(
        "XYZ", cur, tolerance=0.05, min_years=3, date_window_days=30
    )

    assert status == "unadjusted"
    assert any(item["event_kind"] == "inferred_only" for item in evidence)


def test_assess_status_unadjusted_when_provider_ratio_jumps(monkeypatch):
    cur = _setup_cursor(
        [
            ("AAA", "2021-12-31", 4.0),
            ("AAA", "2022-12-31", 2.0),
            ("AAA", "2023-12-31", 2.1),
        ]
    )
    monkeypatch.setattr(
        "split_utils.fetch_split_history",
        lambda ticker: [(date(2022, 6, 1), 2.0, "yfinance")],
    )

    status, evidence = assess_split_adjustment_status(
        "AAA", cur, tolerance=0.05, min_years=3, date_window_days=30
    )

    assert status == "unadjusted"
    assert evidence[0]["classification"] == "jump"


def test_assess_status_inconclusive_when_not_enough_eps(monkeypatch):
    cur = _setup_cursor([("LMN", "2023-12-31", 1.0)])
    monkeypatch.setattr("split_utils.fetch_split_history", lambda ticker: [])

    status, evidence = assess_split_adjustment_status(
        "LMN", cur, tolerance=0.05, min_years=1, date_window_days=30
    )

    assert status == "mismatch/inconclusive"
    assert evidence[0]["issue"] == "insufficient_eps_points"

import os
import sys
from datetime import date

import pytest


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from split_utils import merge_candidate_events


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

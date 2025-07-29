import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generate_earnings_tables import _pick, _clean, _calc_surprise


def test_pick_handles_space():
    row = {"Surprise (%)": 5.0}
    assert _pick(row, "Surprise (%)", "Surprise(%)") == 5.0


def test_calc_surprise_manual():
    est = 2.0
    actual = 2.5
    surprise = None
    result = _calc_surprise(est, actual, surprise)
    assert result == 25.0
    # Provided surprise should be used as-is
    assert _calc_surprise(est, actual, 30.0) == 30.0

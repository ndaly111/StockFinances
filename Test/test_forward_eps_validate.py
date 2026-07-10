import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import forward_eps_validate as v

def test_displayable_when_in_range():
    assert v.is_displayable("QQQ", growth_this_fy=0.20, coverage_weight=0.92) is True

def test_withheld_low_coverage():
    assert v.is_displayable("QQQ", growth_this_fy=0.19, coverage_weight=0.70) is False

def test_withheld_out_of_tolerance():
    assert v.is_displayable("QQQ", growth_this_fy=0.60, coverage_weight=0.95) is False

def test_qqq_wide_band_accepts_high_but_plausible_growth():
    # QQQ real growth runs high in the AI cycle (+38% bottom-up, +40% market-
    # implied, verified 2026-07-10); it must pass with good coverage.
    assert v.is_displayable("QQQ", growth_this_fy=0.38, coverage_weight=0.92) is True

def test_qqq_band_still_rejects_gross_error():
    # A scaling bug (e.g. +90%) must still be withheld even at full coverage.
    assert v.is_displayable("QQQ", growth_this_fy=0.90, coverage_weight=0.99) is False

def test_none_growth_withheld():
    assert v.is_displayable("SPY", growth_this_fy=None, coverage_weight=0.95) is False

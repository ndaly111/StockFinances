import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import forward_eps_validate as v

def test_displayable_when_in_range():
    assert v.is_displayable("QQQ", growth_this_fy=0.20, coverage_weight=0.92) is True

def test_withheld_low_coverage():
    assert v.is_displayable("QQQ", growth_this_fy=0.19, coverage_weight=0.70) is False

def test_withheld_out_of_tolerance():
    assert v.is_displayable("QQQ", growth_this_fy=0.45, coverage_weight=0.95) is False

def test_none_growth_withheld():
    assert v.is_displayable("SPY", growth_this_fy=None, coverage_weight=0.95) is False

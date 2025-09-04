import sys, pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from generate_segment_charts import _choose_scale


def test_choose_scale_uses_billions_when_small_segments_present():
    vals = [1.5e12, 2e9, 5e11]
    div, unit = _choose_scale(vals)
    assert div == 1e9 and unit == "$B"


def test_choose_scale_prefers_trillions_when_all_values_large():
    vals = [2.1e12, 1.8e12, 1.2e12]
    div, unit = _choose_scale(vals)
    assert div == 1e12 and unit == "$T"

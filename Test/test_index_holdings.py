import io, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import index_holdings as ih

def test_parse_slickcharts_holdings():
    html = (ROOT/"Test"/"fixtures"/"slickcharts_nasdaq100.html").read_text(encoding="utf-8")
    rows = ih.parse_slickcharts(html)
    assert len(rows) >= 99
    d = dict(rows)
    assert "NVDA" in d and "AAPL" in d
    assert 0 < d["NVDA"] < 100
    assert 95 < sum(w for _, w in rows) < 105

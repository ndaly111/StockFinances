import pathlib, sqlite3, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import index_forward_eps as ife


def test_ensure_table_creates_schema(tmp_path):
    db = tmp_path / "t.db"
    with sqlite3.connect(db) as conn:
        ife.ensure_forward_eps_table(conn)
        cols = {r[1] for r in conn.execute(
            "PRAGMA table_info(Index_Forward_EPS_History)")}
    assert cols == {
        "date_recorded", "ticker", "forward_eps_etf",
        "forward_eps_index", "forward_pe", "horizon_date", "source",
    }


def test_divisor_matches_chart_module():
    import index_growth_charts as igc
    assert ife.INDEX_EPS_DIVISOR == igc._INDEX_EPS_DIVISOR


def test_forward_from_yf_info_scales_to_index():
    info = {"forwardPE": 22.0, "forwardEps": 25.0, "regularMarketPrice": 550.0}
    fe = ife._forward_from_yf("SPY", info)
    assert fe is not None
    assert fe.forward_pe == 22.0
    assert fe.forward_eps_etf == 25.0
    assert fe.forward_eps_index == 250.0   # 25.0 * divisor(10)
    assert fe.source == "yfinance"


def test_forward_from_yf_derives_eps_when_missing():
    # No forwardEps -> derive from price / forwardPE
    info = {"forwardPE": 20.0, "regularMarketPrice": 500.0}
    fe = ife._forward_from_yf("QQQ", info)
    assert fe is not None
    assert round(fe.forward_eps_etf, 4) == 25.0      # 500/20
    assert round(fe.forward_eps_index, 4) == 100.0   # 25 * divisor(4)


def test_forward_from_yf_returns_none_without_pe():
    assert ife._forward_from_yf("SPY", {"regularMarketPrice": 500.0}) is None

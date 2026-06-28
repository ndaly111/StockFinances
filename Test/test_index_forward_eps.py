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


def test_sanity_accepts_reasonable():
    # forward index EPS 250 vs latest 230 -> +8.7% growth, scale ~1.09x
    assert ife._passes_sanity(forward_pe=22.0, forward_eps_index=250.0,
                              latest_hist_eps=230.0) is True


def test_sanity_rejects_nonpositive_pe():
    assert ife._passes_sanity(0.0, 250.0, 230.0) is False
    assert ife._passes_sanity(-5.0, 250.0, 230.0) is False


def test_sanity_rejects_out_of_band_growth():
    # +200% implied growth is absurd for an index
    assert ife._passes_sanity(22.0, 690.0, 230.0) is False
    # -60% collapse
    assert ife._passes_sanity(22.0, 90.0, 230.0) is False


def test_sanity_rejects_bad_scale():
    # forward index EPS 3x the latest -> scaling/source error
    assert ife._passes_sanity(22.0, 800.0, 230.0) is False


def test_sanity_allows_missing_history():
    # No history to compare against -> only the P/E and absolute checks apply
    assert ife._passes_sanity(22.0, 250.0, None) is True

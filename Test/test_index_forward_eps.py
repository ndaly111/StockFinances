import pathlib, sqlite3, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from unittest.mock import patch

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


def test_sanity_rejects_gross_scaling_error():
    # A 3.5x ETF/index scaling error manifests as absurd implied growth (+248%),
    # so the growth band rejects it.
    assert ife._passes_sanity(22.0, 800.0, 230.0) is False


def test_sanity_allows_missing_history():
    # No history to compare against -> only the P/E and absolute checks apply
    assert ife._passes_sanity(22.0, 250.0, None) is True


def test_parse_stockanalysis_extracts_forward_pe():
    html = (ROOT / "Test" / "fixtures" / "stockanalysis_spy.html").read_text(
        encoding="utf-8")
    pe = ife._parse_forward_pe(html)
    assert pe is not None
    assert 5.0 < pe < 60.0     # sane forward P/E for SPY


def test_fetch_prefers_stockanalysis_with_price():
    # stockanalysis gives forward PE; price comes from yfinance info.
    with (
        patch.object(ife, "_fetch_stockanalysis_pe", return_value=20.0),
        patch.object(ife, "_yf_info", return_value={"regularMarketPrice": 460.0}),
    ):
        fe = ife.fetch_forward_eps("SPY", session=object(), latest_hist_eps=230.0)
    assert fe is not None
    assert fe.source == "stockanalysis"
    assert fe.forward_pe == 20.0
    assert round(fe.forward_eps_etf, 4) == 23.0       # 460/20
    assert round(fe.forward_eps_index, 4) == 230.0    # *10


def test_fetch_falls_back_to_yfinance():
    with (
        patch.object(ife, "_fetch_stockanalysis_pe", return_value=None),
        patch.object(ife, "_yf_info",
                     return_value={"forwardPE": 22.0, "forwardEps": 24.0,
                                   "regularMarketPrice": 528.0}),
    ):
        fe = ife.fetch_forward_eps("SPY", session=object(), latest_hist_eps=230.0)
    assert fe is not None
    assert fe.source == "yfinance"
    assert fe.forward_eps_index == 240.0              # 24 * 10


def test_fetch_returns_none_when_sanity_fails():
    # forward PE 2.0 -> EPS 230 etf -> 2300 index -> absurd scale, rejected
    with (
        patch.object(ife, "_fetch_stockanalysis_pe", return_value=2.0),
        patch.object(ife, "_yf_info", return_value={"regularMarketPrice": 460.0}),
    ):
        fe = ife.fetch_forward_eps("SPY", session=object(), latest_hist_eps=230.0)
    assert fe is None


def test_snapshot_upserts_and_is_idempotent(tmp_path):
    db = tmp_path / "t.db"
    fe = ife.ForwardEPS("SPY", 20.0, 23.0, 230.0, "2027-06-28", "yfinance")

    def fake_fetch(tk, session, latest_hist_eps):
        return fe if tk == "SPY" else None

    with sqlite3.connect(db) as conn:
        ife.ensure_forward_eps_table(conn)
        with patch.object(ife, "fetch_forward_eps", side_effect=fake_fetch), \
             patch.object(ife, "_latest_hist_eps", return_value=225.0):
            ife.snapshot_forward_eps(conn)
            ife.snapshot_forward_eps(conn)   # same day -> overwrite, not dup
        rows = list(conn.execute(
            f"SELECT ticker, forward_eps_index, source FROM {ife.TABLE}"))
    assert rows == [("SPY", 230.0, "yfinance")]   # QQQ skipped (None), no dup

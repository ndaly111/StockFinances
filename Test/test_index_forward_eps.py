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
    assert {
        "date_recorded", "ticker", "forward_eps_etf",
        "forward_eps_index", "forward_pe", "horizon_date", "source",
    } <= cols


def test_divisor_matches_chart_module():
    import generate_index_growth_pages as gigp
    assert ife.INDEX_EPS_DIVISOR == gigp._INDEX_EPS_DIVISOR


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


def test_ensure_table_has_bottomup_columns(tmp_path):
    db = tmp_path / "t.db"
    with sqlite3.connect(db) as conn:
        ife.ensure_forward_eps_table(conn)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(Index_Forward_EPS_History)")}
    for c in ("coverage_weight", "growth_this_fy", "growth_next_fy", "method", "displayable"):
        assert c in cols


def test_ensure_constituents_table(tmp_path):
    db = tmp_path / "t.db"
    with sqlite3.connect(db) as conn:
        ife.ensure_constituents_table(conn)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(index_constituents)")}
    assert {"date_recorded", "index_name", "ticker", "weight"} <= cols


def test_snapshot_bottom_up(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    import index_forward_eps as ife, index_holdings as ih, forward_eps_bottom_up as bu, forward_eps_validate as v
    holdings = [("AAA", 60.0), ("BBB", 30.0), ("CCC", 10.0)]
    fin = {"AAA": {"ttm_eps": 10.0, "shares": 100.0, "this_fy": 12.0, "next_fy": 14.0},
           "BBB": {"ttm_eps": 5.0, "shares": 200.0, "this_fy": 5.5, "next_fy": 6.0}}
    monkeypatch.setattr(ih, "fetch_holdings", lambda idx: holdings)
    monkeypatch.setattr(ih, "uncovered_for_target", lambda h, c, target_pct=90.0: [])
    monkeypatch.setattr(bu, "load_constituent_financials", lambda conn, tks: fin)
    monkeypatch.setattr(v, "is_displayable", lambda idx, growth_this_fy, coverage_weight: True)
    with sqlite3.connect(db) as conn:
        n = ife.snapshot_forward_eps(conn)
        rows = list(conn.execute("SELECT ticker, growth_this_fy, coverage_weight, displayable, method FROM Index_Forward_EPS_History"))
    assert n == 2
    g = {r[0]: r for r in rows}
    assert round(g["QQQ"][1], 4) == round(2300 / 2000 - 1, 4)
    assert g["QQQ"][3] == 1 and g["QQQ"][4] == "bottom_up"

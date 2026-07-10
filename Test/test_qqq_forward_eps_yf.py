"""Tests for qqq_forward_eps_yf — consistent-basis bottom-up QQQ forward EPS
from yfinance analyst estimates, snapshotted like FactSet does for SPY."""
import pathlib, sqlite3, sys
from datetime import date

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import qqq_forward_eps_yf as qy
import index_forward_eps as ife


HOLDINGS = [("AAA", 40.0), ("BBB", 30.0), ("CCC", 20.0), ("DDD", 10.0)]

FAKE = {
    # this_fy / year_ago on the SAME (adjusted) basis; shares scale the dollars
    "AAA": {"this_fy": 12.0, "year_ago": 10.0, "next_fy": 15.0, "shares": 100.0},
    "BBB": {"this_fy": 6.0,  "year_ago": 5.0,  "next_fy": 6.6,  "shares": 200.0},
    "CCC": {"this_fy": 4.4,  "year_ago": 4.0,  "next_fy": None, "shares": 300.0},
    "DDD": None,   # yfinance has nothing for this one
}


def _fetch(tk, info=None):
    return FAKE.get(tk)


def test_compute_aggregates_consistent_basis():
    r = qy.compute_qqq_bottom_up(HOLDINGS, fetch=_fetch, target_pct=100.0)
    # dollars: this = 12*100 + 6*200 + 4.4*300 = 3720 ; yearAgo = 1000+1000+1200 = 3200
    assert abs(r["growth_this_fy"] - (3720.0 / 3200.0 - 1)) < 1e-9
    # next-FY growth restricted to names that HAVE a next-FY estimate (AAA+BBB):
    # next = 15*100 + 6.6*200 = 2820 ; this(same names) = 1200+1200 = 2400
    assert abs(r["growth_next_fy"] - (2820.0 / 2400.0 - 1)) < 1e-9
    # DDD (10% of weight) missing -> coverage 90%
    assert abs(r["coverage_weight"] - 0.90) < 1e-9


def test_compute_none_when_no_data():
    r = qy.compute_qqq_bottom_up(HOLDINGS, fetch=lambda tk, info=None: None,
                                 target_pct=100.0)
    assert r["growth_this_fy"] is None


def test_compute_universe_stops_at_target():
    seen = []
    def spy_fetch(tk, info=None):
        seen.append(tk)
        return FAKE.get(tk)
    qy.compute_qqq_bottom_up(HOLDINGS, fetch=spy_fetch, target_pct=70.0)
    # 40+30 = 70% reaches the target; CCC/DDD must not be fetched
    assert set(seen) == {"AAA", "BBB"}


def _seed_db(tmp_path):
    conn = sqlite3.connect(tmp_path / "t.db")
    ife.ensure_forward_eps_table(conn)
    conn.execute("""CREATE TABLE Index_EPS_History
        (Ticker TEXT, EPS_Type TEXT, Date TEXT, EPS REAL)""")
    conn.execute("INSERT INTO Index_EPS_History VALUES ('QQQ','TTM_REPORTED','2026-05-01',90.0)")
    conn.commit()
    return conn


def test_snapshot_writes_displayable_row(tmp_path):
    conn = _seed_db(tmp_path)
    result = {"growth_this_fy": 0.38, "growth_next_fy": 0.30, "coverage_weight": 0.90}
    n = qy.snapshot_qqq_yf(conn, today=date(2026, 7, 10), result=result)
    assert n == 1
    row = conn.execute("""SELECT forward_eps_index, growth_this_fy, growth_next_fy,
        coverage_weight, source, displayable, horizon_date
        FROM Index_Forward_EPS_History WHERE ticker='QQQ'""").fetchone()
    assert abs(row[0] - 90.0 * 1.38) < 1e-9      # scaled off latest hist EPS
    assert row[1] == 0.38 and row[2] == 0.30 and row[3] == 0.90
    assert row[4] == "yahoo_bottom_up"
    assert row[5] == 1
    assert row[6] == "2026-12-31"


def test_snapshot_withholds_gross_error(tmp_path):
    conn = _seed_db(tmp_path)
    # A x4-divisor style bug -> absurd growth; row written but displayable=0
    result = {"growth_this_fy": 2.5, "growth_next_fy": None, "coverage_weight": 0.95}
    qy.snapshot_qqq_yf(conn, today=date(2026, 7, 10), result=result)
    disp = conn.execute("SELECT displayable FROM Index_Forward_EPS_History "
                        "WHERE ticker='QQQ'").fetchone()[0]
    assert disp == 0


def test_snapshot_noop_without_growth_or_hist(tmp_path):
    conn = _seed_db(tmp_path)
    assert qy.snapshot_qqq_yf(conn, today=date(2026, 7, 10),
                              result={"growth_this_fy": None,
                                      "growth_next_fy": None,
                                      "coverage_weight": 0.0}) == 0
    # and with growth but no Index_EPS_History row for QQQ
    conn.execute("DELETE FROM Index_EPS_History")
    conn.commit()
    assert qy.snapshot_qqq_yf(conn, today=date(2026, 7, 10),
                              result={"growth_this_fy": 0.38,
                                      "growth_next_fy": None,
                                      "coverage_weight": 0.9}) == 0

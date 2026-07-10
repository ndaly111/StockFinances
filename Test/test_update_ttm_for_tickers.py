"""Tests for annual_and_ttm_update.update_ttm_for_tickers — populate TTM_Data for
index constituents via the yfinance path, skipping names with missing/NaN data."""
import pathlib, sqlite3, sys
import math

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import annual_and_ttm_update as atu


def _mk_ttm_table(conn):
    conn.execute("""CREATE TABLE TTM_Data (Symbol TEXT PRIMARY KEY, TTM_Revenue REAL,
        TTM_Net_Income REAL, TTM_EPS REAL, Shares_Outstanding REAL, Quarter TEXT,
        Last_Updated TEXT)""")
    conn.commit()


def test_writes_good_rows_and_skips_bad(monkeypatch, tmp_path):
    fake = {
        "ASML": {"TTM_Revenue": 3e10, "TTM_Net_Income": 1e10, "TTM_EPS": 30.13,
                 "Shares_Outstanding": 385417665, "Quarter": "2026-03-31"},
        "ADI":  {"TTM_Revenue": 1e10, "TTM_Net_Income": 3e9, "TTM_EPS": 6.85,
                 "Shares_Outstanding": 487087040, "Quarter": "2026-04-30"},
        "BAD1": {"TTM_EPS": float("nan"), "Shares_Outstanding": 1e9, "Quarter": "2026-03-31"},
        "BAD2": {"TTM_EPS": 5.0, "Shares_Outstanding": None, "Quarter": "2026-03-31"},
        "NONE": None,
    }
    monkeypatch.setattr(atu, "_fetch_ttm", lambda tk: fake.get(tk))

    conn = sqlite3.connect(tmp_path / "t.db")
    _mk_ttm_table(conn)
    n = atu.update_ttm_for_tickers(["ASML", "ADI", "BAD1", "BAD2", "NONE"], conn.cursor())
    conn.commit()

    assert n == 2
    rows = dict(conn.execute("SELECT Symbol, TTM_EPS FROM TTM_Data").fetchall())
    assert rows == {"ASML": 30.13, "ADI": 6.85}


def test_fetch_exception_is_isolated(monkeypatch, tmp_path):
    def _boom(tk):
        if tk == "ERR":
            raise RuntimeError("yfinance down")
        return {"TTM_EPS": 4.0, "Shares_Outstanding": 1e9, "Quarter": "2026-03-31"}
    monkeypatch.setattr(atu, "_fetch_ttm", _boom)

    conn = sqlite3.connect(tmp_path / "t.db")
    _mk_ttm_table(conn)
    n = atu.update_ttm_for_tickers(["ERR", "OK"], conn.cursor())
    conn.commit()
    assert n == 1
    assert conn.execute("SELECT Symbol FROM TTM_Data").fetchone()[0] == "OK"

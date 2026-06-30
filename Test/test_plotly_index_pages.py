import pathlib, sqlite3, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import generate_index_growth_pages as g
import pandas as pd

def _seed_eps(conn):
    conn.execute("""CREATE TABLE Index_EPS_History (Date TEXT, Ticker TEXT, EPS_Type TEXT, EPS REAL)""")
    conn.executemany("INSERT INTO Index_EPS_History VALUES (?,?,?,?)", [
        ("2024-01-31","QQQ","IMPLIED_FROM_PE",20.0),
        ("2024-02-29","QQQ","IMPLIED_FROM_PE",22.0),
        ("2024-01-31","SPY","TTM_REPORTED",230.0),
    ])
    conn.commit()

def test_index_eps_series_scales_qqq_implied():
    conn = sqlite3.connect(":memory:"); _seed_eps(conn)
    s = g._index_eps_series(conn, "QQQ")
    assert round(float(s.iloc[-1]), 2) == 88.0   # 22 * divisor(4)

def test_index_eps_series_spy_reported():
    conn = sqlite3.connect(":memory:"); _seed_eps(conn)
    s = g._index_eps_series(conn, "SPY")
    assert round(float(s.iloc[-1]), 2) == 230.0

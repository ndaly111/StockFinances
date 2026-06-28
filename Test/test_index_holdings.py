import io, pathlib, sys, sqlite3
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


def test_persist_holdings(tmp_path):
    db = tmp_path/"t.db"
    with sqlite3.connect(db) as conn:
        import index_forward_eps as ife; ife.ensure_constituents_table(conn)
        ih.persist_holdings(conn, "QQQ", [("NVDA",12.3),("AAPL",11.0)], today="2026-06-28")
        ih.persist_holdings(conn, "QQQ", [("NVDA",12.4),("AAPL",11.1)], today="2026-06-28")  # idempotent
        rows = list(conn.execute("SELECT ticker, weight FROM index_constituents WHERE index_name='QQQ' ORDER BY ticker"))
    assert rows == [("AAPL",11.1),("NVDA",12.4)]

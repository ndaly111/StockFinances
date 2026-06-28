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

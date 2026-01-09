import sqlite3
from pathlib import Path

from scripts import load_index_eps_csv as loader


def _fetch_eps_rows(db_path: Path):
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            "SELECT Date, EPS FROM Index_EPS_History WHERE Ticker='SPY' AND EPS_Type='TTM_REPORTED'"
        )
        return dict(cur.fetchall())


def test_load_eps_csv_idempotent(tmp_path: Path):
    db_path = tmp_path / "eps.db"
    csv_path = tmp_path / "eps.csv"
    csv_path.write_text(
        """Date,SPY_EPS
2024/01/01,10
2024-01-01,11
2024-02-01,12
"""
    )

    inserted_first = loader.load_eps_csv(
        db_path=str(db_path),
        csv_path=str(csv_path),
        ticker="SPY",
        eps_type="TTM_REPORTED",
        column="SPY_EPS",
    )
    assert inserted_first == 2

    rows = _fetch_eps_rows(db_path)
    assert rows["2024-01-01"] == 11.0
    assert rows["2024-02-01"] == 12.0

    inserted_second = loader.load_eps_csv(
        db_path=str(db_path),
        csv_path=str(csv_path),
        ticker="SPY",
        eps_type="TTM_REPORTED",
        column="SPY_EPS",
    )
    assert inserted_second == 2

    rows_after = _fetch_eps_rows(db_path)
    assert rows_after == rows

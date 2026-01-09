import sqlite3
from pathlib import Path

from scripts import load_index_price_csv as loader


def _fetch_price_rows(db_path: Path):
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            "SELECT Date, Close FROM Index_Price_History_Monthly WHERE Ticker='SPY'"
        )
        return dict(cur.fetchall())


def test_load_price_csv_idempotent(tmp_path: Path):
    db_path = tmp_path / "prices.db"
    csv_path = tmp_path / "prices.csv"
    csv_path.write_text(
        """Date,Close
2024-01-01,100
2024-01-01,101
2024-02-01,102
"""
    )

    inserted_first = loader.load_price_csv(
        db_path=str(db_path),
        csv_path=str(csv_path),
        ticker="SPY",
        date_column="Date",
        close_column="Close",
    )
    assert inserted_first == 2

    rows = _fetch_price_rows(db_path)
    assert rows["2024-01-01"] == 101.0
    assert rows["2024-02-01"] == 102.0

    inserted_second = loader.load_price_csv(
        db_path=str(db_path),
        csv_path=str(csv_path),
        ticker="SPY",
        date_column="Date",
        close_column="Close",
    )
    assert inserted_second == 2

    rows_after = _fetch_price_rows(db_path)
    assert rows_after == rows

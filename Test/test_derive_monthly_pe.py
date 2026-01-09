import pathlib
import sqlite3
import sys
from pathlib import Path

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import derive_monthly_pe_from_price_and_eps as derive


def test_derive_monthly_pe_from_price_and_eps(tmp_path: Path):
    db_path = tmp_path / "derive.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE Index_Price_History_Monthly (
                Date TEXT NOT NULL,
                Ticker TEXT NOT NULL,
                Close REAL,
                PRIMARY KEY (Date, Ticker)
            );
            CREATE TABLE Index_EPS_History (
                Date TEXT NOT NULL,
                Ticker TEXT NOT NULL,
                EPS_Type TEXT NOT NULL,
                EPS REAL,
                PRIMARY KEY (Date, Ticker, EPS_Type)
            );
            """
        )
        conn.executemany(
            "INSERT INTO Index_Price_History_Monthly (Date, Ticker, Close) VALUES (?,?,?)",
            [
                ("2024-01-01", "SPY", 100.0),
                ("2024-02-01", "SPY", 110.0),
            ],
        )
        conn.executemany(
            "INSERT INTO Index_EPS_History (Date, Ticker, EPS_Type, EPS) VALUES (?,?,?,?)",
            [
                ("2024-01-31", "SPY", "TTM_REPORTED", 10.0),
                ("2024-02-29", "SPY", "TTM_REPORTED", 11.0),
            ],
        )
        conn.commit()

    inserted = derive.derive_monthly_pe(db_path=str(db_path), ticker="SPY")
    assert inserted == 2

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT Date, PE_Ratio FROM Index_PE_History
            WHERE Ticker='SPY' AND PE_Type='TTM_DERIVED_MONTHLY'
            ORDER BY Date
            """
        ).fetchall()

    assert rows == [("2024-01-31", 10.0), ("2024-02-29", 10.0)]


def test_derive_monthly_pe_aligns_by_month(tmp_path: Path):
    db_path = tmp_path / "derive_align.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE Index_Price_History_Monthly (
                Date TEXT NOT NULL,
                Ticker TEXT NOT NULL,
                Close REAL,
                PRIMARY KEY (Date, Ticker)
            );
            CREATE TABLE Index_EPS_History (
                Date TEXT NOT NULL,
                Ticker TEXT NOT NULL,
                EPS_Type TEXT NOT NULL,
                EPS REAL,
                PRIMARY KEY (Date, Ticker, EPS_Type)
            );
            """
        )
        conn.executemany(
            "INSERT INTO Index_Price_History_Monthly (Date, Ticker, Close) VALUES (?,?,?)",
            [
                ("2024-01-01", "SPY", 100.0),
                ("2024-02-01", "SPY", 120.0),
            ],
        )
        conn.executemany(
            "INSERT INTO Index_EPS_History (Date, Ticker, EPS_Type, EPS) VALUES (?,?,?,?)",
            [
                ("2024-01-31", "SPY", "TTM_REPORTED", 10.0),
                ("2024-02-28", "SPY", "TTM_REPORTED", 12.0),
            ],
        )
        conn.commit()

    inserted = derive.derive_monthly_pe(db_path=str(db_path), ticker="SPY")
    assert inserted == 2

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT Date, PE_Ratio FROM Index_PE_History
            WHERE Ticker='SPY' AND PE_Type='TTM_DERIVED_MONTHLY'
            ORDER BY Date
            """
        ).fetchall()

    assert rows == [("2024-01-31", 10.0), ("2024-02-29", 10.0)]

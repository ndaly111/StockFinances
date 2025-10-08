import pathlib
import sqlite3
import sys

import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backfill_index_pe_history import (
    _compute_eps_windows,
    _merge_prices_with_eps,
    _ensure_tables,
    _has_history,
    _upsert_history,
)


def test_compute_eps_windows_produces_ttm_and_forward_eps():
    quarters = pd.date_range("2020-03-31", periods=8, freq="QE-DEC")
    earnings = pd.DataFrame(
        {
            "epsActual": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "epsEstimate": [
                1.1,
                2.1,
                3.1,
                4.1,
                5.1,
                6.1,
                7.1,
                8.1,
            ],
        },
        index=quarters,
    )

    result = _compute_eps_windows(earnings)
    result = result.set_index("date")

    # Trailing EPS should sum the last four actual values.
    assert result.loc[pd.Timestamp("2020-12-31"), "ttm_eps"] == pytest.approx(10.0)
    assert result.loc[pd.Timestamp("2021-03-31"), "ttm_eps"] == pytest.approx(14.0)

    # Forward EPS should sum the next four estimates when available.
    assert result.loc[pd.Timestamp("2020-03-31"), "forward_eps"] == pytest.approx(10.4)
    assert result.loc[pd.Timestamp("2020-06-30"), "forward_eps"] == pytest.approx(14.4)


def test_merge_prices_with_eps_aligns_latest_quarter():
    prices = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=5, freq="D"),
            "close": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )
    eps = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-04")],
            "ttm_eps": [1.0, 2.0],
            "forward_eps": [1.5, 2.5],
        }
    )

    merged = _merge_prices_with_eps(prices, eps)

    # As of 2020-01-03 we should still be using the 2020-01-02 EPS snapshot.
    row = merged.loc[merged["date"] == pd.Timestamp("2020-01-03")].iloc[0]
    assert row["pe_ttm"] == 30.0
    assert row["pe_forward"] == 20.0

    # Once the 2020-01-04 EPS arrives, subsequent days use the updated ratios.
    row = merged.loc[merged["date"] == pd.Timestamp("2020-01-05")].iloc[0]
    assert row["pe_ttm"] == 25.0
    assert row["pe_forward"] == 20.0


def test_upsert_history_creates_rows_and_has_history_flag():
    conn = sqlite3.connect(":memory:")
    try:
        _ensure_tables(conn)

        history = pd.DataFrame(
            {
                "date": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")],
                "pe_ttm": [15.0, 16.0],
                "pe_forward": [14.0, pd.NA],
            }
        )

        inserted = _upsert_history(conn, "TEST", history)
        assert inserted == 3  # two TTM rows + one forward

        cur = conn.cursor()
        cur.execute(
            "SELECT Date, PE_Type, PE_Ratio FROM Index_PE_History ORDER BY Date, PE_Type"
        )
        rows = cur.fetchall()
        assert rows == [
            ("2020-01-01", "Forward", 14.0),
            ("2020-01-01", "TTM", 15.0),
            ("2020-01-02", "TTM", 16.0),
        ]

        assert not _has_history(conn, "TEST", pd.Timestamp("2019-12-31"))
        assert _has_history(conn, "TEST", pd.Timestamp("2020-01-01"))
    finally:
        conn.close()

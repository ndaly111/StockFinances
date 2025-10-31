import sqlite3
from pathlib import Path
from textwrap import dedent

import pandas as pd
import pytest

import populate_index_history as pih


def _fake_fetch_factory(mapping):
    def fetch(url: str) -> str:
        try:
            return mapping[url]
        except KeyError:
            raise AssertionError(f"Unexpected URL requested: {url}")

    return fetch


def test_http_fetch_rejects_binary_marker(monkeypatch):
    class FakeResponse:
        status_code = 200
        headers = {"content-type": "text/html"}

        @staticmethod
        def raise_for_status():
            return None

        @property
        def text(self):  # pragma: no cover - property used in implementation
            return "Binary files are not supported"

    def fake_get(url, timeout):  # pragma: no cover - called by implementation
        assert url == "https://example.test/binary.csv"
        assert timeout == 30
        return FakeResponse()

    monkeypatch.setattr(pih.requests, "get", fake_get)

    with pytest.raises(RuntimeError) as excinfo:
        pih._http_fetch("https://example.test/binary.csv")

    assert "Binary files are not supported" in str(excinfo.value)


@pytest.fixture()
def temp_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            dedent(
                """
                CREATE TABLE economic_data (
                    Date TEXT,
                    indicator TEXT,
                    value REAL
                );
                CREATE TABLE Index_PE_History (
                    Date TEXT,
                    Ticker TEXT,
                    PE_Type TEXT,
                    PE_Ratio REAL
                );
                CREATE TABLE Index_Growth_History (
                    Date TEXT,
                    Ticker TEXT,
                    Growth_Type TEXT,
                    Implied_Growth REAL
                );
                CREATE TABLE Index_EPS_History (
                    Date TEXT,
                    Ticker TEXT,
                    EPS_Type TEXT,
                    EPS REAL
                );
                CREATE TABLE Implied_Growth_History (
                    ticker TEXT,
                    growth_type TEXT,
                    growth_value REAL,
                    date_recorded TEXT
                );
                """
            )
        )

        # Provide a small window of treasury yields.
        yields = pd.DataFrame(
            {
                "Date": pd.date_range("2023-12-28", periods=10, freq="D"),
                "indicator": "DGS10",
                "value": [3.5 + i * 0.01 for i in range(10)],
            }
        )
        yields.to_sql("economic_data", conn, if_exists="append", index=False)
    finally:
        conn.close()
    return db_path


def test_spy_series_interpolates_daily():
    csv = dedent(
        """
        Date,SP500,Earnings
        2023-12-01,4500,150
        2024-01-01,4600,152
        2024-02-01,4700,155
        """
    )
    pe_series, price_series = pih._spy_series(
        pd.Timestamp("2023-12-28"),
        pd.Timestamp("2024-01-05"),
        fetch=lambda _url: csv,
    )
    assert pe_series.index[0] == pd.Timestamp("2023-12-28")
    assert pe_series.index[-1] == pd.Timestamp("2024-01-05")
    assert price_series.index[0] == pd.Timestamp("2023-12-28")
    assert price_series.index[-1] == pd.Timestamp("2024-01-05")
    # Values should interpolate between month endpoints.
    assert pe_series.loc["2023-12-28"] == pytest.approx(4500 / 150, rel=1e-2)
    assert price_series.loc["2024-01-05"] == pytest.approx(
        price_series.loc["2024-01-04"], rel=1e-3
    )


def test_qqq_series_uses_yearly_means():
    csv = dedent(
        """
        symbol,price_to_earnings_ratio_2017,price_to_earnings_ratio_2018,price_to_earnings_ratio_2022,price_to_earnings_ratio_latest
        AAA,20,22,30,35
        BBB,18,21,28,33
        """
    )
    series = pih._qqq_series(
        pd.Timestamp("2023-12-28"),
        pd.Timestamp("2024-01-05"),
        fetch=lambda _url: csv,
    )
    assert not series.empty
    assert series.index[0] == pd.Timestamp("2023-12-28")
    # The first portion of the series reflects the ``_latest`` estimate.
    assert series.iloc[0] == pytest.approx((35 + 33) / 2)


def test_qqq_price_series_from_daily_feed():
    csv = dedent(
        """
        Date,Open,High,Low,Close,Volume
        12/27/2023,470,471,469,470,100
        12/28/2023,471,472,470,471,120
        12/29/2023,472,473,471,472,130
        """
    )
    series = pih._qqq_price_series(
        pd.Timestamp("2023-12-28"),
        pd.Timestamp("2023-12-30"),
        fetch=lambda _url: csv,
    )
    assert list(series.index[[0, -1]]) == [pd.Timestamp("2023-12-28"), pd.Timestamp("2023-12-30")]
    # Weekend days should forward-fill the last close.
    assert series.loc["2023-12-30"] == pytest.approx(472.0)


def test_populate_index_history_inserts_rows(temp_db: Path):
    spy_csv = dedent(
        """
        Date,SP500,Earnings
        2023-12-01,4500,150
        2024-01-01,4600,152
        2024-02-01,4700,155
        """
    )
    qqq_csv = dedent(
        """
        symbol,price_to_earnings_ratio_2017,price_to_earnings_ratio_2018,price_to_earnings_ratio_2019,price_to_earnings_ratio_latest
        AAA,20,22,24,30
        BBB,18,21,23,29
        """
    )

    price_csv = dedent(
        """
        Date,Open,High,Low,Close,Volume
        12/27/2023,470,471,469,470,100
        12/28/2023,471,472,470,471,120
        12/29/2023,472,473,471,472,130
        12/30/2023,472,474,471,473,140
        12/31/2023,473,475,472,474,150
        01/01/2024,474,476,473,475,160
        01/02/2024,475,477,474,476,170
        01/03/2024,476,478,475,477,180
        01/04/2024,477,479,476,478,190
        01/05/2024,478,480,477,479,200
        """
    )

    fetch = _fake_fetch_factory(
        {
            pih.SPY_DATA_URL: spy_csv,
            pih.QQQ_PE_URL: qqq_csv,
            pih.QQQ_PRICE_URL: price_csv,
        }
    )

    pih.populate_index_history(
        db_path=str(temp_db),
        years=1,
        today=pd.Timestamp("2024-01-05"),
        fetch=fetch,
    )

    conn = sqlite3.connect(temp_db)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM Index_PE_History WHERE Ticker='SPY' AND PE_Type='TTM'"
        )
        spy_count = cur.fetchone()[0]
        assert spy_count > 0

        cur.execute(
            "SELECT COUNT(*) FROM Index_Growth_History WHERE Ticker='QQQ' AND Growth_Type='TTM'"
        )
        qqq_count = cur.fetchone()[0]
        assert qqq_count > 0

        cur.execute(
            "SELECT COUNT(*) FROM Implied_Growth_History WHERE ticker='SPY' AND growth_type='TTM'"
        )
        implied_count = cur.fetchone()[0]
        assert implied_count > 0

        cur.execute(
            "SELECT COUNT(*) FROM Index_EPS_History WHERE Ticker='QQQ' AND EPS_Type='TTM'"
        )
        eps_count = cur.fetchone()[0]
        assert eps_count > 0
    finally:
        conn.close()

import pathlib
import sqlite3
import sys
from unittest.mock import patch

import pandas as pd
import pandas.testing as pdt

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import index_growth_charts as igc


def test_render_index_growth_charts_scales_decimal_series():
    dates = pd.date_range("2024-03-31", periods=3, freq="QE-DEC")
    decimal_growth = pd.Series([0.10, 0.25, 0.40], index=dates)
    pe_series = pd.Series([15.0, 16.5, 14.2], index=dates)

    with (
        patch.object(igc, "sqlite3") as mock_sqlite,
        patch.object(igc, "_series_growth", return_value=decimal_growth) as mock_growth,
        patch.object(igc, "_series_pe", return_value=pe_series) as mock_pe,
        patch.object(igc, "_series_pe_monthly_derived", return_value=pd.Series(dtype=float)) as mock_pe_monthly,
        patch.object(igc, "_build_chart_block") as mock_block,
        patch.object(igc, "_write_chart_assets") as mock_write,
    ):
        fake_conn = object()
        mock_sqlite.connect.return_value.__enter__.return_value = fake_conn

        captured = []

        def capture(series, title, ylabel, percent_axis, x_range, callout_text=None, **kwargs):
            captured.append((series, title, ylabel, percent_axis, x_range, callout_text))
            return igc.ChartBlock(
                layout=igc.Div(text="test"),
                fig=None,
                source=None,
                log_axis=False,
                window_div=None,
                percent_axis=False,
                window_mode="ratio",
            )

        mock_block.side_effect = capture

        igc.render_index_growth_charts("TEST")

    # Ensure the growth chart received values scaled to the 0-100 range.
    assert captured, "Expected the chart helper to be invoked at least once"
    growth_series, _, growth_ylabel, percent_axis, x_range, callout = captured[0]
    pdt.assert_series_equal(growth_series, decimal_growth * 100, check_names=False)
    assert growth_ylabel == "Implied Growth Rate (%)"
    assert percent_axis is True
    assert x_range is not None
    assert callout is not None and "implied growth" in callout.lower()

    # Confirm the helper series functions were invoked with the mocked connection.
    mock_growth.assert_called_once_with(fake_conn, "TEST")
    mock_pe.assert_called_once_with(fake_conn, "TEST")
    mock_pe_monthly.assert_called_once_with(fake_conn, "TEST")
    assert mock_write.call_count == 4
    names = [call.args[1] for call in mock_write.call_args_list]
    assert names[0] == "valuation_bundle"


def test_render_index_growth_charts_keeps_daily_pe_series():
    daily_dates = pd.date_range("2024-01-01", periods=10, freq="D")
    monthly_dates = pd.date_range("2010-01-31", periods=3, freq="ME")
    daily_pe = pd.Series(range(10), index=daily_dates, dtype=float)
    monthly_eps = pd.Series([100.0, 105.0, 110.0], index=monthly_dates)
    monthly_pe = pd.Series([12.0, 13.0, 14.0], index=monthly_dates)

    with (
        patch.object(igc, "sqlite3") as mock_sqlite,
        patch.object(igc, "_extend_eps_csv"),
        patch.object(igc, "_series_growth", return_value=daily_pe) as mock_growth,
        patch.object(igc, "_series_pe", return_value=daily_pe) as mock_pe,
        patch.object(igc, "_series_pe_monthly_derived", return_value=monthly_pe) as mock_pe_monthly,
        patch.object(igc, "_series_eps", return_value=monthly_eps) as mock_eps,
        patch.object(igc, "_write_chart_assets") as mock_write,
    ):
        captured = []

        def capture(series, title, ylabel, percent_axis, x_range, callout_text=None, **kwargs):
            captured.append((title, series))
            return igc.ChartBlock(
                layout=igc.Div(text="test"),
                fig=None,
                source=None,
                log_axis=False,
                window_div=None,
                percent_axis=False,
                window_mode="ratio",
            )

        mock_block = patch.object(igc, "_build_chart_block", side_effect=capture)
        with mock_block:
            fake_conn = object()
            mock_sqlite.connect.return_value.__enter__.return_value = fake_conn
            igc.render_index_growth_charts("SPY")

    pe_series = next(series for title, series in captured if "P/E" in title)
    assert set(daily_pe.index).issubset(set(pe_series.index))
    assert monthly_dates.min() in pe_series.index
    mock_growth.assert_called_once()
    mock_pe.assert_called_once()
    mock_pe_monthly.assert_called_once()
    mock_eps.assert_called_once()
    assert mock_write.call_count == 4


def test_latest_forward_eps_reads_newest_row(tmp_path):
    db = tmp_path / "t.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            """CREATE TABLE Index_Forward_EPS_History (
                 date_recorded TEXT, ticker TEXT, forward_eps_etf REAL,
                 forward_eps_index REAL, forward_pe REAL, horizon_date TEXT,
                 source TEXT, PRIMARY KEY (date_recorded, ticker))""")
        conn.executemany(
            "INSERT INTO Index_Forward_EPS_History VALUES (?,?,?,?,?,?,?)",
            [("2026-06-01", "SPY", 23.0, 230.0, 20.0, "2027-06-01", "yfinance"),
             ("2026-06-28", "SPY", 25.0, 250.0, 22.0, "2027-06-28", "stockanalysis")])
        row = igc._latest_forward_eps(conn, "SPY")
    assert row["forward_eps_index"] == 250.0
    assert row["horizon_date"] == "2027-06-28"


def test_forward_eps_callout_text():
    txt = igc._forward_eps_callout(
        forward_eps_index=250.0, latest_hist_eps=230.0,
        horizon_date="2027-06-28", source="stockanalysis",
        forward_implied_growth=0.124)
    assert "8.7%" in txt           # 250/230 - 1 (expected EPS growth)
    assert "stockanalysis" in txt
    assert "$250" in txt
    assert "12.4%" in txt          # forward implied growth (valuation model)


def test_add_forward_eps_overlay_adds_renderers(tmp_path):
    from bokeh.plotting import figure as _figure
    fig = _figure(x_axis_type="datetime")
    before = len(fig.renderers)
    igc._add_forward_eps_overlay(
        fig,
        last_date=pd.Timestamp("2026-06-01"),
        last_eps=230.0,
        forward_date=pd.Timestamp("2027-06-01"),
        forward_eps_index=250.0,
    )
    # dashed connector line + a marker = 2 new renderers
    assert len(fig.renderers) == before + 2

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
    import index_forward_eps as ife
    db = tmp_path / "t.db"
    with sqlite3.connect(db) as conn:
        ife.ensure_forward_eps_table(conn)
        conn.executemany(
            """INSERT INTO Index_Forward_EPS_History
               (date_recorded, ticker, forward_eps_etf, forward_eps_index,
                forward_pe, horizon_date, source, displayable)
               VALUES (?,?,?,?,?,?,?,1)""",
            [("2026-06-01", "SPY", 23.0, 230.0, 20.0, "2027-06-01", "yfinance"),
             ("2026-06-28", "SPY", 25.0, 250.0, 22.0, "2027-06-28", "stockanalysis")])
        conn.commit()
        row = igc._latest_forward_eps(conn, "SPY")
    assert row["forward_eps_index"] == 250.0
    assert row["horizon_date"] == "2027-06-28"



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


def test_render_applies_forward_overlay_when_data_present():
    dates = pd.to_datetime(["2024-03-31", "2024-06-30", "2024-09-30"])
    eps = pd.Series([220.0, 225.0, 230.0], index=dates)
    fwd = {"forward_eps_index": 250.0, "horizon_date": "2027-06-28", "source": "bottom_up",
           "date_recorded": "2026-06-28", "growth_this_fy": 0.19, "growth_next_fy": 0.14,
           "coverage_weight": 0.93, "displayable": 1}

    captured = {}
    with (
        patch.object(igc, "sqlite3") as mock_sqlite,
        patch.object(igc, "_series_growth", return_value=pd.Series(dtype=float)),
        patch.object(igc, "_series_pe", return_value=pd.Series(dtype=float)),
        patch.object(igc, "_series_pe_monthly_derived", return_value=pd.Series(dtype=float)),
        patch.object(igc, "_series_eps", return_value=eps),
        patch.object(igc, "_latest_forward_eps", return_value=fwd),
        patch.object(igc, "_add_forward_eps_overlay") as mock_overlay,
        patch.object(igc, "_build_chart_block") as mock_block,
        patch.object(igc, "_write_chart_assets"),
        patch.object(igc, "_extend_eps_csv"),
    ):
        mock_sqlite.connect.return_value.__enter__.return_value = object()

        def capture(series, title, ylabel, percent_axis, x_range,
                    callout_text=None, **kwargs):
            captured.setdefault("titles", []).append(title)
            # Only capture the callout for the TTM EPS block (not the indexed panel)
            if "EPS" in title and "indexed" not in title.lower():
                captured["eps_callout"] = callout_text
            return igc.ChartBlock(layout=igc.Div(text="x"),
                                   fig=object(), source=None, log_axis=False,
                                   window_div=None, percent_axis=False,
                                   window_mode="ratio")
        mock_block.side_effect = capture
        igc.render_index_growth_charts("QQQ")

    assert mock_overlay.called
    assert "+19.0%" in captured["eps_callout"]   # growth_this_fy
    assert any("indexed" in t.lower() for t in captured.get("titles", [])), \
        f"Expected an 'indexed' titled block, got: {captured.get('titles', [])}"


def test_latest_forward_eps_respects_displayable(tmp_path):
    db = tmp_path/"t.db"
    with sqlite3.connect(db) as conn:
        import index_forward_eps as ife; ife.ensure_forward_eps_table(conn)
        conn.execute("""INSERT INTO Index_Forward_EPS_History
          (date_recorded,ticker,forward_eps_index,horizon_date,source,coverage_weight,
           growth_this_fy,growth_next_fy,method,displayable)
          VALUES ('2026-06-28','QQQ',250.0,'2027-06-28','bottom_up',0.93,0.19,0.14,'bottom_up',0)""")
        conn.commit()
        assert igc._latest_forward_eps(conn,"QQQ") is None
        conn.execute("UPDATE Index_Forward_EPS_History SET displayable=1")
        conn.commit()
        row = igc._latest_forward_eps(conn,"QQQ")
    assert row["growth_this_fy"]==0.19 and row["coverage_weight"]==0.93


def test_forward_eps_callout_bottomup():
    txt = igc._forward_eps_callout_bottomup(growth_this_fy=0.19, growth_next_fy=0.14,
            coverage_weight=0.93, horizon_date="2027-06-28")
    assert "+19.0%" in txt and "+14.0%" in txt and "93%" in txt


def test_clean_log_formatter_returns_formatter():
    from bokeh.models import CustomJSTickFormatter
    f_money = igc._clean_log_formatter(money=True)
    f_plain = igc._clean_log_formatter(money=False)
    assert isinstance(f_money, CustomJSTickFormatter)
    assert "$" in f_money.code
    assert isinstance(f_plain, CustomJSTickFormatter)


def test_indexed_series_rebases_to_100():
    s = pd.Series([50.0, 75.0, 100.0],
                  index=pd.to_datetime(["2024-01-01","2024-06-01","2024-12-01"]))
    out = igc._indexed_series(s)
    assert list(out.round(2)) == [100.0, 150.0, 200.0]

def test_indexed_series_guards_bad_base():
    assert igc._indexed_series(pd.Series(dtype=float)).empty
    s = pd.Series([0.0, 5.0], index=pd.to_datetime(["2024-01-01","2024-02-01"]))
    assert igc._indexed_series(s).empty


def test_attach_measure_tool_adds_widgets():
    from bokeh.plotting import figure as _fig
    from bokeh.models import ColumnDataSource, Button, Div, TapTool
    fig = _fig(x_axis_type="datetime")
    src = ColumnDataSource(data={"date":[1,2,3],"value":[10.0,12.0,15.0]})
    dots = fig.scatter("date","value", source=src)
    before = len(fig.renderers)
    div, btn = igc._attach_measure_tool(fig, src, dots, money=True)
    assert isinstance(div, Div) and isinstance(btn, Button)
    assert any(isinstance(t, TapTool) for t in fig.tools)
    assert len(fig.renderers) > before


def test_build_chart_block_measure_includes_widgets():
    s = pd.Series([10.0,12.0,15.0], index=pd.to_datetime(["2024-01-01","2024-06-01","2024-12-01"]))
    blk = igc._build_chart_block(s, "EPS", "EPS ($)", False, None,
                                 log_axis=True, measure=True, money=True)
    from bokeh.models import Button, Div
    found = {"button": False, "div_readout": False}
    def walk(m):
        if isinstance(m, Button): found["button"]=True
        if isinstance(m, Div) and "measure" in (m.text or "").lower(): found["div_readout"]=True
        for ch in getattr(m, "children", []): walk(ch)
    walk(blk.layout)
    assert found["button"] and found["div_readout"]

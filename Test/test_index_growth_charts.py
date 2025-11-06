import pathlib
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

    captured = []

    with (
        patch.object(igc, "sqlite3") as mock_sqlite,
        patch.object(igc, "_series_growth", return_value=decimal_growth) as mock_growth,
        patch.object(igc, "_series_pe", return_value=pe_series) as mock_pe,
        patch.object(igc, "_build_line_components") as mock_components,
        patch.object(igc, "_write_chart_assets") as mock_write,
        patch.object(igc, "_save_tables") as mock_save,
    ):
        fake_conn = object()
        mock_sqlite.connect.return_value.__enter__.return_value = fake_conn

        def capture(series, title, ylabel, percent_axis=False):
            captured.append((series, title, ylabel, percent_axis))
            return ("<script>", "<div></div>")

        mock_components.side_effect = capture

        igc.render_index_growth_charts("TEST")

    # Ensure the growth chart received values scaled to the 0-100 range.
    assert captured, "Expected the chart helper to be invoked at least once"
    growth_series, _, growth_ylabel, percent_axis = captured[0]
    pdt.assert_series_equal(growth_series, decimal_growth * 100, check_names=False)
    assert growth_ylabel == "Implied Growth Rate (%)"
    assert percent_axis is True

    # Confirm the helper series functions were invoked with the mocked connection.
    mock_growth.assert_called_once_with(fake_conn, "TEST")
    mock_pe.assert_called_once_with(fake_conn, "TEST")
    mock_write.assert_called()
    mock_save.assert_called_once()

#!/usr/bin/env python3
# index_growth_charts.py  –  FULL FILE  (v2025-07-14 g)
# -----------------------------------------------------------
# • Reads Implied_Growth from   Index_Growth_History
# • Reads P/E          from     Index_PE_History
# • Generates charts + tables under all legacy filenames
# -----------------------------------------------------------

import datetime as _dt
import os
import sqlite3

import numpy as np
import pandas as pd

from bokeh.embed import components
from bokeh.layouts import column
from bokeh.models import (
    Band,
    ColumnDataSource,
    CustomJS,
    DateRangeSlider,
    Div,
    HoverTool,
    NumeralTickFormatter,
    Range1d,
    Span,
)
from bokeh.plotting import figure

DB_PATH, OUT_DIR = "Stock Data.db", "charts"
os.makedirs(OUT_DIR, exist_ok=True)

# ───────── uniform CSS (blue frame + grey grid) ────────────
SUMMARY_CSS = """
<style>
.summary-table{border-collapse:collapse;
  font-family:Verdana,Arial,sans-serif;font-size:12px;
  border:3px solid #003366;margin:0 auto;width:auto;
  max-width:520px;}
.summary-table th{background:#f2f2f2;padding:4px 6px;
  border:1px solid #B0B0B0;text-align:center;white-space:nowrap;}
.summary-table td{padding:4px 6px;border:1px solid #B0B0B0;text-align:center;}
</style>
"""

YEARS = (1, 2, 3, 5, 10)

# ───────── helpers ─────────────────────────────────────────
def _series_growth(conn, tk):
    """Return Implied_Growth (TTM) series for ticker tk."""
    df = pd.read_sql(
        """SELECT Date, Implied_Growth AS val
             FROM Index_Growth_History
            WHERE Ticker=? AND Growth_Type='TTM'
         ORDER BY Date""", conn, params=(tk,))
    df["Date"] = pd.to_datetime(df["Date"])
    return pd.to_numeric(df.set_index("Date")["val"], errors="coerce").dropna()

def _series_pe(conn, tk):
    """Return PE_Ratio (TTM) series for ticker tk."""
    df = pd.read_sql(
        """SELECT Date, PE_Ratio AS val
             FROM Index_PE_History
            WHERE Ticker=? AND PE_Type='TTM'
         ORDER BY Date""", conn, params=(tk,))
    df["Date"] = pd.to_datetime(df["Date"])
    return pd.to_numeric(df.set_index("Date")["val"], errors="coerce").dropna()


def _series_eps(conn, tk):
    """Return EPS (TTM) series for ticker tk."""
    try:
        df = pd.read_sql(
            """SELECT Date, EPS AS val
                 FROM Index_EPS_History
                WHERE Ticker=? AND EPS_Type='TTM'
             ORDER BY Date""",
            conn,
            params=(tk,),
        )
    except AttributeError:  # pragma: no cover - allows mocked connections in tests
        return pd.Series(dtype=float)
    df["Date"] = pd.to_datetime(df["Date"])
    return pd.to_numeric(df.set_index("Date")["val"], errors="coerce").dropna()

def _pctile(s) -> str:                      # whole-number percentile
    """Return percentile rank of the latest value in *s* (1-99)."""
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return "—"
    val = s.iloc[-1]
    s_sorted = s.sort_values()
    rank = np.searchsorted(s_sorted.values, float(val), side="right")
    pct  = (rank / len(s_sorted)) * 100
    return str(int(round(max(1, min(99, pct)))))

def _pct_fmt(x: float) -> str:              # 0.1923 → '19.23 %'
    return f"{x * 100:.2f} %"

def _format_value(val, pct: bool) -> str:
    try:
        if pd.isna(val) or not np.isfinite(val):
            return "N/A"
    except TypeError:  # pragma: no cover - defensive, val may be str/object
        return "N/A"
    return _pct_fmt(val) if pct else f"{val:.2f}"


def _rows_by_years(series: pd.Series, pct: bool = False) -> pd.DataFrame:
    if series.empty:
        return pd.DataFrame(
            [
                dict(
                    Years=str(yrs),
                    Current="N/A",
                    Average="N/A",
                    Min="N/A",
                    Max="N/A",
                    Percentile="—",
                )
                for yrs in YEARS
            ]
        )

    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        return pd.DataFrame(
            [
                dict(
                    Years=str(yrs),
                    Current="N/A",
                    Average="N/A",
                    Min="N/A",
                    Max="N/A",
                    Percentile="—",
                )
                for yrs in YEARS
            ]
        )

    latest_val = series.iloc[-1]
    latest_fmt = _format_value(latest_val, pct)
    end = series.index.max()

    rows = []
    for yrs in YEARS:
        start = end - pd.DateOffset(years=yrs)
        window = series[series.index >= start].dropna()
        if window.empty:
            rows.append(
                dict(
                    Years=str(yrs),
                    Current="N/A",
                    Average="N/A",
                    Min="N/A",
                    Max="N/A",
                    Percentile="—",
                )
            )
            continue

        rows.append(
            dict(
                Years=str(yrs),
                Current=latest_fmt,
                Average=_format_value(window.mean(), pct),
                Min=_format_value(window.min(), pct),
                Max=_format_value(window.max(), pct),
                Percentile=_pctile(window),
            )
        )

    return pd.DataFrame(rows)

def _ten_year_window(series: pd.Series) -> pd.Series:
    """Return the slice of ``series`` covering the last ten years (or all data)."""

    if series.empty:
        return series

    end = series.index.max()
    start = end - pd.DateOffset(years=10)
    window = series[series.index >= start].dropna()
    return window if not window.empty else series


def _calc_hover_metrics(series: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (yoy_diff, yoy_pct, zscore) arrays aligned with ``series``."""

    if series.empty:
        empty = np.full(len(series), np.nan)
        return empty, empty, empty

    s = series.astype(float)
    yoy_periods = 252 if len(s) > 252 else max(len(s) - 1, 1)
    shifted = s.shift(yoy_periods)
    yoy_diff = (s - shifted).to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        yoy_pct = ((s / shifted) - 1.0).to_numpy(dtype=float) * 100.0

    mean = float(s.mean()) if len(s) else float("nan")
    std = float(s.std(ddof=0)) if len(s) else float("nan")
    if std and np.isfinite(std) and std > 0:
        zscore = ((s - mean) / std).to_numpy(dtype=float)
    else:
        zscore = np.full(len(s), np.nan)

    return yoy_diff, yoy_pct, zscore


def _summary_sentence(label: str, summary_df: pd.DataFrame) -> str:
    """Compose a brief textual takeaway from the 10-year summary row."""

    if summary_df.empty:
        return ""

    pick = summary_df[summary_df["Years"] == "10"]
    if pick.empty:
        pick = summary_df.iloc[[-1]]

    row = pick.iloc[0]
    current = row.get("Current", "N/A")
    percentile = row.get("Percentile", "—")
    avg = row.get("Average", "N/A")
    min_v = row.get("Min", "N/A")
    max_v = row.get("Max", "N/A")
    years = row.get("Years", "")

    pct_text = (
        f"the {percentile}th percentile"
        if percentile and str(percentile).strip("—")
        else "an unavailable percentile"
    )
    years_text = f"over the past {years} years" if years else "recent history"

    return (
        f"Current {label} is {current}, placing it in {pct_text} {years_text} "
        f"(avg {avg}, min {min_v}, max {max_v})."
    )


def _build_chart_block(
    series: pd.Series,
    title: str,
    ylab: str,
    percent_axis: bool,
    x_range: Range1d | None,
    callout_text: str | None = None,
):
    """Return a Bokeh layout block (figure + optional callout)."""

    if series is None or series.dropna().empty:
        placeholder = Div(
            text=(
                f"<div class=\"chart-card\"><h3>{title}</h3>"
                "<div class=\"chart-placeholder\">No data available.</div></div>"
            ),
            sizing_mode="stretch_width",
        )
        return placeholder

    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        placeholder = Div(
            text=(
                f"<div class=\"chart-card\"><h3>{title}</h3>"
                "<div class=\"chart-placeholder\">No data available.</div></div>"
            ),
            sizing_mode="stretch_width",
        )
        return placeholder

    values = series.astype(float)
    window = _ten_year_window(values)
    avg = float(window.mean()) if len(window) else float("nan")
    q25 = float(window.quantile(0.25)) if len(window) else float("nan")
    q75 = float(window.quantile(0.75)) if len(window) else float("nan")
    latest = float(values.iloc[-1])

    yoy_diff, yoy_pct, zscore = _calc_hover_metrics(values)

    def _clean(arr):
        cleaned = []
        for val in arr:
            if val is None or (isinstance(val, float) and not np.isfinite(val)):
                cleaned.append(None)
            else:
                cleaned.append(float(val))
        return cleaned

    source = ColumnDataSource(
        data={
            "date": values.index.to_pydatetime(),
            "value": values.values,
            "yoy_diff": _clean(yoy_diff),
            "yoy_pct": _clean(yoy_pct),
            "zscore": _clean(zscore),
            "avg": [avg] * len(values),
            "p25": [q25] * len(values),
            "p75": [q75] * len(values),
        }
    )

    fig = figure(
        title=title,
        x_axis_type="datetime",
        height=320,
        sizing_mode="stretch_width",
        toolbar_location="above",
        tools="pan,wheel_zoom,box_zoom,reset,save",
        x_range=x_range,
    )
    fig.line("date", "value", source=source, line_width=2, color="#1f77b4")
    fig.circle("date", "value", source=source, size=5, color="#1f77b4", alpha=0.65)

    band_source = ColumnDataSource(
        data={
            "date": values.index.to_pydatetime(),
            "lower": [q25] * len(values),
            "upper": [q75] * len(values),
        }
    )
    band = Band(
        base="date",
        lower="lower",
        upper="upper",
        source=band_source,
        level="underlay",
        fill_alpha=0.12,
        fill_color="#1f77b4",
    )
    fig.add_layout(band)

    avg_span = Span(location=avg, dimension="width", line_color="#555555", line_dash="dashed")
    current_span = Span(
        location=latest,
        dimension="width",
        line_color="#ff8800",
        line_dash="dotdash",
        line_width=2,
    )
    fig.add_layout(avg_span)
    fig.add_layout(current_span)

    value_tip = "@value{0.00}" + (" %" if percent_axis else "")
    yoy_diff_label = "@yoy_diff{0.00}" + (" pts" if percent_axis else "")
    yoy_pct_label = "@yoy_pct{0.00}%"
    avg_label = "@avg{0.00}" + (" %" if percent_axis else "")

    hover = HoverTool(
        tooltips=[
            ("Date", "@date{%F}"),
            ("Value", value_tip),
            ("YoY Δ", yoy_diff_label),
            ("YoY %", yoy_pct_label),
            ("Z-score", "@zscore{0.00}"),
            ("10y avg", avg_label),
        ],
        formatters={"@date": "datetime"},
        mode="vline",
    )
    fig.add_tools(hover)
    fig.toolbar.autohide = True

    fig.yaxis.axis_label = ylab
    fig.yaxis.formatter = NumeralTickFormatter(format="0.00")
    if percent_axis:
        fig.yaxis.formatter = NumeralTickFormatter(format="0.0")

    block_children = [fig]
    if callout_text:
        block_children.append(
            Div(
                text=f"<p><strong>Takeaway:</strong> {callout_text}</p>",
                sizing_mode="stretch_width",
                css_classes=["chart-callout"],
            )
        )

    return column(*block_children, sizing_mode="stretch_width", spacing=6)


def _write_chart_assets(tk_lower: str, name: str, components_pair):
    """Persist the Bokeh components for *name* (growth, pe, eps, ...)."""

    script_path = os.path.join(OUT_DIR, f"{tk_lower}_{name}_chart.js")
    div_path = os.path.join(OUT_DIR, f"{tk_lower}_{name}_chart_div.html")

    if not components_pair:
        script_text = ""
        div_text = "<div class=\"chart-placeholder\">No data available</div>"
    else:
        script_text, div_text = components_pair

    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script_text)

    with open(div_path, "w", encoding="utf-8") as f:
        f.write(div_text)

def _pct_color(v):                          # green ≤30, red ≥70
    try:
        v=float(v)
        if v<=30: return "color:#008800;font-weight:bold"
        if v>=70: return "color:#CC0000;font-weight:bold"
    except: pass
    return ""

def _build_html(df):
    sty = (df.style
             .hide(axis="index")
             .applymap(_pct_color, subset="Percentile")
             .set_table_attributes('class="summary-table"'))
    return SUMMARY_CSS + sty.to_html()

def _save_tables(tk, ig_df, pe_df):
    tk_lower = tk.lower()
    files = {
        f"{tk_lower}_growth_summary.html": _build_html(ig_df),
        f"{tk_lower}_pe_summary.html":     _build_html(pe_df),
    }
    for name, html in files.items():
        with open(os.path.join(OUT_DIR, name), "w", encoding="utf-8") as f:
            f.write(html)

# ───────── callable entry-point / mini-main ────────────────
def render_index_growth_charts(tk="SPY"):
    with sqlite3.connect(DB_PATH) as conn:
        ig_s = _series_growth(conn, tk)
        pe_s = _series_pe(conn, tk)
        eps_s = _series_eps(conn, tk)

    ig_plot = ig_s.copy()
    ig_ylabel = "Implied Growth Rate"
    ig_percent_axis = False
    if not ig_s.empty:
        med = ig_s.median(skipna=True)
        max_abs = ig_s.abs().max()
        if (
            pd.notna(med)
            and np.isfinite(med)
            and abs(med) < 1
            and pd.notna(max_abs)
            and np.isfinite(max_abs)
            and max_abs <= 2
        ):
            ig_plot = ig_s * 100
            ig_ylabel = "Implied Growth Rate (%)"
            ig_percent_axis = True

    ig_summary = _rows_by_years(ig_s, pct=True)
    pe_summary = _rows_by_years(pe_s, pct=False)
    _save_tables(tk, ig_summary, pe_summary)

    tk_lower = tk.lower()

    non_empty_series = [
        s.dropna()
        for s in (ig_plot, pe_s, eps_s)
        if isinstance(s, pd.Series) and not s.dropna().empty
    ]

    common_range = None
    slider = None
    if non_empty_series:
        min_date = min(s.index.min() for s in non_empty_series)
        max_date = max(s.index.max() for s in non_empty_series)
        start_dt = min_date.to_pydatetime()
        end_dt = max_date.to_pydatetime()
        common_range = Range1d(start=start_dt, end=end_dt)

        default_start = end_dt - _dt.timedelta(days=365 * 5)
        if default_start < start_dt:
            default_start = start_dt
        common_range.start = default_start

        slider = DateRangeSlider(
            title="Focus window",
            start=start_dt,
            end=end_dt,
            value=(default_start, end_dt),
            sizing_mode="stretch_width",
            format="%b %Y",
        )
        slider.js_on_change(
            "value",
            CustomJS(
                args={"rng": common_range},
                code="const [start, end] = cb_obj.value; rng.start = start; rng.end = end;",
            ),
        )

    blocks = []
    growth_callout = _summary_sentence("implied growth", ig_summary)
    blocks.append(
        _build_chart_block(
            ig_plot,
            f"{tk} Implied Growth (TTM)",
            ig_ylabel,
            ig_percent_axis,
            common_range,
            callout_text=growth_callout,
        )
    )

    pe_callout = _summary_sentence("P/E ratio", pe_summary)
    blocks.append(
        _build_chart_block(
            pe_s,
            f"{tk} P/E Ratio",
            "P/E",
            False,
            common_range,
            callout_text=pe_callout,
        )
    )

    blocks.append(
        _build_chart_block(
            eps_s,
            f"{tk} EPS (TTM)",
            "EPS ($)",
            False,
            common_range,
        )
    )

    layout_children = []
    if slider is not None:
        slider.css_classes = ["chart-range-slider"]
        layout_children.append(slider)
    layout_children.extend(blocks)

    if layout_children:
        layout = column(*layout_children, sizing_mode="stretch_width", spacing=18)
        valuation_components = components(layout, wrap_script=False)
    else:
        placeholder = Div(
            text="<div class=\"chart-placeholder\">No chart data available.</div>",
            sizing_mode="stretch_width",
        )
        valuation_components = components(placeholder, wrap_script=False)

    _write_chart_assets(tk_lower, "valuation_bundle", valuation_components)

    placeholder_div = (
        "<div class=\"chart-placeholder\">This view now lives inside the combined "
        "valuation dashboard above.</div>"
    )
    for legacy in ("growth", "pe", "eps"):
        _write_chart_assets(tk_lower, legacy, ("", placeholder_div))

# legacy alias
mini_main = render_index_growth_charts

if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    render_index_growth_charts(ticker)
    print("Tables & charts generated for", ticker)

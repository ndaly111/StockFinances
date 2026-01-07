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
from dataclasses import dataclass

import numpy as np
import pandas as pd

from bokeh.events import DocumentReady
from bokeh.embed import components
from bokeh.layouts import column, row
from bokeh.models import (
    Band,
    BoxZoomTool,
    Button,
    Circle,
    ColumnDataSource,
    CrosshairTool,
    CustomJS,
    DateRangeSlider,
    Div,
    HoverTool,
    NumeralTickFormatter,
    PanTool,
    Range1d,
    RangeTool,
    Span,
    Toggle,
    WheelZoomTool,
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


@dataclass
class ChartBlock:
    layout: object
    fig: object | None
    source: ColumnDataSource | None
    log_axis: bool
    window_div: Div | None
    percent_axis: bool
    window_mode: str

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


def _eps_growth_by_horizon(series: pd.Series) -> pd.DataFrame:
    horizons = [1, 2, 3, 5, 10]
    if series.empty:
        return pd.DataFrame(
            [
                dict(
                    Years=str(yrs),
                    **{
                        "Start Date": "N/A",
                        "Start EPS": "N/A",
                        "End EPS": "N/A",
                        "Total Change %": "N/A",
                        "CAGR %": "N/A",
                    },
                )
                for yrs in horizons
            ]
        )

    end_date = series.index.max()
    end_eps = float(series.loc[end_date])
    rows = []
    for yrs in horizons:
        start_date = end_date - pd.DateOffset(years=yrs)
        start_window = series[series.index <= start_date]
        if start_window.empty:
            rows.append(
                dict(
                    Years=str(yrs),
                    **{
                        "Start Date": "N/A",
                        "Start EPS": "N/A",
                        "End EPS": f"{end_eps:.2f}",
                        "Total Change %": "N/A",
                        "CAGR %": "N/A",
                    },
                )
            )
            continue

        start_idx = start_window.index.max()
        start_eps = float(series.loc[start_idx])
        if start_eps <= 0 or end_eps <= 0:
            total_change = "N/A"
            cagr = "N/A"
        else:
            total_change_val = (end_eps / start_eps - 1.0) * 100.0
            years = max((end_date - start_idx).days / 365.25, 1e-6)
            cagr_val = (end_eps / start_eps) ** (1 / years) - 1.0
            total_change = f"{total_change_val:.2f} %"
            cagr = f"{cagr_val * 100:.2f} %"

        rows.append(
            dict(
                Years=str(yrs),
                **{
                    "Start Date": start_idx.strftime("%Y-%m-%d"),
                    "Start EPS": f"{start_eps:.2f}",
                    "End EPS": f"{end_eps:.2f}",
                    "Total Change %": total_change,
                    "CAGR %": cagr,
                },
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


def _calc_hover_metrics(series: pd.Series) -> np.ndarray:
    """Return YoY % changes aligned with ``series`` for hover text."""

    if series.empty:
        return np.full(len(series), np.nan)

    s = series.astype(float)
    yoy_periods = 252 if len(s) > 252 else max(len(s) - 1, 1)
    shifted = s.shift(yoy_periods)
    with np.errstate(divide="ignore", invalid="ignore"):
        yoy_pct = ((s / shifted) - 1.0).to_numpy(dtype=float) * 100.0

    return yoy_pct


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
    log_axis: bool = False,
    table_html: str | None = None,
    window_div: Div | None = None,
    window_mode: str = "ratio",
):
    """Return a Bokeh layout block (figure + optional callout/table)."""

    if series is None or series.dropna().empty:
        placeholder = Div(
            text=(
                f"<div class=\"metric-card\"><h3>{title}</h3>"
                "<div class=\"chart-placeholder\">No data available.</div></div>"
            ),
            sizing_mode="stretch_width",
        )
        return ChartBlock(
            layout=placeholder,
            fig=None,
            source=None,
            log_axis=log_axis,
            window_div=window_div,
            percent_axis=percent_axis,
            window_mode=window_mode,
        )

    series = pd.to_numeric(series, errors="coerce").dropna()
    if log_axis:
        series = series[series > 0]
    if series.empty:
        placeholder = Div(
            text=(
                f"<div class=\"metric-card\"><h3>{title}</h3>"
                "<div class=\"chart-placeholder\">No data available.</div></div>"
            ),
            sizing_mode="stretch_width",
        )
        return ChartBlock(
            layout=placeholder,
            fig=None,
            source=None,
            log_axis=log_axis,
            window_div=window_div,
            percent_axis=percent_axis,
            window_mode=window_mode,
        )

    values = series.astype(float)
    window = _ten_year_window(values)
    avg = float(window.mean()) if len(window) else float("nan")
    q25 = float(window.quantile(0.25)) if len(window) else float("nan")
    q75 = float(window.quantile(0.75)) if len(window) else float("nan")
    latest = float(values.iloc[-1])

    yoy_pct = _calc_hover_metrics(values)

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
            "yoy_pct": _clean(yoy_pct),
            "avg": [avg] * len(values),
            "p25": [q25] * len(values),
            "p75": [q75] * len(values),
        }
    )

    fig = figure(
        title=None,
        x_axis_type="datetime",
        y_axis_type="log" if log_axis else "linear",
        height=340,
        sizing_mode="stretch_width",
        toolbar_location="above",
        tools="",
        x_range=x_range,
    )

    pan_tool = PanTool(dimensions="width")
    wheel_zoom = WheelZoomTool(dimensions="width")
    box_zoom = BoxZoomTool(dimensions="width")
    crosshair = CrosshairTool(dimensions="both")
    hover = HoverTool(
        tooltips=[
            ("Date", "@date{%F}"),
            ("Value", "@value{0.00}" + (" %" if percent_axis else "")),
            ("YoY %", "@yoy_pct{0.00}%"),
        ],
        formatters={"@date": "datetime"},
        mode="vline",
    )
    fig.add_tools(pan_tool, wheel_zoom, box_zoom, hover, crosshair)
    fig.toolbar.autohide = True
    fig.toolbar.active_scroll = wheel_zoom
    fig.toolbar.active_drag = pan_tool

    fig.line("date", "value", source=source, line_width=2, color="#1f77b4")
    dots = fig.circle("date", "value", source=source, size=7, color="#1f77b4", alpha=0)
    dots.hover_glyph = Circle(
        x="date",
        y="value",
        radius=4.5,
        radius_units="screen",
        fill_color="#1f77b4",
        line_color="#1f77b4",
        fill_alpha=0.9,
        line_alpha=0.9,
    )

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

    fig.yaxis.axis_label = ylab
    fig.yaxis.formatter = NumeralTickFormatter(format="0.00")
    if percent_axis:
        fig.yaxis.formatter = NumeralTickFormatter(format="0.0")

    if window_div is None:
        window_div = Div(text="", sizing_mode="stretch_width", css_classes=["metric-meta"])

    block_children = [Div(text=f"<h3>{title}</h3>", sizing_mode="stretch_width"), fig]
    if callout_text:
        block_children.append(
            Div(
                text=f"<div class='metric-meta'>{callout_text}</div>",
                sizing_mode="stretch_width",
            )
        )
    if window_div:
        block_children.append(window_div)
    if table_html:
        block_children.append(
            Div(
                text=f"<div class='table-wrap'>{table_html}</div>",
                sizing_mode="stretch_width",
            )
        )

    layout = column(*block_children, sizing_mode="stretch_width", spacing=8, css_classes=["metric-card"])
    return ChartBlock(
        layout=layout,
        fig=fig,
        source=source,
        log_axis=log_axis,
        window_div=window_div,
        percent_axis=percent_axis,
        window_mode=window_mode,
    )


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
    sty = df.style.hide(axis="index").set_table_attributes('class="summary-table"')
    if "Percentile" in df.columns:
        sty = sty.applymap(_pct_color, subset=pd.IndexSlice[:, ["Percentile"]])
    return SUMMARY_CSS + sty.to_html()


def _write_legacy_tables(tk_lower: str, ig_table_html: str, pe_table_html: str, eps_table_html: str):
    """Persist legacy summary HTML fragments used by older templates."""

    legacy_paths = {
        f"{tk_lower}_growth_summary.html": ig_table_html,
        f"{tk_lower}_pe_summary.html": pe_table_html,
        f"{tk_lower}_eps_summary.html": eps_table_html,
        f"{tk_lower}_implied_growth_summary.html": ig_table_html,
    }

    for filename, html in legacy_paths.items():
        with open(os.path.join(OUT_DIR, filename), "w", encoding="utf-8") as f:
            f.write(html)

# ───────── callable entry-point / mini-main ────────────────
def render_index_growth_charts(tk="SPY"):
    with sqlite3.connect(DB_PATH) as conn:
        ig_s = _series_growth(conn, tk)
        pe_s = _series_pe(conn, tk)
        eps_s = _series_eps(conn, tk)

    eps_s = pd.to_numeric(eps_s, errors="coerce")
    eps_s = eps_s[eps_s > 0].dropna()

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
    eps_snapshot = _rows_by_years(eps_s, pct=False)
    eps_growth = _eps_growth_by_horizon(eps_s)

    ig_table_html = _build_html(ig_summary)
    pe_table_html = _build_html(pe_summary)
    eps_table_html = (
        "<div class='metric-meta'><strong>EPS snapshot</strong></div>" + _build_html(eps_snapshot)
        + "<div class='metric-meta' style='margin-top:12px;'><strong>EPS growth</strong></div>" + _build_html(eps_growth)
    )

    tk_lower = tk.lower()

    _write_legacy_tables(tk_lower, ig_table_html, pe_table_html, eps_table_html)

    non_empty_series = [
        s.dropna()
        for s in (ig_plot, pe_s, eps_s)
        if isinstance(s, pd.Series) and not s.dropna().empty
    ]

    common_range = None
    slider = None
    preset_buttons: list[Button] = []
    auto_toggle = Toggle(label="Auto Y", active=True, button_type="success")
    callback = None
    min_date_ms = max_date_ms = None
    range_nav = None
    if non_empty_series:
        min_date = min(s.index.min() for s in non_empty_series)
        max_date = max(s.index.max() for s in non_empty_series)
        start_dt = min_date.to_pydatetime()
        end_dt = max_date.to_pydatetime()
        common_range = Range1d(start=start_dt, end=end_dt)

        min_date_ms = int(start_dt.timestamp() * 1000)
        max_date_ms = int(end_dt.timestamp() * 1000)

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
        slider.css_classes = ["chart-range-slider"]
        slider.js_on_change(
            "value",
            CustomJS(
                args={"rng": common_range},
                code="""
                    const [start, end] = cb_obj.value;
                    rng.start = Number(start);
                    rng.end = Number(end);
                """,
            ),
        )

        preset_spans = [("1Y", 1), ("3Y", 3), ("5Y", 5), ("10Y", 10), ("MAX", None)]
        for label, yrs in preset_spans:
            span_ms = None if yrs is None else int(yrs * 365.25 * 24 * 60 * 60 * 1000)
            btn = Button(label=label, button_type="primary")
            args = {
                "rng": common_range,
                "min_ms": min_date_ms,
                "max_ms": max_date_ms,
                "span_ms": span_ms,
                "slider": slider,
            }
            btn.js_on_event(
                "button_click",
                CustomJS(
                    args=args,
                    code="""
                    const end = max_ms;
                    let start = span_ms ? Math.max(min_ms, end - span_ms) : min_ms;
                    rng.start = start;
                    rng.end = end;
                    if (slider){ slider.value = [start, end]; }
                """,
                ),
            )
            preset_buttons.append(btn)

        navigator_source_series = next((s for s in (ig_plot, pe_s, eps_s) if not s.empty), None)
        if navigator_source_series is not None:
            nav_src = ColumnDataSource(
                data={"date": navigator_source_series.index.to_pydatetime(), "value": navigator_source_series.values}
            )
            range_nav = figure(
                height=140,
                x_axis_type="datetime",
                y_axis_type="linear",
                sizing_mode="stretch_width",
                toolbar_location=None,
                tools="",
            )
            range_nav.line("date", "value", source=nav_src, line_width=2, color="#7c8aa5")
            rt = RangeTool(x_range=common_range)
            rt.overlay.fill_color = "#1f77b4"
            rt.overlay.fill_alpha = 0.2
            range_nav.add_tools(rt)
            range_nav.ygrid.grid_line_color = None
            range_nav.xaxis.axis_label = "Time Navigator"

    blocks = []
    growth_callout = _summary_sentence("implied growth", ig_summary)
    ig_window_div = Div(text="", sizing_mode="stretch_width", css_classes=["metric-meta"])
    blocks.append(
        _build_chart_block(
            ig_plot,
            f"{tk} Implied Growth (TTM)",
            ig_ylabel,
            ig_percent_axis,
            common_range,
            callout_text=growth_callout,
            table_html=ig_table_html,
            window_div=ig_window_div,
            window_mode="rate",
        )
    )

    pe_callout = _summary_sentence("P/E ratio", pe_summary)
    pe_window_div = Div(text="", sizing_mode="stretch_width", css_classes=["metric-meta"])
    blocks.append(
        _build_chart_block(
            pe_s,
            f"{tk} P/E Ratio",
            "P/E",
            False,
            common_range,
            callout_text=pe_callout,
            table_html=pe_table_html,
            window_div=pe_window_div,
            window_mode="rate",
        )
    )

    eps_window_div = Div(text="", sizing_mode="stretch_width", css_classes=["metric-meta"])
    blocks.append(
        _build_chart_block(
            eps_s,
            f"{tk} EPS (TTM)",
            "EPS ($)",
            False,
            common_range,
            log_axis=True,
            table_html=eps_table_html,
            window_div=eps_window_div,
        )
    )

    chart_refs = [b for b in blocks if b.fig is not None and b.source is not None]
    if common_range and chart_refs:
        callback = CustomJS(
            args={
                "xrange": common_range,
                "figs": [b.fig for b in chart_refs],
                "sources": [b.source for b in chart_refs],
                "logs": [b.log_axis for b in chart_refs],
                "window_divs": [b.window_div for b in chart_refs],
                "auto_toggle": auto_toggle,
                "percent_axes": [b.percent_axis for b in chart_refs],
                "window_modes": [b.window_mode for b in chart_refs],
            },
            code="""
                const startMs = Number(xrange.start);
                const endMs = Number(xrange.end);
                const dayMs = 1000 * 60 * 60 * 24;

                for (let i = 0; i < figs.length; i++) {
                    const fig = figs[i];
                    const source = sources[i];
                    const windowDiv = window_divs[i];
                    const logAxis = logs[i];
                    const percentAxis = percent_axes[i];
                    const windowMode = window_modes[i];
                    const dates = source.data.date;
                    const values = source.data.value;

                    let ymin = Number.POSITIVE_INFINITY;
                    let ymax = Number.NEGATIVE_INFINITY;
                    let firstVal = null;
                    let lastVal = null;
                    let firstDate = null;
                    let lastDate = null;

                    for (let j = 0; j < dates.length; j++) {
                        const d = Number(dates[j]);
                        const v = values[j];
                        if (d < startMs || d > endMs || v == null || !isFinite(v)) {
                            continue;
                        }
                        if (logAxis && v <= 0) {
                            continue;
                        }
                        if (firstDate === null || d < firstDate) {
                            firstDate = d;
                            firstVal = v;
                        }
                        if (lastDate === null || d > lastDate) {
                            lastDate = d;
                            lastVal = v;
                        }
                        ymin = Math.min(ymin, v);
                        ymax = Math.max(ymax, v);
                    }

                    const hasPoints = isFinite(ymin) && isFinite(ymax);

                    if (auto_toggle.active && hasPoints) {
                        if (logAxis) {
                            fig.y_range.start = ymin * 0.93;
                            fig.y_range.end = ymax * 1.07;
                        } else {
                            if (ymax === ymin) {
                                const pad = Math.abs(ymax) * 0.07 || 1;
                                fig.y_range.start = ymin - pad;
                                fig.y_range.end = ymax + pad;
                            } else {
                                const pad = (ymax - ymin) * 0.07;
                                fig.y_range.start = ymin - pad;
                                fig.y_range.end = ymax + pad;
                            }
                        }
                    }

                    if (windowDiv) {
                        if (!hasPoints || firstDate === null || lastDate === null || firstVal === null || lastVal === null) {
                            windowDiv.text = "";
                        } else {
                            const formatter = new Intl.NumberFormat('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
                            const startText = new Date(firstDate).toISOString().slice(0,10);
                            const endText = new Date(lastDate).toISOString().slice(0,10);

                            if (windowMode === "rate") {
                                const delta = lastVal - firstVal;
                                const startValText = `${formatter.format(firstVal)}${percentAxis ? ' %' : ''}`;
                                const endValText = `${formatter.format(lastVal)}${percentAxis ? ' %' : ''}`;
                                const deltaText = `${delta >= 0 ? '+' : ''}${formatter.format(delta)}${percentAxis ? ' pts' : ''}`;
                                windowDiv.text = `Window: ${startText} → ${endText} — ${startValText} → ${endValText} (Δ ${deltaText})`;
                            } else {
                                const pct = firstVal !== 0 ? (lastVal / firstVal - 1) * 100 : NaN;
                                const years = Math.max((lastDate - firstDate) / (365.25 * dayMs), 1e-6);
                                const cagr = firstVal > 0 && lastVal > 0 ? (Math.pow(lastVal / firstVal, 1 / years) - 1) * 100 : NaN;
                                const pctText = isFinite(pct) ? `${formatter.format(pct)} %` : 'N/A';
                                const cagrText = isFinite(cagr) ? `${formatter.format(cagr)} %` : 'N/A';
                                windowDiv.text = `Window: ${startText} → ${endText} — Δ ${pctText} (CAGR ${cagrText})`;
                            }
                        }
                    }
                }
            """,
        )
        common_range.js_on_change("start", callback)
        common_range.js_on_change("end", callback)
        auto_toggle.js_on_change("active", callback)
        if slider is not None:
            slider.js_on_change("value", callback)

    layout_children = []
    if range_nav is not None:
        layout_children.append(range_nav)
    if preset_buttons:
        control_items = [*preset_buttons, auto_toggle]
        controls_row = row(*control_items, sizing_mode="stretch_width", css_classes=["control-bar"])
        layout_children.append(controls_row)
    if slider is not None:
        slider_row = row(slider, sizing_mode="stretch_width", css_classes=["control-bar"])
        layout_children.append(slider_row)
    layout_children.extend([b.layout for b in blocks])

    if layout_children:
        layout = column(*layout_children, sizing_mode="stretch_width", spacing=18)
        layout.css_classes = ["metric-cards-root"]
        if callback is not None:
            layout.js_on_event(DocumentReady, callback)
        valuation_components = components(layout, wrap_script=False)
    else:
        placeholder = Div(
            text="<div class=\"chart-placeholder\">No chart data available.</div>",
            sizing_mode="stretch_width",
            css_classes=["metric-cards-root"],
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

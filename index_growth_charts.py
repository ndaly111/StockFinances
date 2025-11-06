#!/usr/bin/env python3
# index_growth_charts.py  –  FULL FILE  (v2025-07-14 g)
# -----------------------------------------------------------
# • Reads Implied_Growth from   Index_Growth_History
# • Reads P/E          from     Index_PE_History
# • Generates charts + tables under all legacy filenames
# -----------------------------------------------------------

import os, sqlite3, pandas as pd, numpy as np

from bokeh.embed import components
from bokeh.models import ColumnDataSource, HoverTool, NumeralTickFormatter
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

def _build_line_components(
    series: pd.Series,
    title: str,
    ylab: str,
    percent_axis: bool = False,
):
    """Return the (script, div) pair for a Bokeh line chart or ``None``."""

    if series is None or series.empty:
        return None

    source = ColumnDataSource(
        data={
            "date": series.index.to_pydatetime(),
            "value": series.astype(float).values,
        }
    )

    fig = figure(
        title=title,
        x_axis_type="datetime",
        height=320,
        sizing_mode="stretch_width",
        toolbar_location="above",
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    fig.line("date", "value", source=source, line_width=2, color="#1f77b4")
    fig.circle("date", "value", source=source, size=5, color="#1f77b4", alpha=0.65)

    hover = HoverTool(
        tooltips=[
            ("Date", "@date{%F}"),
            (
                "Value",
                "@value{0.00}" + (" %" if percent_axis else ""),
            ),
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

    script, div = components(fig, wrap_script=False)
    return script, div


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
             .map(_pct_color, subset="Percentile")
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

    ig_plot = ig_s
    ig_ylabel = "Implied Growth Rate"
    if not ig_s.empty:
        med = ig_s.median(skipna=True)
        max_abs = ig_s.abs().max()
        if (
            pd.notna(med) and np.isfinite(med)
            and abs(med) < 1
            and pd.notna(max_abs) and np.isfinite(max_abs)
            and max_abs <= 2
        ):
            # Stored as decimals (e.g., 0.18 for 18%) → scale a copy for plotting.
            ig_plot = ig_s * 100
            ig_ylabel = "Implied Growth Rate (%)"

    tk_lower = tk.lower()

    ig_components = _build_line_components(
        ig_plot,
        f"{tk} Implied Growth (TTM)",
        ig_ylabel,
        percent_axis="%" in ig_ylabel,
    )
    pe_components = _build_line_components(
        pe_s,
        f"{tk} P/E Ratio",
        "P/E",
        percent_axis=False,
    )
    eps_components = _build_line_components(
        eps_s,
        f"{tk} EPS (TTM)",
        "EPS ($)",
        percent_axis=False,
    )

    _write_chart_assets(tk_lower, "growth", ig_components)
    _write_chart_assets(tk_lower, "pe", pe_components)
    _write_chart_assets(tk_lower, "eps", eps_components)

    _save_tables(
        tk,
        _rows_by_years(ig_s, pct=True),
        _rows_by_years(pe_s, pct=False)
    )

# legacy alias
mini_main = render_index_growth_charts

if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    render_index_growth_charts(ticker)
    print("Tables & charts generated for", ticker)

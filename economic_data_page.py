#!/usr/bin/env python3
# economic_data_page.py – rev 09-Feb-2026
# Full-page economic dashboard: retro.css theme, recession shading,
# reference lines, percentile callouts, grouped sections, summary table
# -------------------------------------------------------------------
import sqlite3, numpy as np, pandas as pd, plotly.graph_objects as go
from pathlib import Path
from datetime import datetime

DB_PATH  = "Stock Data.db"
HTML_OUT = Path("economic_charts.html")

# ───────────── NBER recession periods (post-WWII) ─────────────
NBER_RECESSIONS = [
    ("1948-11-01", "1949-10-01"),
    ("1953-07-01", "1954-05-01"),
    ("1957-08-01", "1958-04-01"),
    ("1960-04-01", "1961-02-01"),
    ("1969-12-01", "1970-11-01"),
    ("1973-11-01", "1975-03-01"),
    ("1980-01-01", "1980-07-01"),
    ("1981-07-01", "1982-11-01"),
    ("1990-07-01", "1991-03-01"),
    ("2001-03-01", "2001-11-01"),
    ("2007-12-01", "2009-06-01"),
    ("2020-02-01", "2020-04-01"),
]

# ───────────── section layout ─────────────
SECTION_ORDER = [
    ("Labor & Prices", ["UNRATE", "CPIAUCSL", "PCEPI", "ICSA", "UMCSENT"]),
    ("Rates & Growth", ["DGS10", "DGS2", "T10Y2Y", "GDPC1", "FEDFUNDS"]),
]

# Reference lines per indicator
_REF_LINES = {
    "CPIAUCSL": [dict(y=2, label="2% Target", color="#CC0000", dash="dash")],
    "PCEPI":    [dict(y=2, label="2% Target", color="#CC0000", dash="dash")],
    "UNRATE":   [dict(y=4.0, y2=5.0, label="NAIRU Band", color="rgba(0,80,0,0.10)")],
    "T10Y2Y":   [dict(y=0, label="Inversion Line", color="#CC0000", dash="dash")],
}

# ───────────── retro chart styling ─────────────
_CHART_LAYOUT = dict(
    paper_bgcolor="#F0F0FF",
    plot_bgcolor="#FFFFFF",
    font=dict(family="Verdana, Geneva, sans-serif", color="#000080", size=12),
    margin=dict(l=50, r=20, t=30, b=40),
    height=370,
    xaxis=dict(
        gridcolor="#E0E0FF",
        rangeselector=dict(
            bgcolor="#E0E0E0",
            bordercolor="#000080",
            font=dict(size=11),
            buttons=[
                dict(count=6,  label="6M",  step="month", stepmode="backward"),
                dict(count=1,  label="1Y",  step="year",  stepmode="backward"),
                dict(count=3,  label="3Y",  step="year",  stepmode="backward"),
                dict(count=5,  label="5Y",  step="year",  stepmode="backward"),
                dict(count=10, label="10Y", step="year",  stepmode="backward"),
                dict(step="all", label="All"),
            ],
        ),
        rangeslider=dict(visible=False),
        type="date",
    ),
    yaxis=dict(gridcolor="#E0E0FF"),
)

# ───────────── helper functions ─────────────
def _get_series(conn, sid):
    """Read an indicator from the DB. For FEDFUNDS, compute midpoint of target range."""
    if sid == "FEDFUNDS":
        lo = pd.read_sql(
            "SELECT substr(date,1,10) AS date, value FROM economic_data "
            "WHERE indicator='DFEDTARL' ORDER BY date", conn)
        hi = pd.read_sql(
            "SELECT substr(date,1,10) AS date, value FROM economic_data "
            "WHERE indicator='DFEDTARU' ORDER BY date", conn)
        if lo.empty or hi.empty:
            return pd.DataFrame(columns=["date", "value"])
        merged = lo.merge(hi, on="date", suffixes=("_lo", "_hi"))
        merged["value"] = (merged["value_lo"] + merged["value_hi"]) / 2.0
        return merged[["date", "value"]]
    return pd.read_sql(
        "SELECT substr(date,1,10) AS date, value FROM economic_data "
        "WHERE indicator=? ORDER BY substr(date,1,10)", conn, params=(sid,))


def _pctile(series: pd.Series) -> int:
    """Percentile rank (1-99) of the latest value in the full series."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return 50
    val = float(s.iloc[-1])
    rank = np.searchsorted(np.sort(s.values), val, side="right")
    pct = (rank / len(s)) * 100
    return int(round(max(1, min(99, pct))))


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _stats(df, sid=None):
    """Compute latest value, date-based 1-month and 1-year deltas."""
    if df.empty:
        return dict(latest="—", date="—", mchg="—", ychg="—")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").dropna(subset=["value"])
    if df.empty:
        return dict(latest="—", date="—", mchg="—", ychg="—")

    last_row = df.iloc[-1]
    latest_val = float(last_row["value"])
    last_date = last_row["date"]
    date_str = str(last_date.date())

    # date-based lookbacks
    target_1m = last_date - pd.DateOffset(months=1)
    target_1y = last_date - pd.DateOffset(years=1)
    prev_1m = df.loc[df["date"] <= target_1m]
    prev_1y = df.loc[df["date"] <= target_1y]
    val_1m = float(prev_1m.iloc[-1]["value"]) if not prev_1m.empty else None
    val_1y = float(prev_1y.iloc[-1]["value"]) if not prev_1y.empty else None

    def _delta(cur, prev, fmt_type):
        if prev is None:
            return "—"
        diff = cur - prev
        if fmt_type == "pp":
            return f"{diff:+.2f} pp"
        if fmt_type == "bp":
            return f"{diff * 100:+.0f} bp"
        if fmt_type == "K":
            return f"{diff / 1000:+.0f} K"
        return f"{diff:+.1f}"

    if sid in ("CPIAUCSL", "PCEPI", "UNRATE"):
        return dict(latest=f"{latest_val:,.2f}%", date=date_str,
                    mchg=_delta(latest_val, val_1m, "pp"),
                    ychg=_delta(latest_val, val_1y, "pp"))
    if sid in ("DGS10", "DGS2", "T10Y2Y"):
        return dict(latest=f"{latest_val:,.2f}%", date=date_str,
                    mchg=_delta(latest_val, val_1m, "bp"),
                    ychg=_delta(latest_val, val_1y, "bp"))
    if sid == "ICSA":
        return dict(latest=f"{latest_val / 1000:,.0f}K", date=date_str,
                    mchg=_delta(latest_val, val_1m, "K"),
                    ychg=_delta(latest_val, val_1y, "K"))
    if sid == "GDPC1":
        return dict(latest=f"{latest_val / 1000:,.1f}T", date=date_str,
                    mchg=_delta(latest_val, val_1m, "plain") if val_1m else "—",
                    ychg=_delta(latest_val, val_1y, "plain") if val_1y else "—")
    if sid == "FEDFUNDS":
        return dict(latest=f"{latest_val:,.2f}%", date=date_str,
                    mchg=_delta(latest_val, val_1m, "bp"),
                    ychg=_delta(latest_val, val_1y, "bp"))
    return dict(latest=f"{latest_val:,.2f}", date=date_str,
                mchg=_delta(latest_val, val_1m, "plain"),
                ychg=_delta(latest_val, val_1y, "plain"))


def _plot_div(df, sid):
    """Build a styled Plotly chart div with recession shading and reference lines."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["value"], mode="lines",
        line=dict(color="#003366", width=2),
        hovertemplate="%{x|%b %d, %Y}<br>%{y:.2f}<extra></extra>",
    ))

    fig.update_layout(**_CHART_LAYOUT)

    # recession shading
    for start, end in NBER_RECESSIONS:
        fig.add_vrect(x0=start, x1=end,
                      fillcolor="rgba(180,180,180,0.25)", line_width=0,
                      layer="below")

    # reference lines
    for ref in _REF_LINES.get(sid, []):
        if "y2" in ref:  # band (e.g. NAIRU)
            fig.add_hrect(y0=ref["y"], y1=ref["y2"],
                          fillcolor=ref["color"], line_width=0,
                          annotation_text=ref["label"],
                          annotation_position="top left",
                          layer="below")
        else:  # single line
            fig.add_hline(y=ref["y"], line_dash=ref.get("dash", "dash"),
                          line_color=ref["color"], line_width=1.5,
                          annotation_text=ref["label"],
                          annotation_position="top left")

    return fig.to_html(full_html=False, include_plotlyjs=False)


def _build_dashboard_table(conn, indicators):
    """Build the summary dashboard table HTML at the top of the page."""
    rows_html = []
    for sid, meta in indicators.items():
        df = _get_series(conn, sid)
        if df.empty:
            continue
        s = _stats(df, sid)
        pct = _pctile(pd.to_numeric(df["value"], errors="coerce").dropna())
        pct_class = "pct-low" if pct <= 20 else ("pct-high" if pct >= 80 else "")
        rows_html.append(
            f'<tr><td><a href="#{sid}">{meta["name"]}</a></td>'
            f'<td>{s["latest"]}</td><td>{s["date"]}</td>'
            f'<td>{s["mchg"]}</td><td>{s["ychg"]}</td>'
            f'<td class="{pct_class}">{_ordinal(pct)}</td></tr>'
        )
    return (
        '<table class="dashboard-table"><thead>'
        '<tr><th>Indicator</th><th>Latest</th><th>As Of</th>'
        '<th>1-Mo &Delta;</th><th>YoY &Delta;</th><th>Percentile</th></tr>'
        '</thead><tbody>' + "".join(rows_html) + '</tbody></table>'
    )


def _summary_sentence(sid, df):
    """Build a one-line percentile callout for an indicator."""
    s = pd.to_numeric(df["value"], errors="coerce").dropna()
    if len(s) < 20:
        return ""
    pct = _pctile(s)
    latest = float(s.iloc[-1])
    avg = float(s.mean())
    mn = float(s.min())
    mx = float(s.max())

    if sid == "ICSA":
        fmt = lambda v: f"{v / 1000:,.0f}K"
    elif sid == "GDPC1":
        fmt = lambda v: f"{v / 1000:,.1f}T"
    elif sid == "UMCSENT":
        fmt = lambda v: f"{v:.1f}"
    else:
        fmt = lambda v: f"{v:.2f}%"

    return (
        f"Current reading is {fmt(latest)}, placing it in the "
        f"<strong>{_ordinal(pct)} percentile</strong> of available history "
        f"(avg {fmt(avg)}, range {fmt(mn)} – {fmt(mx)})."
    )


# ───────────── main renderer ─────────────
def render_single_page(timestamp: str, indicators: dict):
    sections = []

    with sqlite3.connect(DB_PATH) as conn:
        dashboard_table = _build_dashboard_table(conn, indicators)

        for section_title, sids in SECTION_ORDER:
            charts_html = []
            for sid in sids:
                meta = indicators.get(sid)
                if not meta:
                    continue
                df = _get_series(conn, sid)
                if df.empty:
                    continue
                s = _stats(df, sid)
                div = _plot_div(df, sid)
                callout = _summary_sentence(sid, df)

                charts_html.append(f"""
                  <div class="chart-card" id="{sid}">
                    <h3>{meta['name']}</h3>
                    {div}
                    <p class="stats-line">
                      Latest: {s['latest']} ({s['date']}) &middot;
                      1-Mo &Delta;: {s['mchg']} &middot;
                      YoY &Delta;: {s['ychg']}
                    </p>
                    {"<p class='callout'>" + callout + "</p>" if callout else ""}
                  </div>
                """)

            if charts_html:
                sections.append(
                    f'<div class="section"><h2>{section_title}</h2>'
                    + "".join(charts_html)
                    + '</div>'
                )

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>U.S. Economic Indicators</title>
<link rel="stylesheet" href="retro.css">
<link rel="stylesheet" href="econ.css">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
</head>
<body>
<div class="econ-page">

  <h1>U.S. Economic Indicators</h1>
  <p class="updated">Updated: {timestamp} &middot; Sources: BLS &middot; FRED &middot; BEA &middot; U.S. Treasury</p>

  <h2>Dashboard Summary</h2>
  {dashboard_table}

  {''.join(sections)}

  <p class="back-link"><a href="index.html">&larr; Back to Dashboard</a></p>
</div>
</body>
</html>"""

    HTML_OUT.write_text(html, encoding="utf-8")
    print(f"[econ_page] wrote → {HTML_OUT}")

# CLI
if __name__ == "__main__":
    from generate_economic_data import INDICATORS
    render_single_page(datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"), INDICATORS)

#!/usr/bin/env python3
# economic_data_page.py — investor-focused dashboard rewrite
# ----------------------------------------------------------
# Replaces the flat indicator list with a 4-section dashboard organized
# around the questions a stock investor needs to answer:
#
#   1. Cycle & Recession Risk   — where are we in the business cycle?
#   2. Fed Policy Stance        — is the Fed helping or hurting equities?
#   3. Risk Pricing             — is risk priced cheaply or expensively?
#   4. Earnings & Valuation     — are corporate earnings supportable?
#
# Each section opens with a composite signal (gauge + label) and drills
# down into the individual FRED series. Each indicator is tagged with a
# stock-market signal (BULLISH / NEUTRAL / BEARISH) based on its current
# regime.
#
# Visual: 1999 dot-com aesthetic matching ticker.html — beveled chrome,
# navy headers, LCD metric strips, hi-yellow accents.
# ----------------------------------------------------------

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

DB_PATH = "Stock Data.db"
HTML_OUT = Path("economic_charts.html")

# ───────────── NBER recession periods (post-WWII) ─────────────
NBER_RECESSIONS = [
    ("1948-11-01", "1949-10-01"), ("1953-07-01", "1954-05-01"),
    ("1957-08-01", "1958-04-01"), ("1960-04-01", "1961-02-01"),
    ("1969-12-01", "1970-11-01"), ("1973-11-01", "1975-03-01"),
    ("1980-01-01", "1980-07-01"), ("1981-07-01", "1982-11-01"),
    ("1990-07-01", "1991-03-01"), ("2001-03-01", "2001-11-01"),
    ("2007-12-01", "2009-06-01"), ("2020-02-01", "2020-04-01"),
]

# ───────────── section layout ─────────────
# Each section: (title, prose intro, list of indicator IDs to drill into)
SECTIONS = [
    ("Cycle & Recession Risk",
     "How close are we to the next recession? Stocks lose 30-50% in recessions and "
     "lead the recovery before the data turns. The yield curve, jobless claims trend, "
     "and the Sahm Rule are the earliest reliable warnings.",
     ["T10Y3M", "T10Y2Y", "SAHMREALTIME", "RECPROUSM156N",
      "USSLIND", "UNRATE", "ICSA"]),
    ("Fed Policy Stance",
     "What is the Fed doing to equity multiples? Restrictive policy (real Fed Funds "
     "above ~0.5%) compresses P/E ratios. Easing expands them. Watch the 2-year "
     "Treasury — it forecasts where Fed Funds will be 12 months out.",
     ["DFF", "FEDFUNDS", "DGS2", "DGS10", "T10YIE", "CPIAUCSL", "PCEPI"]),
    ("Risk Pricing",
     "Is the market in risk-on or risk-off mode? Credit spreads, VIX, and financial "
     "stress all rise together when stocks are about to take a hit. The dollar "
     "strengthens in flight-to-safety; oil reflects both inflation and demand.",
     ["BAMLH0A0HYM2", "VIXCLS", "STLFSI4", "DTWEXBGS", "DCOILWTICO"]),
    ("Earnings Drivers",
     "Will corporate earnings hold up? Industrial production and payrolls drive "
     "revenue growth; real consumer spending drives the 70% of GDP that consumer "
     "stocks depend on; sentiment is a leading indicator of both.",
     ["INDPRO", "PAYEMS", "PCEC96", "UMCSENT"]),
]

# ───────────── indicator metadata ─────────────
# unit_kind controls how values format and how deltas read (pp vs bp vs raw).
META = {
    # Cycle
    "T10Y3M":        dict(name="10Y - 3M Treasury Spread",       unit_kind="pct",  invert=False),
    "T10Y2Y":        dict(name="10Y - 2Y Treasury Spread",       unit_kind="pct",  invert=False),
    "SAHMREALTIME":  dict(name="Sahm Rule (Recession Trigger)",  unit_kind="pp",   invert=False),
    "RECPROUSM156N": dict(name="NY Fed: 12-Mo Recession Probability", unit_kind="pct", invert=False),
    "USSLIND":       dict(name="Leading Economic Index (% chg)", unit_kind="pct",  invert=False),
    "UNRATE":        dict(name="Unemployment Rate",              unit_kind="pct",  invert=False),
    "ICSA":          dict(name="Initial Jobless Claims",         unit_kind="K",    invert=False),
    # Fed
    "DFF":           dict(name="Effective Fed Funds Rate",       unit_kind="pct",  invert=False),
    "FEDFUNDS":      dict(name="Fed Funds Target (Midpoint)",    unit_kind="pct",  invert=False),
    "DGS2":          dict(name="2-Year Treasury",                unit_kind="pct",  invert=False),
    "DGS10":         dict(name="10-Year Treasury",               unit_kind="pct",  invert=False),
    "T10YIE":        dict(name="10Y Breakeven Inflation",        unit_kind="pct",  invert=False),
    "CPIAUCSL":      dict(name="CPI (Headline, YoY)",            unit_kind="pct",  invert=False),
    "PCEPI":         dict(name="PCE Price Index (YoY)",          unit_kind="pct",  invert=False),
    # Risk
    "BAMLH0A0HYM2":  dict(name="HY Credit Spread (BAML OAS)",    unit_kind="pct",  invert=False),
    "VIXCLS":        dict(name="VIX (Equity Volatility)",        unit_kind="raw",  invert=False),
    "STLFSI4":       dict(name="STLFSI4 (Financial Stress)",     unit_kind="raw",  invert=False),
    "DTWEXBGS":      dict(name="Trade-Weighted Dollar Index",    unit_kind="raw",  invert=False),
    "DCOILWTICO":    dict(name="WTI Crude Oil ($/bbl)",          unit_kind="raw",  invert=False),
    # Earnings
    "INDPRO":        dict(name="Industrial Production (Index)",  unit_kind="raw",  invert=False),
    "PAYEMS":        dict(name="Nonfarm Payrolls (Thousands)",   unit_kind="K_total", invert=False),
    "PCEC96":        dict(name="Real Consumer Spending",         unit_kind="raw",  invert=False),
    "UMCSENT":       dict(name="Consumer Sentiment (UMich)",     unit_kind="raw",  invert=False),
    "GDPC1":         dict(name="Real GDP",                       unit_kind="T",    invert=False),
}


# ============================================================
#  Data helpers
# ============================================================
def _get_series(conn, sid):
    """Read an indicator from the DB; for FEDFUNDS use midpoint of DFEDTARL/U."""
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
        out = merged[["date", "value"]].copy()
        out["date"] = pd.to_datetime(out["date"])
        return out
    df = pd.read_sql(
        "SELECT substr(date,1,10) AS date, value FROM economic_data "
        "WHERE indicator=? ORDER BY substr(date,1,10)", conn, params=(sid,))
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def _last_value(df):
    if df.empty:
        return None
    s = pd.to_numeric(df["value"], errors="coerce").dropna()
    return float(s.iloc[-1]) if not s.empty else None


def _value_n_periods_ago(df, n):
    if df.empty or len(df) <= n:
        return None
    s = pd.to_numeric(df["value"], errors="coerce").dropna()
    return float(s.iloc[-(n + 1)]) if len(s) > n else None


def _value_by_offset(df, months=None, weeks=None, years=None):
    """Get value at approximately N months/weeks/years before the latest date."""
    if df.empty:
        return None
    df = df.dropna(subset=["value"]).sort_values("date")
    last_date = df["date"].iloc[-1]
    if months:
        target = last_date - pd.DateOffset(months=months)
    elif weeks:
        target = last_date - pd.DateOffset(weeks=weeks)
    elif years:
        target = last_date - pd.DateOffset(years=years)
    else:
        return None
    prior = df.loc[df["date"] <= target]
    return float(prior.iloc[-1]["value"]) if not prior.empty else None


def _pctile(df):
    """Percentile rank (1-99) of the latest value in full history."""
    if df.empty:
        return None
    s = pd.to_numeric(df["value"], errors="coerce").dropna()
    if s.empty:
        return None
    val = float(s.iloc[-1])
    rank = np.searchsorted(np.sort(s.values), val, side="right")
    pct = (rank / len(s)) * 100
    return int(round(max(1, min(99, pct))))


def _ordinal(n):
    if n is None:
        return "—"
    if 10 <= n % 100 <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


# ============================================================
#  Formatting helpers
# ============================================================
def _fmt_value(v, kind):
    if v is None:
        return "—"
    if kind == "pct":
        return f"{v:+.2f}%" if kind == "pct" and False else f"{v:.2f}%"
    if kind == "pp":
        return f"{v:+.2f} pp"
    if kind == "K":
        return f"{v / 1000:,.0f}K"
    if kind == "K_total":
        return f"{v / 1000:,.1f}M"
    if kind == "T":
        return f"${v / 1000:,.1f}T"
    if kind == "raw":
        return f"{v:,.2f}"
    return f"{v:.2f}"


def _fmt_delta(cur, prev, kind):
    if cur is None or prev is None:
        return "—"
    diff = cur - prev
    if kind == "pct" or kind == "pp":
        return f"{diff:+.2f} pp"
    if kind == "K":
        return f"{diff / 1000:+.0f}K"
    if kind == "K_total":
        return f"{diff:+,.0f}"
    if kind == "raw":
        return f"{diff:+.2f}"
    return f"{diff:+.2f}"


# ============================================================
#  Per-indicator stock-market signal logic
# ============================================================
def _signal(sid, df):
    """Return (label, css_class) for stock-market signal:
       BULLISH (green) / NEUTRAL (yellow) / BEARISH (red).
       Logic per indicator captures the regime, not the absolute level."""
    v = _last_value(df)
    if v is None:
        return ("—", "neu")

    # Yield curve: inverted = bearish (recession leading); steep = bullish
    if sid in ("T10Y3M", "T10Y2Y"):
        if v < 0:
            return ("BEARISH", "bear")
        if v < 0.5:
            return ("NEUTRAL", "neu")
        return ("BULLISH", "bull")

    # Sahm Rule: ≥0.5pp = recession signal
    if sid == "SAHMREALTIME":
        if v >= 0.50:
            return ("BEARISH", "bear")
        if v >= 0.30:
            return ("NEUTRAL", "neu")
        return ("BULLISH", "bull")

    # NY Fed recession prob
    if sid == "RECPROUSM156N":
        if v >= 30:
            return ("BEARISH", "bear")
        if v >= 15:
            return ("NEUTRAL", "neu")
        return ("BULLISH", "bull")

    # USSLIND: leading index growth rate
    if sid == "USSLIND":
        if v < -0.5:
            return ("BEARISH", "bear")
        if v < 0.5:
            return ("NEUTRAL", "neu")
        return ("BULLISH", "bull")

    # UNRATE: rising = bearish; very low = late cycle (neutral)
    if sid == "UNRATE":
        v_1y = _value_by_offset(df, months=12)
        if v_1y is not None and v - v_1y >= 0.5:
            return ("BEARISH", "bear")
        if v >= 5.5:
            return ("NEUTRAL", "neu")
        return ("BULLISH", "bull")

    # Initial claims: 4w MA above 350K = bearish
    if sid == "ICSA":
        if v > 350_000:
            return ("BEARISH", "bear")
        if v > 275_000:
            return ("NEUTRAL", "neu")
        return ("BULLISH", "bull")

    # Real Fed Funds: this is computed externally — placeholder here
    if sid in ("DFF", "FEDFUNDS"):
        # Restrictive = bearish for multiples
        return ("NEUTRAL", "neu")  # handled in section header instead

    # 2Y: market-implied Fed path. Lower than current FF = expecting cuts = bullish
    if sid == "DGS2":
        # Compare to effective Fed Funds
        return ("NEUTRAL", "neu")

    if sid == "DGS10":
        return ("NEUTRAL", "neu")

    if sid == "T10YIE":
        # Inflation expectations: anchored near 2% = bullish; above 3% or below 1.5% = neutral/bearish
        if 1.7 <= v <= 2.5:
            return ("BULLISH", "bull")
        if 1.3 <= v <= 3.0:
            return ("NEUTRAL", "neu")
        return ("BEARISH", "bear")

    # Inflation: high or volatile = bearish for stocks
    if sid in ("CPIAUCSL", "PCEPI"):
        if v >= 4.0 or v < 0:
            return ("BEARISH", "bear")
        if v >= 3.0 or v < 1.0:
            return ("NEUTRAL", "neu")
        return ("BULLISH", "bull")

    # Credit spreads: tighter = bullish risk-on; wide = bearish
    if sid == "BAMLH0A0HYM2":
        if v >= 6.0:
            return ("BEARISH", "bear")
        if v >= 4.0:
            return ("NEUTRAL", "neu")
        return ("BULLISH", "bull")

    # VIX: below 15 = complacent, 15-20 = normal, 20-30 = elevated, >30 = panic
    if sid == "VIXCLS":
        if v >= 25:
            return ("BEARISH", "bear")
        if v >= 18:
            return ("NEUTRAL", "neu")
        return ("BULLISH", "bull")

    # Financial stress index: > 1 = elevated stress
    if sid == "STLFSI4":
        if v >= 1.5:
            return ("BEARISH", "bear")
        if v >= 0:
            return ("NEUTRAL", "neu")
        return ("BULLISH", "bull")

    # Dollar: strengthening = headwind for multinationals
    if sid == "DTWEXBGS":
        v_1y = _value_by_offset(df, months=12)
        if v_1y is not None and (v / v_1y - 1) * 100 >= 8:
            return ("BEARISH", "bear")
        if v_1y is not None and (v / v_1y - 1) * 100 <= -5:
            return ("BULLISH", "bull")
        return ("NEUTRAL", "neu")

    # Oil: very high or very low = stress signal
    if sid == "DCOILWTICO":
        if v >= 100:
            return ("BEARISH", "bear")
        if v >= 80 or v < 50:
            return ("NEUTRAL", "neu")
        return ("BULLISH", "bull")

    # Industrial production: YoY change matters
    if sid == "INDPRO":
        v_1y = _value_by_offset(df, months=12)
        if v_1y is None:
            return ("NEUTRAL", "neu")
        chg = (v / v_1y - 1) * 100
        if chg < 0:
            return ("BEARISH", "bear")
        if chg < 1.5:
            return ("NEUTRAL", "neu")
        return ("BULLISH", "bull")

    # Payrolls: monthly change matters
    if sid == "PAYEMS":
        v_1m = _value_by_offset(df, months=1)
        if v_1m is None:
            return ("NEUTRAL", "neu")
        chg = v - v_1m
        if chg < 0:
            return ("BEARISH", "bear")
        if chg < 100:  # below 100K = warning
            return ("NEUTRAL", "neu")
        return ("BULLISH", "bull")

    # Real consumer spending: YoY
    if sid == "PCEC96":
        v_1y = _value_by_offset(df, months=12)
        if v_1y is None:
            return ("NEUTRAL", "neu")
        chg = (v / v_1y - 1) * 100
        if chg < 0:
            return ("BEARISH", "bear")
        if chg < 1.5:
            return ("NEUTRAL", "neu")
        return ("BULLISH", "bull")

    # Consumer sentiment: below 70 = recession-zone
    if sid == "UMCSENT":
        if v < 65:
            return ("BEARISH", "bear")
        if v < 80:
            return ("NEUTRAL", "neu")
        return ("BULLISH", "bull")

    return ("NEUTRAL", "neu")


# ============================================================
#  Composite indicators
# ============================================================
def compute_recession_risk(series_map):
    """0-100 score where higher = more recession risk.
       Components: T10Y3M inversion (40), Sahm (30), claims trend (15), LEI (15)."""
    score = 0.0
    notes = []

    # T10Y3M inversion (40 weight)
    t10y3m = _last_value(series_map.get("T10Y3M", pd.DataFrame()))
    if t10y3m is not None:
        # -1 to +3 normal range; -1.5 = 40pts, +0.5 = 0pts
        c = max(0, min(40, (0.5 - t10y3m) * 20))
        score += c
        notes.append(f"Yield curve {t10y3m:+.2f}% (+{c:.0f})")

    # Sahm Rule (30 weight)
    sahm = _last_value(series_map.get("SAHMREALTIME", pd.DataFrame()))
    if sahm is not None:
        c = max(0, min(30, sahm * 60))  # 0.5pp = 30pts
        score += c
        notes.append(f"Sahm {sahm:+.2f}pp (+{c:.0f})")

    # Initial claims trend (15 weight): 4w MA vs 26w MA
    claims = series_map.get("ICSA", pd.DataFrame())
    if not claims.empty:
        s = pd.to_numeric(claims["value"], errors="coerce").dropna()
        if len(s) >= 26:
            ma4 = s.iloc[-4:].mean()
            ma26 = s.iloc[-26:].mean()
            ratio = ma4 / ma26 - 1
            c = max(0, min(15, ratio * 100))  # +15% claims trend = 15pts
            score += c
            notes.append(f"Claims trend {ratio*100:+.1f}% (+{c:.0f})")

    # LEI (15 weight): if negative, add points
    lei = _last_value(series_map.get("USSLIND", pd.DataFrame()))
    if lei is not None:
        c = max(0, min(15, (1.0 - lei) * 10))
        score += c
        notes.append(f"LEI {lei:+.2f}% (+{c:.0f})")

    return int(round(score)), notes


def compute_real_fed_funds(series_map):
    """Real Fed Funds = effective FF - YoY core PCE inflation."""
    ff = _last_value(series_map.get("DFF", pd.DataFrame()))
    core_pce = series_map.get("PCEPILFE", pd.DataFrame())
    if ff is None or core_pce.empty:
        return None, None
    s = pd.to_numeric(core_pce["value"], errors="coerce").dropna()
    if len(s) < 13:
        return None, None
    yoy = (s.iloc[-1] / s.iloc[-13] - 1) * 100
    return ff - yoy, yoy


def compute_real_fed_funds_history(series_map):
    """Time series of real Fed Funds rate for charting."""
    ff_df = series_map.get("DFF", pd.DataFrame())
    cpce_df = series_map.get("PCEPILFE", pd.DataFrame())
    if ff_df.empty or cpce_df.empty:
        return pd.DataFrame(columns=["date", "value"])
    # Monthly-align: take month-end values
    ff = ff_df.copy()
    ff["date"] = pd.to_datetime(ff["date"])
    ff_m = (ff.set_index("date")["value"]
              .resample("ME").mean()
              .dropna())
    cpce = cpce_df.copy()
    cpce["date"] = pd.to_datetime(cpce["date"])
    cpce_m = (cpce.set_index("date")["value"]
                  .resample("ME").last()
                  .dropna())
    yoy = (cpce_m / cpce_m.shift(12) - 1) * 100
    aligned = pd.concat([ff_m.rename("ff"), yoy.rename("yoy")], axis=1).dropna()
    aligned["value"] = aligned["ff"] - aligned["yoy"]
    out = aligned.reset_index()[["date", "value"]]
    return out


def compute_risk_pricing(series_map):
    """0-100 score where higher = more risk-off (stress)."""
    pcts = []
    notes = []

    for sid, label in [("BAMLH0A0HYM2", "HY Spread"),
                       ("VIXCLS", "VIX"),
                       ("STLFSI4", "Stress Idx")]:
        df = series_map.get(sid, pd.DataFrame())
        p = _pctile(df)
        if p is not None:
            pcts.append(p)
            notes.append(f"{label} {_ordinal(p)} pctile")

    if not pcts:
        return None, []
    score = int(round(sum(pcts) / len(pcts)))
    return score, notes


# ============================================================
#  Plotly charts (1999 styling)
# ============================================================
_CHART_LAYOUT = dict(
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#F8F8FF",
    font=dict(family="Verdana, Geneva, sans-serif", color="#000080", size=11),
    margin=dict(l=42, r=14, t=8, b=32),
    height=260,
    xaxis=dict(
        gridcolor="#E0E0FF", linecolor="#000080", type="date",
        rangeselector=dict(
            bgcolor="#C0C0C0", bordercolor="#000080",
            font=dict(size=10, color="#000080", family="Arial,sans-serif"),
            buttons=[
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(count=5, label="5Y", step="year", stepmode="backward"),
                dict(count=10, label="10Y", step="year", stepmode="backward"),
                dict(count=25, label="25Y", step="year", stepmode="backward"),
                dict(step="all", label="MAX"),
            ],
        ),
        rangeslider=dict(visible=False),
    ),
    yaxis=dict(gridcolor="#E0E0FF", linecolor="#000080", zerolinecolor="#000080", zerolinewidth=1),
)


def _line_chart_div(df, sid, ref_lines=None, default_years=25):
    """Build a styled Plotly line chart for a time series."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["value"], mode="lines",
        line=dict(color="#000080", width=1.6),
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>%{y:.2f}<extra></extra>",
        showlegend=False,
    ))
    layout = dict(_CHART_LAYOUT)
    fig.update_layout(**layout)

    # Recession shading
    for start, end in NBER_RECESSIONS:
        fig.add_vrect(x0=start, x1=end,
                      fillcolor="rgba(128,0,0,0.10)", line_width=0,
                      layer="below")

    # Reference lines
    for ref in (ref_lines or []):
        fig.add_hline(y=ref["y"], line_dash=ref.get("dash", "dash"),
                      line_color=ref.get("color", "#CC0000"),
                      line_width=1.2,
                      annotation_text=ref.get("label", ""),
                      annotation_position="top right",
                      annotation_font=dict(size=9, color=ref.get("color", "#CC0000")))

    # Default to last N years visible
    if not df.empty:
        end_date = pd.to_datetime(df["date"]).max()
        start_date = end_date - pd.DateOffset(years=default_years)
        fig.update_layout(xaxis_range=[start_date.strftime("%Y-%m-%d"),
                                       end_date.strftime("%Y-%m-%d")])

    return fig.to_html(full_html=False, include_plotlyjs=False, config={
        "displayModeBar": False, "responsive": True, "scrollZoom": False,
        "doubleClick": False, "showAxisDragHandles": False, "staticPlot": False,
    })


def _gauge_div(score, label, color):
    """Build a small gauge (0-100) showing a composite score."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score if score is not None else 0,
        number=dict(font=dict(size=42, color="#33FF33", family="Courier New, monospace"),
                    suffix="<span style='font-size:13px;color:#88CCFF'> /100</span>"),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor="#000080",
                      tickfont=dict(size=10, color="#FFFFFF")),
            bar=dict(color=color, thickness=0.30),
            bgcolor="#000000",
            borderwidth=2,
            bordercolor="#000080",
            steps=[
                dict(range=[0, 33], color="rgba(0,128,0,0.20)"),
                dict(range=[33, 66], color="rgba(204,204,0,0.20)"),
                dict(range=[66, 100], color="rgba(204,0,0,0.25)"),
            ],
        ),
        title=dict(text=f"<b>{label}</b>",
                   font=dict(size=14, color="#FFFFFF", family="Arial Black")),
        domain=dict(x=[0, 1], y=[0, 1]),
    ))
    fig.update_layout(
        paper_bgcolor="#000000",
        height=200, margin=dict(l=20, r=20, t=42, b=12),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False, config={
        "displayModeBar": False, "responsive": True, "staticPlot": True,
    })


# ============================================================
#  Indicator card HTML
# ============================================================
def _indicator_card(sid, df):
    meta = META.get(sid, dict(name=sid, unit_kind="raw"))
    name = meta["name"]
    kind = meta["unit_kind"]

    latest = _last_value(df)
    delta_1m = _fmt_delta(latest, _value_by_offset(df, months=1), kind)
    delta_1y = _fmt_delta(latest, _value_by_offset(df, years=1), kind)
    latest_disp = _fmt_value(latest, kind)
    pct = _pctile(df)

    label, sig_cls = _signal(sid, df)

    # Determine if a ref line makes sense
    ref_lines = None
    if sid == "T10Y3M" or sid == "T10Y2Y":
        ref_lines = [dict(y=0, color="#CC0000", dash="dash", label="Inversion")]
    elif sid == "SAHMREALTIME":
        ref_lines = [dict(y=0.5, color="#CC0000", dash="dash", label="Recession Trigger")]
    elif sid in ("CPIAUCSL", "PCEPI"):
        ref_lines = [dict(y=2, color="#CC0000", dash="dash", label="2% Target")]
    elif sid == "T10YIE":
        ref_lines = [dict(y=2, color="#666666", dash="dash", label="2% Target")]
    elif sid == "UNRATE":
        ref_lines = [dict(y=4.0, color="#666666", dash="dot", label="NAIRU ~4%")]
    elif sid == "BAMLH0A0HYM2":
        ref_lines = [dict(y=5, color="#CC0000", dash="dash", label="Stress Zone")]
    elif sid == "VIXCLS":
        ref_lines = [dict(y=20, color="#CC0000", dash="dash", label="Elevated")]
    elif sid == "STLFSI4":
        ref_lines = [dict(y=0, color="#666666", dash="dash", label="Avg Stress")]

    chart_html = _line_chart_div(df, sid, ref_lines=ref_lines)

    pct_str = f"{_ordinal(pct)} pctile" if pct is not None else "—"

    return f"""
  <div class="ind-card" id="{sid}">
    <div class="ind-header">
      <span class="ind-name">{name}</span>
      <span class="signal sig-{sig_cls}">{label}</span>
    </div>
    <div class="ind-metrics">
      <span class="m-cell"><span class="m-label">Latest</span><span class="m-val">{latest_disp}</span></span>
      <span class="m-cell"><span class="m-label">1Mo Δ</span><span class="m-val">{delta_1m}</span></span>
      <span class="m-cell"><span class="m-label">YoY Δ</span><span class="m-val">{delta_1y}</span></span>
      <span class="m-cell"><span class="m-label">Pctile</span><span class="m-val">{pct_str}</span></span>
    </div>
    <div class="ind-chart">{chart_html}</div>
  </div>
"""


# ============================================================
#  Section / composite headers
# ============================================================
def _cycle_section_header(series_map):
    score, notes = compute_recession_risk(series_map)
    if score is None:
        return ""
    if score >= 66:
        label, color = "HIGH RISK", "#CC0000"
    elif score >= 33:
        label, color = "ELEVATED", "#CC9900"
    else:
        label, color = "LOW RISK", "#008000"
    gauge = _gauge_div(score, "Recession Risk", color)
    notes_str = " · ".join(notes)
    return f"""
  <div class="composite-strip">
    <div class="composite-gauge">{gauge}</div>
    <div class="composite-text">
      <h3>Recession Risk Score: <span style='color:{color}'>{score}/100 · {label}</span></h3>
      <p class="composite-detail">{notes_str}</p>
      <p class="composite-explain">A composite of the 10Y-3M curve (40 wt), Sahm Rule
      (30), jobless claims trend (15), and the Leading Index (15). Above 50 historically
      precedes recessions by 6-18 months. Below 33 has never coincided with a recession start.</p>
    </div>
  </div>"""


def _fed_section_header(series_map):
    real_ff, yoy = compute_real_fed_funds(series_map)
    if real_ff is None:
        return ""
    if real_ff >= 1.5:
        label, color = "VERY RESTRICTIVE", "#CC0000"
    elif real_ff >= 0.5:
        label, color = "RESTRICTIVE", "#CC9900"
    elif real_ff >= -0.5:
        label, color = "NEUTRAL", "#000080"
    elif real_ff >= -1.5:
        label, color = "ACCOMMODATIVE", "#66AA00"
    else:
        label, color = "VERY EASY", "#008000"
    # Render as a "metric chip" instead of gauge for this section
    return f"""
  <div class="composite-strip">
    <div class="composite-metric">
      <div class="lcd-strip">
        <div class="lcd-cell"><span class="lcd-label">REAL FED FUNDS</span>
             <span class="lcd-val">{real_ff:+.2f}%</span></div>
        <div class="lcd-cell"><span class="lcd-label">CORE PCE YoY</span>
             <span class="lcd-val">{yoy:.2f}%</span></div>
        <div class="lcd-cell"><span class="lcd-label">STANCE</span>
             <span class="lcd-val" style="color:{color}">{label}</span></div>
      </div>
    </div>
    <div class="composite-text">
      <h3>Fed Policy: <span style='color:{color}'>{label}</span></h3>
      <p class="composite-explain">Real Fed Funds = Effective Fed Funds rate minus
      core-PCE inflation (YoY). The "neutral" rate is widely estimated at ~0.5%.
      Above that range, policy is restrictive — historically a 6-18 month headwind to
      equity multiples. Below -0.5%, policy is accommodative and supportive of risk assets.</p>
    </div>
  </div>"""


def _risk_section_header(series_map):
    score, notes = compute_risk_pricing(series_map)
    if score is None:
        return ""
    if score >= 66:
        label, color = "RISK-OFF", "#CC0000"
    elif score >= 33:
        label, color = "NEUTRAL", "#CC9900"
    else:
        label, color = "RISK-ON", "#008000"
    gauge = _gauge_div(score, "Stress Score", color)
    notes_str = " · ".join(notes)
    return f"""
  <div class="composite-strip">
    <div class="composite-gauge">{gauge}</div>
    <div class="composite-text">
      <h3>Risk Pricing: <span style='color:{color}'>{score}/100 · {label}</span></h3>
      <p class="composite-detail">{notes_str}</p>
      <p class="composite-explain">Average percentile rank of HY credit spreads, VIX,
      and the St. Louis Fed Financial Stress Index. Above 66 = stress regime (equity
      drawdowns more likely). Below 33 = complacent (equity drawdowns deeper when they come).</p>
    </div>
  </div>"""


def _earnings_section_header(series_map):
    # Simple count-of-bullish-signals approach for earnings
    bullish = neutral = bearish = 0
    sids = ["INDPRO", "PAYEMS", "PCEC96", "UMCSENT"]
    for sid in sids:
        label, _ = _signal(sid, series_map.get(sid, pd.DataFrame()))
        if label == "BULLISH": bullish += 1
        elif label == "BEARISH": bearish += 1
        else: neutral += 1
    if bearish >= 2:
        label, color = "WEAKENING", "#CC0000"
    elif bullish >= 3:
        label, color = "STRONG", "#008000"
    else:
        label, color = "MIXED", "#CC9900"
    return f"""
  <div class="composite-strip">
    <div class="composite-metric">
      <div class="lcd-strip">
        <div class="lcd-cell"><span class="lcd-label">BULLISH</span>
             <span class="lcd-val" style="color:#33FF33">{bullish}</span></div>
        <div class="lcd-cell"><span class="lcd-label">NEUTRAL</span>
             <span class="lcd-val" style="color:#FFCC33">{neutral}</span></div>
        <div class="lcd-cell"><span class="lcd-label">BEARISH</span>
             <span class="lcd-val" style="color:#FF3333">{bearish}</span></div>
        <div class="lcd-cell"><span class="lcd-label">VERDICT</span>
             <span class="lcd-val" style="color:{color}">{label}</span></div>
      </div>
    </div>
    <div class="composite-text">
      <h3>Earnings Outlook: <span style='color:{color}'>{label}</span></h3>
      <p class="composite-explain">Industrial production, payrolls, real consumer
      spending, and sentiment together explain ~70% of next-year earnings revisions.
      When 2+ flip bearish, S&amp;P forward EPS estimates typically follow within 1-2 quarters.</p>
    </div>
  </div>"""


# ============================================================
#  Hero metric strip at top of page
# ============================================================
def _hero_strip(series_map):
    """Top-of-page LCD strip with the four composite headlines."""
    rec_score, _ = compute_recession_risk(series_map)
    real_ff, _ = compute_real_fed_funds(series_map)
    risk_score, _ = compute_risk_pricing(series_map)

    def _color(score, ranges):
        for thresh, color in ranges:
            if score >= thresh:
                return color
        return ranges[-1][1]

    rec_color = _color(rec_score or 0,
                       [(66, "#FF3333"), (33, "#FFCC33"), (0, "#33FF33")])
    risk_color = _color(risk_score or 0,
                        [(66, "#FF3333"), (33, "#FFCC33"), (0, "#33FF33")])
    if real_ff is None:
        ff_disp = "—"; ff_color = "#33FF33"
    else:
        ff_disp = f"{real_ff:+.2f}%"
        if real_ff >= 1.5: ff_color = "#FF3333"
        elif real_ff >= 0.5: ff_color = "#FFCC33"
        elif real_ff >= -0.5: ff_color = "#33FF33"
        else: ff_color = "#33FF33"

    rec_disp = f"{rec_score}/100" if rec_score is not None else "—"
    risk_disp = f"{risk_score}/100" if risk_score is not None else "—"

    return f"""
  <div class="hero-strip">
    <div class="hero-cell">
      <span class="hero-label">RECESSION RISK</span>
      <span class="hero-val" style="color:{rec_color}">{rec_disp}</span>
    </div>
    <div class="hero-cell">
      <span class="hero-label">REAL FED FUNDS</span>
      <span class="hero-val" style="color:{ff_color}">{ff_disp}</span>
    </div>
    <div class="hero-cell">
      <span class="hero-label">RISK PRICING</span>
      <span class="hero-val" style="color:{risk_color}">{risk_disp}</span>
    </div>
  </div>"""


# ============================================================
#  Page CSS — ports 1999 dot-com aesthetic from ticker.html
# ============================================================
PAGE_CSS = """
:root{
  --navy:#000080;
  --navy-dim:#000060;
  --gray:#C0C0C0;
  --gray-bg:#DDDDDD;
  --hi-yellow:#FFFF00;
  --mustard:#FFCC00;
  --text:#000000;
}
*{ box-sizing:border-box }
html,body{ margin:0; padding:0 }
body{
  font-family:"Times New Roman",Times,serif;
  color:var(--text); font-size:13px; line-height:1.4;
  background-color:#9999BB;
  background-image:
    repeating-linear-gradient(45deg, rgba(255,255,255,.18) 0 2px, transparent 2px 8px),
    repeating-linear-gradient(-45deg, rgba(0,0,0,.10) 0 2px, transparent 2px 8px),
    linear-gradient(180deg, #9faabe 0%, #7e8aa6 100%);
  background-attachment:fixed; min-height:100vh;
}
.wrap{ max-width:1100px; margin:0 auto; padding:0 8px }

/* Marquee at top */
.wrap::before{
  content:"◄ ▌U.S. ECONOMIC INDICATORS FOR STOCK INVESTORS ▌ DATA: FRED · BLS · BEA · TREASURY ▌ ALL VALUES DELAYED ▌ NOT INVESTMENT ADVICE ▌ © 1999. NICK'S STOCK FINANCIALS™ ►";
  display:block;
  background:#000000;
  color:var(--hi-yellow);
  font-family:"Courier New",monospace;
  font-size:12px; font-weight:700; letter-spacing:.5px;
  white-space:nowrap; overflow:hidden;
  padding:5px 0;
  border-top:2px solid var(--navy);
  border-bottom:2px solid var(--navy);
  margin:0 -8px 6px;
  animation:ticker 40s linear infinite;
  transform:translateX(100%);
  text-shadow:0 0 4px rgba(255,255,0,.5);
}
@keyframes ticker{ from{transform:translateX(100%)} to{transform:translateX(-100%)} }

/* Back button */
.back-bar{ margin:0 -4px 6px; padding:0 4px }
.back-btn{
  display:inline-block; text-decoration:none;
  font-family:Arial,sans-serif; font-weight:700; font-size:11px;
  color:#000; background:#C0C0C0;
  border:2px outset #C0C0C0; padding:4px 12px 5px;
  text-shadow:1px 1px 0 #FFFFFF;
  text-transform:uppercase; letter-spacing:.4px;
}
.back-btn:active{ border:2px inset #C0C0C0; background:#B8B8B8 }
.back-btn::before{ content:"« " }

/* Page title */
.page-title{
  font-family:Arial Black,Arial,sans-serif;
  font-size:18px; color:#FFFFFF;
  background:var(--navy);
  margin:0 0 4px;
  padding:6px 12px 5px;
  letter-spacing:.5px; text-transform:uppercase;
  border-top:1px solid #4444CC;
  border-bottom:2px ridge var(--gray);
  box-shadow:inset 0 1px 0 #4444CC;
  position:relative;
}
.page-title::before{ content:"› "; color:var(--hi-yellow); text-shadow:0 0 4px rgba(255,255,0,.6) }
.page-title::after{
  content:"◉ LIVE"; position:absolute; right:10px; top:50%; transform:translateY(-50%);
  font-family:"Courier New",monospace; font-size:10px; letter-spacing:1px;
  color:#FF3333; font-weight:700;
  background:#000000; padding:2px 5px;
  border:1px solid #FF3333;
  box-shadow:0 0 4px rgba(255,0,0,.5);
  animation:blink 1.2s steps(2) infinite;
}
@keyframes blink{ 50%{ opacity:.25 } }
.updated{
  font-family:"Courier New",monospace;
  font-size:11px; color:#FFFFCC; background:var(--navy-dim);
  padding:3px 12px 4px; margin:0 0 6px;
  border-bottom:2px solid var(--navy);
}

/* Hero LCD strip */
.hero-strip{
  display:grid; grid-template-columns:repeat(3,1fr); gap:4px;
  background:var(--gray); padding:5px;
  border:2px outset var(--gray);
  margin-bottom:8px;
}
.hero-cell{
  background:#000000; border:2px inset var(--gray);
  padding:8px 6px; text-align:center;
  display:flex; flex-direction:column; gap:3px;
  box-shadow:inset 0 0 0 1px #222;
}
.hero-label{
  font-family:Arial,sans-serif; font-size:10px;
  font-weight:700; letter-spacing:1.2px;
  color:#88CCFF; text-transform:uppercase;
}
.hero-val{
  font-family:"Courier New",monospace;
  font-size:24px; font-weight:700; line-height:1;
  text-shadow:0 0 6px currentColor;
}

/* Section heading */
.section{
  margin:14px 0 8px;
  background:#FFFFFF;
  border:2px outset var(--gray);
}
.section-h2{
  font-family:Arial Black,Arial,sans-serif;
  font-size:14px; color:#FFFFFF;
  background:var(--navy);
  margin:0; padding:5px 10px 4px;
  letter-spacing:.5px; text-transform:uppercase;
  border-top:1px solid #4444CC;
  border-bottom:2px ridge var(--gray);
  box-shadow:inset 0 1px 0 #4444CC;
  position:relative;
}
.section-h2::before{ content:"» "; color:var(--hi-yellow); text-shadow:0 0 4px rgba(255,255,0,.6) }
.section-prose{
  font-family:"Times New Roman",serif;
  font-size:12px; font-style:italic;
  color:#444; padding:8px 12px 4px;
  margin:0;
}

/* Composite strip */
.composite-strip{
  display:grid; grid-template-columns:300px 1fr; gap:10px;
  margin:6px 8px 10px; padding:8px;
  background:var(--gray-bg);
  border:2px inset var(--gray);
}
.composite-gauge, .composite-metric{ min-width:0 }
.composite-text h3{
  font-family:Arial Black,Arial,sans-serif;
  font-size:14px; color:var(--navy);
  margin:0 0 6px; padding:0;
}
.composite-detail{
  font-family:"Courier New",monospace;
  font-size:11px; color:#333;
  margin:0 0 6px;
}
.composite-explain{
  font-family:"Times New Roman",serif;
  font-size:11.5px; font-style:italic;
  color:#444; margin:0;
}
@media (max-width:720px){
  .composite-strip{ grid-template-columns:1fr }
}

/* LCD strip inside fed/earnings composites */
.lcd-strip{
  display:flex; gap:4px; flex-wrap:wrap;
  background:var(--gray); padding:4px;
  border:2px inset var(--gray);
}
.lcd-cell{
  background:#000000; border:2px inset var(--gray);
  padding:6px 8px; flex:1; min-width:90px;
  display:flex; flex-direction:column; gap:2px;
  text-align:center;
}
.lcd-label{
  font-family:Arial,sans-serif; font-size:9px;
  letter-spacing:1px; color:#88CCFF; font-weight:700;
}
.lcd-val{
  font-family:"Courier New",monospace;
  font-size:15px; font-weight:700; color:#33FF33;
  text-shadow:0 0 4px rgba(51,255,51,.5);
}

/* Indicator card */
.ind-card{
  background:#FFFFFF;
  border:2px ridge var(--gray);
  margin:6px 8px 8px;
}
.ind-header{
  background:var(--navy-dim);
  padding:4px 10px 3px;
  display:flex; justify-content:space-between; align-items:center;
  border-bottom:1px solid var(--navy);
}
.ind-name{
  font-family:Arial,sans-serif; font-weight:700;
  font-size:12px; color:#FFFFFF; letter-spacing:.4px;
  text-transform:uppercase;
}
.signal{
  display:inline-block;
  font-family:Arial Black,Arial,sans-serif;
  font-size:10px; letter-spacing:1px;
  padding:2px 8px;
  border:1px solid #000;
  text-shadow:1px 1px 0 rgba(0,0,0,.3);
}
.sig-bull{ background:#33CC33; color:#003300 }
.sig-neu{ background:#FFCC33; color:#553300 }
.sig-bear{ background:#FF3333; color:#330000 }

.ind-metrics{
  display:grid; grid-template-columns:repeat(4,1fr); gap:3px;
  background:var(--gray); padding:4px;
}
.m-cell{
  background:#000000; border:2px inset var(--gray);
  padding:4px 4px 5px; text-align:center;
  display:flex; flex-direction:column; gap:2px;
}
.m-label{
  font-family:Arial,sans-serif; font-size:8.5px;
  letter-spacing:1px; color:#88CCFF; font-weight:700;
}
.m-val{
  font-family:"Courier New",monospace;
  font-size:13px; font-weight:700; color:#33FF33;
  text-shadow:0 0 3px rgba(51,255,51,.5);
}
.ind-chart{ padding:4px 8px 8px }

/* Footer */
.footer{
  font-family:"Times New Roman",serif; font-style:italic;
  font-size:11px; text-align:center; color:#000080;
  background:var(--gray); border:2px ridge var(--gray);
  padding:8px 10px; margin:12px 0 8px;
}
"""


# ============================================================
#  Main renderer
# ============================================================
def render_single_page(timestamp: str, indicators: dict):
    # Load EVERY series we might need into memory, keyed by FRED sid
    needed = set()
    for _title, _prose, sids in SECTIONS:
        needed.update(sids)
    # Composites need a few additional series
    needed.update({"DFF", "PCEPILFE"})

    series_map = {}
    with sqlite3.connect(DB_PATH) as conn:
        for sid in needed:
            df = _get_series(conn, sid)
            series_map[sid] = df

    hero = _hero_strip(series_map)

    section_blocks = []
    for title, prose, sids in SECTIONS:
        # composite header per section
        if "Cycle" in title:
            comp = _cycle_section_header(series_map)
        elif "Fed" in title:
            comp = _fed_section_header(series_map)
        elif "Risk" in title:
            comp = _risk_section_header(series_map)
        else:
            comp = _earnings_section_header(series_map)

        cards = []
        for sid in sids:
            df = series_map.get(sid, pd.DataFrame())
            if df.empty:
                cards.append(
                    f'<div class="ind-card"><div class="ind-header">'
                    f'<span class="ind-name">{META.get(sid,{}).get("name",sid)}</span>'
                    f'<span class="signal sig-neu">PENDING DATA</span></div>'
                    f'<p style="font-style:italic;color:#666;padding:10px 14px;margin:0">'
                    f'Data not yet populated. The next weekly refresh will fill this series.</p></div>')
                continue
            cards.append(_indicator_card(sid, df))

        section_blocks.append(f"""
  <div class="section">
    <h2 class="section-h2">{title}</h2>
    <p class="section-prose">{prose}</p>
    {comp}
    {''.join(cards)}
  </div>""")

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>U.S. Economic Indicators — Nick's Stock Financials</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>{PAGE_CSS}</style>
</head>
<body>
<div class="wrap">

  <div class="back-bar"><a class="back-btn" href="index.html" onclick="if(document.referrer&&history.length>1){{history.back();return false}}">Back</a></div>

  <h1 class="page-title">U.S. Economic Indicators</h1>
  <div class="updated">UPDATED: {timestamp} · SOURCES: FRED · BLS · BEA · U.S. TREASURY</div>

  {hero}

  {''.join(section_blocks)}

  <div class="back-bar" style="margin-top:14px"><a class="back-btn" href="index.html" onclick="if(document.referrer&&history.length>1){{history.back();return false}}">Back to Home</a></div>

  <p class="footer">
    For educational purposes. Past relationships between macro indicators and equity
    returns do not guarantee future performance. © 1999.
  </p>

</div>
</body>
</html>"""

    HTML_OUT.write_text(html, encoding="utf-8")
    print(f"[econ_page] wrote → {HTML_OUT}  ({len(html):,} bytes)")


# ============================================================
#  CLI
# ============================================================
if __name__ == "__main__":
    from generate_economic_data import INDICATORS
    render_single_page(
        datetime.now(timezone.utc).replace(tzinfo=None).strftime("%Y-%m-%d %H:%M UTC"),
        INDICATORS,
    )

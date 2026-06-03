#!/usr/bin/env python3
"""Build a mobile-friendly Plotly mockup of the AAPL ticker page.

Demonstrates what the per-stock pages could look like if we:
  - Sourced financial history from SEC EDGAR (15+ years vs yfinance's 4-5).
  - Replaced matplotlib PNG charts with Plotly interactive (touch-friendly).
  - Added a sticky values strip up top so current numbers don't require
    hovering on a tooltip.
  - Used a built-in range selector (1Y/5Y/10Y/MAX) instead of a fixed view.

Output: aapl_mockup.html at the repo root. The site-refresh-daily workflow
rsyncs the repo into gh-pages, so the file will appear at
https://nicksstockfinancials.com/aapl_mockup.html within one refresh cycle.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import yfinance as yf
from plotly.subplots import make_subplots


HEADERS = {"User-Agent": "Nick Daly StockFinances ndaly111@gmail.com"}
EDGAR_CONCEPT = "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{concept}.json"
CIK = "0000320193"  # AAPL
TICKER = "AAPL"
OUT_FILE = Path(__file__).resolve().parents[1] / "aapl_mockup.html"


# -------------------------- EDGAR helpers ----------------------------------
def fetch_concept(concept: str, fallbacks: tuple[str, ...] = ()) -> list[dict]:
    """Merge results from concept + fallbacks (companies often switch tag names
    over time, e.g. AAPL Revenues → RevenueFromContractWithCustomer at ASC 606
    adoption in 2018)."""
    combined: list[dict] = []
    seen_keys: set = set()
    for c in (concept, *fallbacks):
        url = EDGAR_CONCEPT.format(cik=CIK, concept=c)
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code != 200:
            continue
        units = r.json().get("units", {})
        for key in ("USD", "USD/shares"):
            if key in units and units[key]:
                for v in units[key]:
                    dedupe = (v.get("start"), v.get("end"), v.get("form"))
                    if dedupe in seen_keys:
                        continue
                    seen_keys.add(dedupe)
                    combined.append(v)
                break  # take first matching unit per concept
    return combined


def annual_series(entries: list[dict]) -> pd.Series:
    """Annual values from 10-K filings (span ~363 days)."""
    out: dict[pd.Timestamp, float] = {}
    for v in entries:
        sd, ed, val = v.get("start"), v.get("end"), v.get("val")
        if not sd or not ed or val is None or v.get("form") != "10-K":
            continue
        try:
            span = (dt.date.fromisoformat(ed) - dt.date.fromisoformat(sd)).days
        except Exception:
            continue
        if 340 <= span <= 380:
            ts = pd.Timestamp(ed).normalize()
            out.setdefault(ts, float(val))
    return pd.Series(out).sort_index()


def quarterly_series(entries: list[dict]) -> pd.Series:
    """Single-quarter values (span ~90 days)."""
    out: dict[pd.Timestamp, float] = {}
    for v in entries:
        sd, ed, val = v.get("start"), v.get("end"), v.get("val")
        if not sd or not ed or val is None:
            continue
        try:
            span = (dt.date.fromisoformat(ed) - dt.date.fromisoformat(sd)).days
        except Exception:
            continue
        if 75 <= span <= 105:
            ts = pd.Timestamp(ed).normalize()
            out.setdefault(ts, float(val))
    return pd.Series(out).sort_index()


def ttm_from_quarterly(q: pd.Series) -> pd.Series:
    """4-quarter rolling sum on a quarterly Series."""
    return q.rolling(window=4, min_periods=4).sum()


# ---------------------- assemble ------------------------------------------
def main() -> int:
    print("Fetching AAPL EDGAR data...")
    rev = fetch_concept("Revenues",
                         ("RevenueFromContractWithCustomerExcludingAssessedTax",
                          "SalesRevenueNet"))
    ni = fetch_concept("NetIncomeLoss")
    eps_entries = fetch_concept("EarningsPerShareDiluted",
                                ("EarningsPerShareBasic",))

    rev_a = annual_series(rev) / 1e9   # billions
    rev_q = quarterly_series(rev) / 1e9
    rev_ttm = ttm_from_quarterly(rev_q)

    ni_a = annual_series(ni) / 1e9
    ni_q = quarterly_series(ni) / 1e9
    ni_ttm = ttm_from_quarterly(ni_q)

    eps_a = annual_series(eps_entries)
    eps_q = quarterly_series(eps_entries)
    eps_ttm = ttm_from_quarterly(eps_q)

    # Split-adjust EPS using yfinance splits
    tk = yf.Ticker(TICKER)
    splits = tk.splits
    if splits is not None and not splits.empty:
        if getattr(splits.index, "tz", None) is not None:
            splits.index = splits.index.tz_localize(None)
        splits = splits.copy()
        splits.index = pd.DatetimeIndex(splits.index.date)
        def adjust(s: pd.Series) -> pd.Series:
            adjusted = s.copy()
            for end_ts, v in s.items():
                factor = 1.0
                for sd, ratio in splits.sort_index().items():
                    if sd > end_ts:
                        factor *= float(ratio)
                adjusted.loc[end_ts] = float(v) / factor
            return adjusted
        eps_a = adjust(eps_a)
        eps_q = adjust(eps_q)
        eps_ttm = adjust(eps_ttm)

    # Current price + recent
    info = tk.info if isinstance(tk.info, dict) else {}
    price = info.get("regularMarketPrice") or info.get("previousClose")
    market_cap = info.get("marketCap")
    pe = info.get("trailingPE")
    name = info.get("longName") or info.get("shortName") or "Apple Inc."

    last_rev_ttm = rev_ttm.dropna().iloc[-1]
    last_ni_ttm = ni_ttm.dropna().iloc[-1]
    last_eps_ttm = eps_ttm.dropna().iloc[-1]
    rev_yoy = None
    if len(rev_a) >= 2:
        rev_yoy = (rev_a.iloc[-1] / rev_a.iloc[-2] - 1.0) * 100
    ni_yoy = None
    if len(ni_a) >= 2:
        ni_yoy = (ni_a.iloc[-1] / ni_a.iloc[-2] - 1.0) * 100
    eps_yoy = None
    if len(eps_a) >= 2:
        eps_yoy = (eps_a.iloc[-1] / eps_a.iloc[-2] - 1.0) * 100

    print(f"  revenue annual rows: {len(rev_a)}  ({rev_a.index.min().date()} → {rev_a.index.max().date()})")
    print(f"  net income annual rows: {len(ni_a)}")
    print(f"  EPS annual rows: {len(eps_a)}")
    print(f"  current price: {price}, mcap: {market_cap}, PE: {pe}")

    # -------------------- Plotly chart ----------------------------------
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        subplot_titles=("Revenue (annual + TTM, $B)",
                        "Net Income (annual + TTM, $B)",
                        f"Diluted EPS ($)"),
    )

    # Common style for each pair
    bar_color = "#1f77b4"
    ttm_color = "#ff7f0e"

    fig.add_trace(go.Bar(x=rev_a.index, y=rev_a.values, name="Annual revenue",
                          marker_color=bar_color,
                          hovertemplate="%{x|%Y}: $%{y:,.1f}B<extra>Annual revenue</extra>"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=rev_ttm.index, y=rev_ttm.values, mode="lines",
                              name="TTM revenue", line=dict(color=ttm_color, width=2),
                              hovertemplate="%{x|%b %Y}: $%{y:,.1f}B<extra>TTM revenue</extra>"),
                  row=1, col=1)

    fig.add_trace(go.Bar(x=ni_a.index, y=ni_a.values, name="Annual net income",
                          marker_color=bar_color, showlegend=False,
                          hovertemplate="%{x|%Y}: $%{y:,.1f}B<extra>Annual NI</extra>"),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=ni_ttm.index, y=ni_ttm.values, mode="lines",
                              name="TTM net income", line=dict(color=ttm_color, width=2),
                              showlegend=False,
                              hovertemplate="%{x|%b %Y}: $%{y:,.1f}B<extra>TTM NI</extra>"),
                  row=2, col=1)

    fig.add_trace(go.Bar(x=eps_a.index, y=eps_a.values, name="Annual EPS",
                          marker_color=bar_color, showlegend=False,
                          hovertemplate="%{x|%Y}: $%{y:.2f}<extra>Annual EPS</extra>"),
                  row=3, col=1)
    fig.add_trace(go.Scatter(x=eps_ttm.index, y=eps_ttm.values, mode="lines",
                              name="TTM EPS", line=dict(color=ttm_color, width=2),
                              showlegend=False,
                              hovertemplate="%{x|%b %Y}: $%{y:.2f}<extra>TTM EPS</extra>"),
                  row=3, col=1)

    # Annotate the latest TTM point on each panel
    for series, row, fmt in [(rev_ttm, 1, "${:.1f}B"), (ni_ttm, 2, "${:.1f}B"), (eps_ttm, 3, "${:.2f}")]:
        s = series.dropna()
        if s.empty: continue
        fig.add_annotation(
            x=s.index[-1], y=s.iloc[-1],
            text=fmt.format(s.iloc[-1]),
            showarrow=True, arrowhead=2, arrowsize=1, arrowcolor="#ff7f0e",
            ax=-40, ay=-30, bgcolor="white", bordercolor="#ff7f0e", borderwidth=1,
            font=dict(size=12, color="#ff7f0e"),
            row=row, col=1,
        )

    # Layout - default 10Y window, but full data is available via MAX
    end = pd.Timestamp.today().normalize()
    default_start = end - pd.DateOffset(years=10)
    fig.update_layout(
        height=900,
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1),
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
        bargap=0.15,
        xaxis3=dict(
            range=[default_start, end],
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(count=3, label="3Y", step="year", stepmode="backward"),
                    dict(count=5, label="5Y", step="year", stepmode="backward"),
                    dict(count=10, label="10Y", step="year", stepmode="backward"),
                    dict(step="all", label="MAX"),
                ],
                bgcolor="#f4f8ff", activecolor="#1f77b4",
                font=dict(size=14), x=0, y=-0.15,
            ),
            rangeslider=dict(visible=False),
            type="date",
            showspikes=True, spikethickness=1, spikedash="dot", spikecolor="#666",
        ),
    )
    # propagate same x-range default to row 1/2 (shared_xaxes already syncs)
    fig.update_xaxes(showgrid=True, gridcolor="#eee")
    fig.update_yaxes(showgrid=True, gridcolor="#eee", zerolinecolor="#ccc")

    chart_html = fig.to_html(include_plotlyjs="cdn", full_html=False, config={
        "displayModeBar": False,
        "responsive": True,
        "scrollZoom": False,
    })

    # ---------------------- assemble final HTML --------------------------
    yoy_fmt = lambda v: f"<span style='color:{'#0a7' if (v or 0) > 0 else '#c33'}'>{v:+.1f}%</span>" if v is not None else "—"
    pe_str = f"{pe:.2f}" if isinstance(pe, (int, float)) else "—"
    price_str = f"${price:,.2f}" if isinstance(price, (int, float)) else "—"
    mcap_str = f"${market_cap/1e12:.2f}T" if isinstance(market_cap, (int, float)) else "—"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<title>{TICKER} — {name} (Plotly mockup)</title>
<style>
  :root {{
    --frame:#003366; --grid:#B0B0B0; --bg:#FFFFFF; --text:#1a2c47;
    --positive:#0a7; --negative:#c33; --highlight:#1f77b4;
  }}
  *{{box-sizing:border-box}}
  body{{margin:0;padding:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Verdana,Arial,sans-serif;background:#f6f8fb;color:var(--text);-webkit-font-smoothing:antialiased}}
  .wrap{{max-width:980px;margin:0 auto;padding:12px}}
  .ticker-card{{background:white;border:2px solid var(--frame);border-radius:12px;padding:14px 16px;margin-bottom:12px;box-shadow:0 1px 3px rgba(0,0,0,0.06)}}
  .ticker-name{{font-size:18px;font-weight:600;margin:0 0 2px;color:var(--frame)}}
  .ticker-sub{{font-size:13px;color:#666;margin:0 0 12px}}
  .values{{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:10px}}
  .v{{background:#f4f8ff;border-left:3px solid var(--highlight);padding:8px 10px;border-radius:4px}}
  .v-label{{font-size:11px;text-transform:uppercase;letter-spacing:0.04em;color:#557;margin-bottom:2px}}
  .v-num{{font-size:18px;font-weight:600;color:var(--frame);line-height:1.1}}
  .v-chg{{font-size:12px;margin-top:2px}}
  .chart-card{{background:white;border:1px solid #d0d8ea;border-radius:12px;padding:8px;margin-bottom:12px}}
  .meta{{font-size:12px;color:#666;text-align:center;padding:10px}}
  @media (max-width:640px){{
    .ticker-name{{font-size:16px}} .v-num{{font-size:16px}}
    .values{{grid-template-columns:repeat(2,1fr)}}
  }}
  a{{color:var(--highlight)}}
</style>
</head>
<body>
<div class="wrap">

  <div class="ticker-card">
    <h1 class="ticker-name">{name} ({TICKER})</h1>
    <p class="ticker-sub">Mobile-first Plotly mockup • EDGAR-sourced financials back to {min(rev_a.index.min().year, ni_a.index.min().year)}</p>
    <div class="values">
      <div class="v"><div class="v-label">Price</div><div class="v-num">{price_str}</div></div>
      <div class="v"><div class="v-label">Market Cap</div><div class="v-num">{mcap_str}</div></div>
      <div class="v"><div class="v-label">P/E (TTM)</div><div class="v-num">{pe_str}</div></div>
      <div class="v"><div class="v-label">Revenue TTM</div><div class="v-num">${last_rev_ttm:,.1f}B</div><div class="v-chg">vs annual YoY: {yoy_fmt(rev_yoy)}</div></div>
      <div class="v"><div class="v-label">Net Income TTM</div><div class="v-num">${last_ni_ttm:,.1f}B</div><div class="v-chg">vs annual YoY: {yoy_fmt(ni_yoy)}</div></div>
      <div class="v"><div class="v-label">EPS TTM</div><div class="v-num">${last_eps_ttm:,.2f}</div><div class="v-chg">vs annual YoY: {yoy_fmt(eps_yoy)}</div></div>
    </div>
  </div>

  <div class="chart-card">
    {chart_html}
  </div>

  <div class="meta">
    <strong>What this demonstrates:</strong> SEC EDGAR data (15+ year history) + Plotly interactive (mobile-friendly touch, range selector, sticky values up top, annotation on the latest TTM point).
    Pinch-zoom is disabled in the chart pane — drag the range buttons instead. Tap any bar/line for the value.
    <br><br>Generated {dt.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}.
  </div>

</div>
</body>
</html>"""

    OUT_FILE.write_text(html, encoding="utf-8")
    print(f"\nWrote {OUT_FILE}  ({OUT_FILE.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

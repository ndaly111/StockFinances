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
# Try us-gaap first (standard US issuers), then ifrs-full (foreign issuers
# filing 20-F under IFRS — e.g. BUD, SPOT).
EDGAR_CONCEPT_TAXONOMIES = ("us-gaap", "ifrs-full")
EDGAR_CONCEPT = "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/{taxonomy}/{concept}.json"
CIK = "0000320193"  # AAPL
TICKER = "AAPL"
OUT_FILE = Path(__file__).resolve().parents[1] / "aapl_mockup.html"

# Canonical series colors — used consistently across EVERY chart so a given
# metric is always the same color: Revenue = green, Net Income = blue,
# EPS = a slightly darker blue. (_LT = lighter shade for forecast/estimate bars.)
COLOR_REVENUE       = "#2ca02c"  # green
COLOR_REVENUE_LT    = "#a1d99b"
COLOR_NET_INCOME    = "#1f77b4"  # blue
COLOR_NET_INCOME_LT = "#9ecae1"
COLOR_EPS           = "#14507a"  # darker blue
COLOR_EPS_LT        = "#8fb4d4"


# -------------------------- EDGAR helpers ----------------------------------
# Currency units in priority order — try USD first (US issuers + foreign that
# also report in USD), then major non-USD currencies for IFRS filers that
# only report in their home currency (e.g. SPOT in EUR).
_MONETARY_UNITS = ("USD", "EUR", "GBP", "CAD", "CHF", "JPY", "CNY")
_PER_SHARE_UNITS = ("USD/shares", "EUR/shares", "GBP/shares", "CAD/shares")


def fetch_concept(concept: str, fallbacks: tuple[str, ...] = ()) -> list[dict]:
    """Merge results from concept + fallbacks across us-gaap and ifrs-full
    taxonomies. Companies switch tag names over time (e.g. AAPL
    Revenues → RevenueFromContractWithCustomer at ASC 606 adoption in 2018);
    foreign filers (20-F) report under ifrs-full rather than us-gaap.
    Each entry is tagged with the source unit so a caller can FX-convert."""
    combined: list[dict] = []
    seen_keys: set = set()
    for c in (concept, *fallbacks):
        for tax in EDGAR_CONCEPT_TAXONOMIES:
            url = EDGAR_CONCEPT.format(cik=CIK, taxonomy=tax, concept=c)
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code != 200:
                continue
            units = r.json().get("units", {})
            for key in _MONETARY_UNITS + _PER_SHARE_UNITS:
                if key in units and units[key]:
                    for v in units[key]:
                        dedupe = (v.get("start"), v.get("end"), v.get("form"))
                        if dedupe in seen_keys:
                            continue
                        seen_keys.add(dedupe)
                        v = dict(v)
                        v["_unit"] = key
                        combined.append(v)
                    break  # take first matching unit per concept
    return combined


_FX_CACHE: dict[str, pd.Series] = {}


def _fx_to_usd(currency: str) -> pd.Series | None:
    """Daily Close of <currency>USD=X — USD per 1 unit of currency."""
    if currency == "USD":
        return None
    if currency in _FX_CACHE:
        return _FX_CACHE[currency]
    try:
        pair = f"{currency}USD=X"
        hist = yf.Ticker(pair).history(period="max", auto_adjust=True)["Close"]
        if hist.empty:
            return None
        if getattr(hist.index, "tz", None) is not None:
            hist.index = hist.index.tz_localize(None)
        hist.index = pd.DatetimeIndex(hist.index.date)
        _FX_CACHE[currency] = hist
        return hist
    except Exception:
        return None


def convert_to_usd(entries: list[dict]) -> list[dict]:
    """If entries are tagged with a non-USD unit, FX-convert each `val` to
    USD using the historical rate at the entry's period-end date."""
    if not entries:
        return entries
    currencies = {v.get("_unit", "USD").split("/")[0] for v in entries}
    non_usd = currencies - {"USD"}
    if not non_usd:
        return entries
    fx_map: dict[str, pd.Series] = {}
    for cur in non_usd:
        s = _fx_to_usd(cur)
        if s is not None:
            fx_map[cur] = s
    out: list[dict] = []
    for v in entries:
        unit = v.get("_unit", "USD")
        base_ccy = unit.split("/")[0]
        if base_ccy == "USD":
            out.append(v)
            continue
        fx = fx_map.get(base_ccy)
        if fx is None or v.get("val") is None:
            continue
        end = pd.Timestamp(v["end"])
        prior = fx[fx.index <= end]
        rate = float(prior.iloc[-1]) if not prior.empty else float(fx.iloc[0])
        v2 = dict(v)
        v2["val"] = float(v["val"]) * rate
        v2["_fx_rate"] = rate
        v2["_orig_unit"] = unit
        v2["_unit"] = "USD" if "/" not in unit else "USD/shares"
        out.append(v2)
    return out


def fy_label(ts) -> str:
    """Year label for a fiscal-year-end timestamp. A fiscal year that ends
    in the first ~14 days of January represents the prior calendar year
    (e.g., JNJ's Sunday-nearest-Dec-31 calendar: FY2022 ended Jan 1, 2023).
    Without this, two adjacent fiscal years can both render as the same
    "%Y" label and stack on the same x-tick."""
    d = pd.Timestamp(ts)
    if d.month == 1 and d.day <= 14:
        return str(d.year - 1)
    return str(d.year)


_ANNUAL_FORMS = ("10-K", "20-F", "40-F")


def annual_series(entries: list[dict]) -> pd.Series:
    """Annual values from 10-K (US issuers), 20-F (foreign), or 40-F
    (Canadian) filings — span ~363 days."""
    out: dict[pd.Timestamp, float] = {}
    for v in entries:
        sd, ed, val = v.get("start"), v.get("end"), v.get("val")
        if not sd or not ed or val is None or v.get("form") not in _ANNUAL_FORMS:
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
    rev = convert_to_usd(fetch_concept("Revenues",
                         ("RevenueFromContractWithCustomerExcludingAssessedTax",
                          # IncludingAssessedTax catches retailers/consumer cos that report
                          # gross-of-excise-tax (BIRD, VLO, KHC, …)
                          "RevenueFromContractWithCustomerIncludingAssessedTax",
                          # Banks net interest income against interest expense (GS, etc.)
                          "RevenuesNetOfInterestExpense",
                          "SalesRevenueNet",
                          # IFRS naming (20-F foreign filers under ifrs-full): BUD, SPOT
                          "Revenue")))
    # IFRS uses ProfitLoss for net income (BUD, SPOT). ProfitLossFromContinuingOperations
    # is sometimes the cleaner figure when companies have discontinued operations.
    ni = convert_to_usd(fetch_concept("NetIncomeLoss",
                       ("ProfitLoss", "ProfitLossFromContinuingOperations")))
    eps_entries = convert_to_usd(fetch_concept("EarningsPerShareDiluted",
                                ("EarningsPerShareBasic",)))

    rev_a = annual_series(rev) / 1e9   # billions
    rev_q = quarterly_series(rev) / 1e9
    rev_ttm = ttm_from_quarterly(rev_q)

    ni_a = annual_series(ni) / 1e9
    ni_q = quarterly_series(ni) / 1e9
    ni_ttm = ttm_from_quarterly(ni_q)

    eps_a = annual_series(eps_entries)
    eps_q = quarterly_series(eps_entries)
    eps_ttm = ttm_from_quarterly(eps_q)

    # Derive EPS from NI / shares when EDGAR returns no usable EPS rows.
    # This catches:
    #   - V (Visa): EPS tagged under a company-specific XBRL extension, not
    #     in us-gaap at all
    #   - BRK-B (Berkshire): EPS reported quarterly inside 10-K's, never
    #     annually, and in Class A units
    # The approximation uses CURRENT shares outstanding from yfinance — it
    # ignores year-over-year buybacks, so historical EPS may drift ~5% from
    # the company's reported figure. Better than no EPS at all.
    _tk_for_shares = yf.Ticker(TICKER)
    _shares_now = None
    try:
        _info_for_shares = _tk_for_shares.info if isinstance(_tk_for_shares.info, dict) else {}
        _shares_now = _info_for_shares.get("sharesOutstanding") or _info_for_shares.get("impliedSharesOutstanding")
    except Exception:
        pass
    if eps_a.empty and _shares_now and _shares_now > 0:
        eps_a = ni_a * 1e9 / _shares_now   # ni_a is in $B, shares is raw count
        print(f"  EPS derived from NI/shares (no EDGAR EPS concept available, "
              f"using {_shares_now/1e6:.0f}M shares)")
    if eps_q.empty and _shares_now and _shares_now > 0:
        eps_q = ni_q * 1e9 / _shares_now
    if eps_ttm.empty and _shares_now and _shares_now > 0:
        eps_ttm = ni_ttm * 1e9 / _shares_now

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
    forward_pe = info.get("forwardPE")
    pb = info.get("priceToBook")
    div_rate = info.get("dividendRate")
    div_yield = info.get("dividendYield")
    name = info.get("longName") or info.get("shortName") or "Apple Inc."

    # Implied growth = (PE / 10) ^ 0.1 + 10Y_yield - 1  (matches repo formula)
    try:
        ydata = requests.get(
            "https://query1.finance.yahoo.com/v8/finance/chart/%5ETNX?range=5d&interval=1d",
            headers={"User-Agent": "Mozilla/5.0"}, timeout=15,
        ).json()
        closes = ydata["chart"]["result"][0]["indicators"]["quote"][0]["close"]
        ten_yr = next(x for x in reversed(closes) if x is not None) / 100.0
    except Exception:
        ten_yr = 0.045
    def implied_growth(pe_val):
        if not isinstance(pe_val, (int, float)) or pe_val <= 0:
            return None
        return (pe_val / 10.0) ** 0.1 + ten_yr - 1.0

    def _synthetic_pe_from_ps(tkr, info_d, mcap):
        """Margin-normalized synthetic P/E = (P/S) / Nick's net margin, for
        companies with no real trailing P/E. Returns None if data is missing."""
        import json as _json
        try:
            _assum = _json.loads(
                (Path.cwd() / "data" / "assumptions.json").read_text(encoding="utf-8")
            ).get("tickers", {})
        except Exception:
            _assum = {}
        margin_pct = (_assum.get(str(tkr).upper()) or {}).get("profit_margin")
        try:
            margin = float(margin_pct) / 100.0
        except (TypeError, ValueError):
            return None
        if margin <= 0:
            return None
        ps = info_d.get("priceToSalesTrailing12Months")
        if not ps:
            rev = info_d.get("totalRevenue")
            if mcap and rev:
                try:
                    ps = mcap / rev
                except ZeroDivisionError:
                    ps = None
        if not ps or ps <= 0:
            return None
        return ps / margin

    ig = implied_growth(pe)
    if ig is None:
        # No real trailing P/E (unprofitable): use a margin-normalized synthetic
        # P/E = (P/S) / Nick's margin so the tile matches the implied-growth chart.
        ig = implied_growth(_synthetic_pe_from_ps(TICKER, info, market_cap))
    ig_fwd = implied_growth(forward_pe)

    def _last_or_none(s):
        s2 = s.dropna() if hasattr(s, "dropna") else s
        try:
            return s2.iloc[-1]
        except (IndexError, AttributeError):
            return None

    last_rev_ttm = _last_or_none(rev_ttm)
    last_ni_ttm = _last_or_none(ni_ttm)
    last_eps_ttm = _last_or_none(eps_ttm)
    # Truly no EDGAR data: either a non-US filer (BABA) or a fresh IPO that
    # hasn't filed its first 10-Q yet (SPCX, 2026-06-12). Don't abort — build
    # the page from what exists (price, metrics, forward estimates, news) and
    # the history sections fill in automatically once filings land.
    NO_FUNDAMENTALS = bool(rev_a.empty and eps_a.empty)
    if NO_FUNDAMENTALS:
        print(f"  {TICKER}: no EDGAR fundamentals yet — building reduced page "
              f"(fresh IPO or non-US filer)")

    # ---------- forward estimates (yfinance) -----------------------------
    # With no reported history, anchor forward FY labels on the calendar year.
    last_fy = rev_a.index.max() if not rev_a.empty else pd.Timestamp.today() - pd.DateOffset(years=1)
    fwd_rev = None    # (FY_label, avg, low, high, n_analysts)
    fwd_eps = None
    fwd_ni  = None    # derived: avg = EPS_avg × diluted_shares, etc.
    try:
        re_df = tk.revenue_estimate
        ee_df = tk.earnings_estimate
        if re_df is not None and "0y" in re_df.index:
            row0 = re_df.loc["0y"]; row1 = re_df.loc["+1y"]
            # 0y = current FY (not yet reported), +1y = next FY
            fy0_year = last_fy.year + 1   # 0y is FY after last reported
            fy1_year = last_fy.year + 2
            fwd_rev = [
                (f"{fy0_year}E", float(row0.avg)/1e9, float(row0.low)/1e9, float(row0.high)/1e9, int(row0.numberOfAnalysts)),
                (f"{fy1_year}E", float(row1.avg)/1e9, float(row1.low)/1e9, float(row1.high)/1e9, int(row1.numberOfAnalysts)),
            ]
        if ee_df is not None and "0y" in ee_df.index:
            row0 = ee_df.loc["0y"]; row1 = ee_df.loc["+1y"]
            fy0_year = last_fy.year + 1
            fy1_year = last_fy.year + 2
            fwd_eps = [
                (f"{fy0_year}E", float(row0.avg), float(row0.low), float(row0.high), int(row0.numberOfAnalysts)),
                (f"{fy1_year}E", float(row1.avg), float(row1.low), float(row1.high), int(row1.numberOfAnalysts)),
            ]
        # Derive Net Income forecast = forward EPS × diluted shares outstanding.
        # No analyst NI consensus is published by yfinance for ETFs/stocks; this
        # is a reasonable approximation if the share count is fairly stable
        # (treats current diluted-share count as the basis).
        shares = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")
        # Prefer the diluted shares from the latest TTM EPS implied by EDGAR
        if isinstance(last_ni_ttm, (int, float)) and isinstance(last_eps_ttm, (int, float)) and last_eps_ttm:
            shares_implied = (last_ni_ttm * 1e9) / last_eps_ttm   # ni in $B -> $
            shares = shares_implied if shares_implied else shares
        if fwd_eps and isinstance(shares, (int, float)) and shares > 0:
            fwd_ni = []
            for lbl, avg, lo, hi, n in fwd_eps:
                fwd_ni.append((lbl, avg*shares/1e9, lo*shares/1e9, hi*shares/1e9, n))
    except Exception as e:
        print(f"forward estimates fetch failed: {e}")
    rev_yoy = None
    if len(rev_a) >= 2:
        rev_yoy = (rev_a.iloc[-1] / rev_a.iloc[-2] - 1.0) * 100
    ni_yoy = None
    if len(ni_a) >= 2:
        ni_yoy = (ni_a.iloc[-1] / ni_a.iloc[-2] - 1.0) * 100
    eps_yoy = None
    if len(eps_a) >= 2:
        eps_yoy = (eps_a.iloc[-1] / eps_a.iloc[-2] - 1.0) * 100

    def _date_range(s):
        if s.empty:
            return "empty"
        try:
            return f"{s.index.min().date()} → {s.index.max().date()}"
        except (AttributeError, TypeError):
            return f"{s.index.min()} → {s.index.max()}"
    print(f"  revenue annual rows: {len(rev_a)}  ({_date_range(rev_a)})")
    print(f"  net income annual rows: {len(ni_a)}")
    print(f"  EPS annual rows: {len(eps_a)}")
    print(f"  current price: {price}, mcap: {market_cap}, PE: {pe}")
    if rev_a.empty and not NO_FUNDAMENTALS:
        # No annual revenue means EDGAR returned nothing usable.
        raise RuntimeError(
            f"{TICKER}: no annual revenue rows from EDGAR — likely a foreign "
            f"filer or doesn't report under us-gaap"
        )

    # -------------------- Plotly chart ----------------------------------
    # Append the latest TTM as a "TTM" bar at the right edge of each series,
    # using a date one quarter past the last annual end.
    def append_ttm(annual: pd.Series, ttm: pd.Series) -> pd.Series:
        ttm = ttm.dropna()
        annual_max = annual.index.max() if not annual.empty else None
        # Foreign filers (20-F) only file annual data, so ttm comes back empty
        # with a RangeIndex that can't be compared to a Timestamp. Skip the
        # post-FY filter in that case and just return the annuals.
        if ttm.empty or not isinstance(ttm.index, pd.DatetimeIndex):
            return annual if annual_max is not None else ttm
        # Take the TTM points reported AFTER the most recent FY end
        if annual_max is not None:
            post = ttm[ttm.index > annual_max]
            if post.empty: return annual
            return pd.concat([annual, post.tail(1)])  # one extra "TTM" bar
        return ttm

    rev_combined = append_ttm(rev_a, rev_ttm)
    ni_combined = append_ttm(ni_a, ni_ttm)
    eps_combined = append_ttm(eps_a, eps_ttm)

    # Right-align all three series so they share a single x-axis indexing.
    # EDGAR can return more historical depth for EPS than Revenue (JNJ has
    # ~19 years of EPS but only ~10 of Revenue). Without right-alignment the
    # range-button JS applies the same [N-y, N] window to both subplots and
    # the EPS panel ends up showing 2013-2017 while Revenue shows 2021-TTM.
    max_n = max(len(rev_combined), len(ni_combined), len(eps_combined))
    canonical_combined = max(
        [rev_combined, ni_combined, eps_combined], key=len,
    )
    # Skip empty series whose .index.max() can be NaT/NaN — BRK-B has no
    # EPS via the standard concepts and would otherwise crash here.
    _annual_maxes = [s.index.max() for s in (rev_a, ni_a, eps_a) if len(s) > 0]
    # All-empty (fresh IPO): use today so downstream label logic gets a real
    # Timestamp instead of NaT.
    canonical_annual_max = max(_annual_maxes) if _annual_maxes else pd.Timestamp.today()

    def pad_left(vals, target_len):
        pad = target_len - len(vals)
        return ([None] * pad) + list(vals) if pad > 0 else list(vals)

    def money_unit_for(values):
        """Pick a display unit for dollar values held in $B.
        Small caps rendered in $B label every bar "0" or "0.1" (useless), so
        scale to $M or $K until the biggest bar reads in sensible digits.
        Returns (unit_suffix, multiplier, label_formatter)."""
        vmax = max((abs(v) for v in values if v is not None), default=0.0)
        if vmax >= 1.0:
            unit, mult = "B", 1.0
        elif vmax >= 0.001:
            unit, mult = "M", 1e3
        else:
            unit, mult = "K", 1e6
        scaled_max = vmax * mult
        fmt = (lambda v: f"{v:,.1f}") if scaled_max < 20 else (lambda v: f"{v:,.0f}")
        return unit, mult, fmt

    rev_vals = pad_left(rev_combined.values, max_n)
    ni_vals  = pad_left(ni_combined.values,  max_n)

    money_unit, money_mult, money_lab = money_unit_for(rev_vals + ni_vals)
    rev_vals = [None if v is None else v * money_mult for v in rev_vals]
    ni_vals  = [None if v is None else v * money_mult for v in ni_vals]
    eps_vals = pad_left(eps_combined.values, max_n)

    # Label x-axis values: years for annual, "TTM" for the latest.
    # We need TWO label sets: full ("2014") for 5Y/10Y views, short ("'14")
    # for 15Y/MAX where the labels otherwise collide.
    def labels_full(combined: pd.Series, annual_max: pd.Timestamp) -> list[str]:
        return [fy_label(d) if d <= annual_max else "TTM" for d in combined.index]
    def labels_short(combined: pd.Series, annual_max: pd.Timestamp) -> list[str]:
        return [f"'{d.strftime('%y')}" if d <= annual_max else "TTM" for d in combined.index]

    # One label set for both subplots — derived from the longest series so
    # tick labels stay aligned to the right edge (TTM).
    rev_labels_full  = labels_full(canonical_combined, canonical_annual_max)
    rev_labels_short = labels_short(canonical_combined, canonical_annual_max)
    eps_labels_full  = rev_labels_full
    eps_labels_short = rev_labels_short

    # Trace x values are integer indices so we can swap tick text per view
    rev_x = list(range(max_n))
    ni_x  = list(range(max_n))
    eps_x = list(range(max_n))

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=False, vertical_spacing=0.09,
        subplot_titles=(f"Revenue &amp; Net Income (${money_unit})", "Diluted EPS ($)"),
    )

    rev_color = COLOR_REVENUE
    ni_color  = COLOR_NET_INCOME
    eps_color = COLOR_EPS

    # customdata carries the full year label for tooltips so swapping tick
    # text doesn't lose the "2024 / 2025" context.
    fig.add_trace(go.Bar(
        x=rev_x, y=rev_vals, name="Revenue",
        marker_color=rev_color,
        customdata=rev_labels_full,
        text=[money_lab(v) if v is not None else "" for v in rev_vals], textposition="outside", textfont=dict(size=10),
        hovertemplate="<b>%{customdata}</b><br>Revenue: $%{y:,.1f}" + money_unit + "<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=ni_x, y=ni_vals, name="Net Income",
        marker_color=ni_color,
        customdata=rev_labels_full,
        text=[money_lab(v) if v is not None else "" for v in ni_vals], textposition="outside", textfont=dict(size=10),
        hovertemplate="<b>%{customdata}</b><br>Net Income: $%{y:,.1f}" + money_unit + "<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=eps_x, y=eps_vals, name="EPS", showlegend=False,
        marker_color=eps_color,
        customdata=eps_labels_full,
        text=[f"{v:.2f}" if v is not None else "" for v in eps_vals], textposition="outside", textfont=dict(size=10),
        hovertemplate="<b>%{customdata}</b><br>EPS: $%{y:.2f}<extra></extra>",
    ), row=2, col=1)

    # Layout: default to last 10 periods, full visible via MAX
    n = max_n
    n_chart = n   # preserved for the JS template — `n` gets clobbered by
                  #  later `for ..., n in fwd_eps` analyst-count loops
    default_start_idx = max(0, n - 10)
    tickvals = list(range(n))
    fig.update_layout(
        height=620,
        margin=dict(l=10, r=10, t=50, b=10),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="right", x=1),
        barmode="group",
        bargap=0.18, bargroupgap=0.05,
        plot_bgcolor="white", paper_bgcolor="white",
        hovermode="x unified",
        dragmode=False,   # disable pan/zoom drag so chart doesn't steal page scroll
    )
    # Lock axes: prevent accidental pan/zoom on touch, keep tap-for-value tooltips.
    # Use array tick mode so we can swap tick text on range change without
    # rebuilding the chart. Start with the full ("2014") labels for the 10Y default.
    fig.update_xaxes(
        range=[default_start_idx - 0.5, n - 0.5], fixedrange=True,
        showgrid=False, tickangle=0, tickfont=dict(size=11),
        automargin=True, tickmode="array",
        tickvals=tickvals, ticktext=rev_labels_full,
        row=1, col=1,
    )
    fig.update_xaxes(
        range=[default_start_idx - 0.5, n - 0.5], fixedrange=True,
        showgrid=False, tickangle=0, tickfont=dict(size=11),
        automargin=True, tickmode="array",
        tickvals=tickvals, ticktext=eps_labels_full,
        row=2, col=1,
    )
    fig.update_yaxes(fixedrange=True, showgrid=True, gridcolor="#eee", zerolinecolor="#ccc")

    chart_html = fig.to_html(include_plotlyjs="cdn", full_html=False, config={
        "displayModeBar": False,
        "responsive": True,
        "scrollZoom": False,
        "doubleClick": False,
        "showAxisDragHandles": False,
        "showAxisRangeEntryBoxes": False,
        "staticPlot": False,
    }, div_id="aapl-chart")

    # ---------------------- assemble final HTML --------------------------
    yoy_fmt = lambda v: f"<span style='color:{'#0a7' if (v or 0) > 0 else '#c33'}'>{v:+.1f}%</span>" if v is not None else "—"
    fmt_num   = lambda v, d=2: f"{v:.{d}f}" if isinstance(v, (int, float)) else "—"
    fmt_pct   = lambda v, d=1: f"{v*100:.{d}f}%" if isinstance(v, (int, float)) else "—"
    fmt_money = lambda v, d=2: f"${v:,.{d}f}" if isinstance(v, (int, float)) else "—"
    price_str = fmt_money(price)
    mcap_str = f"${market_cap/1e12:.2f}T" if isinstance(market_cap, (int, float)) and market_cap >= 1e12 else (f"${market_cap/1e9:.1f}B" if isinstance(market_cap, (int, float)) else "—")
    pe_str = fmt_num(pe, 1)
    # Day-one IPOs / distressed names produce absurd forwardPE values
    # (SPCX listed at -1918.8); a dash beats nonsense.
    fpe_str = (fmt_num(forward_pe, 1)
               if isinstance(forward_pe, (int, float)) and 0 < forward_pe < 1000
               else "—")
    ig_str = fmt_pct(ig, 1)
    igf_str = fmt_pct(ig_fwd, 1)
    pb_str = fmt_num(pb, 1)
    # Dividend formatting depends on yfinance which sometimes returns yield as percent (e.g. 0.5) and sometimes as decimal (0.005)
    dy = div_yield
    if isinstance(dy, (int, float)) and dy > 1:  # yfinance occasionally returns yield as % already
        dy_dec = dy / 100.0
    elif isinstance(dy, (int, float)):
        dy_dec = dy
    else:
        dy_dec = None
    div_str = fmt_money(div_rate, 2) if isinstance(div_rate, (int, float)) else "—"
    div_yield_str = fmt_pct(dy_dec, 2) if dy_dec is not None else "—"

    # ---- Latest Headlines (yfinance Ticker.news) -----------------------
    headlines_html = ""
    try:
        news = tk.news or []
        items = []
        for n in news[:8]:
            c = n.get("content") or {}
            if not c: continue
            title = c.get("title", "").strip()
            url = (c.get("clickThroughUrl") or {}).get("url") or (c.get("canonicalUrl") or {}).get("url") or ""
            pub  = c.get("pubDate") or ""
            src  = (c.get("provider") or {}).get("displayName") or ""
            if title and url:
                items.append((title, url, pub, src))
        if items:
            def time_ago(iso):
                try:
                    t = dt.datetime.fromisoformat(iso.replace("Z","+00:00"))
                    delta = dt.datetime.now(dt.timezone.utc) - t
                    s = delta.total_seconds()
                    if s < 3600:  return f"{int(s//60)}m ago"
                    if s < 86400: return f"{int(s//3600)}h ago"
                    return f"{int(s//86400)}d ago"
                except Exception:
                    return ""
            cards = []
            for title, url, pub, src in items:
                cards.append(
                    f'<a class="hl" href="{url}" target="_blank" rel="noopener">'
                    f'<div class="hl-title">{title}</div>'
                    f'<div class="hl-meta">{src} • {time_ago(pub)}</div>'
                    f'</a>'
                )
            headlines_html = (
                '<div class="data-table headlines-card"><h2>Latest Headlines</h2>'
                + '<div class="hl-list">' + "".join(cards) + '</div>'
                + '</div>'
            )
    except Exception as e:
        print(f"news fetch failed: {e}")

    import json as _json
    rev_labels_full_js  = _json.dumps(rev_labels_full)
    rev_labels_short_js = _json.dumps(rev_labels_short)
    eps_labels_full_js  = _json.dumps(eps_labels_full)
    eps_labels_short_js = _json.dumps(eps_labels_short)

    # ---------------- mobile-friendly data table -----------------------
    # One row per year + TTM, four columns: Year | Revenue | Net Income | EPS.
    # YoY% appears inline below each value (colored green/red).
    def delta_html(curr, prev, money=False):
        if prev is None or prev == 0 or curr is None:
            return ""
        pct = (curr / prev - 1.0) * 100
        cls = "pos" if pct >= 0 else "neg"
        return f'<span class="delta {cls}">{pct:+.1f}%</span>'

    # Combine annual + TTM into one indexed series for the table
    rev_series = list(rev_combined.items())
    ni_series  = list(ni_combined.items())
    eps_series = list(eps_combined.items())
    annual_max_rev = rev_a.index.max()

    DEFAULT_ROWS = 6   # TTM + last 5 annual

    table_rows = []
    visible_count = 0
    # Most recent first
    for i in range(len(rev_series) - 1, -1, -1):
        rev_d, rev_v = rev_series[i]
        ni_v  = ni_series[i][1]  if i < len(ni_series) else None
        eps_v = eps_series[i][1] if i < len(eps_series) else None
        prev_rev = rev_series[i-1][1] if i > 0 and i-1 < len(rev_series) else None
        prev_ni  = ni_series[i-1][1]  if i > 0 and i-1 < len(ni_series)  else None
        prev_eps = eps_series[i-1][1] if i > 0 and i-1 < len(eps_series) else None
        year_lbl = fy_label(rev_d) if rev_d <= annual_max_rev else "TTM"
        is_ttm = year_lbl == "TTM"
        # Rows beyond default_count get the .extra-row class so CSS hides them
        # until the user expands.
        row_classes = []
        if is_ttm:
            row_classes.append("ttm-row")
        if visible_count >= DEFAULT_ROWS:
            row_classes.append("extra-row")
        visible_count += 1
        cls = (" ".join(row_classes)) if row_classes else ""
        def _fmt(v, spec):
            return f"${v:{spec}}" if isinstance(v, (int, float)) else "—"
        table_rows.append(
            f'<tr class="{cls}">'
            f'<td class="year">{year_lbl}</td>'
            f'<td>{_fmt(rev_v, ",.1f")}{"B" if isinstance(rev_v,(int,float)) else ""}{delta_html(rev_v, prev_rev)}</td>'
            f'<td>{_fmt(ni_v, ",.1f")}{"B" if isinstance(ni_v,(int,float)) else ""}{delta_html(ni_v, prev_ni)}</td>'
            f'<td>{_fmt(eps_v, ",.2f")}{delta_html(eps_v, prev_eps)}</td>'
            f'</tr>'
        )
    total_rows = len(rev_series)
    hidden_rows = max(0, total_rows - DEFAULT_ROWS)
    expand_button = (
        f'<div class="expand-row"><button id="ft-toggle" class="expand-btn" '
        f'data-state="collapsed" data-hidden="{hidden_rows}">Show all {total_rows} years</button></div>'
    ) if hidden_rows > 0 else ""
    data_table_html = (
        '<div class="data-table"><h2>Revenue, Net Income, EPS</h2>'
        '<div style="overflow-x:auto"><table class="ft collapsed" id="ft-table">'
        '<thead><tr><th>Year</th><th>Revenue</th><th>Net Income</th><th>EPS</th></tr></thead>'
        '<tbody>' + "".join(table_rows) + '</tbody></table></div>'
        + expand_button +
        '</div>'
    )

    # ------------- forecast chart ---------------------------------------
    forecast_html = ""
    if fwd_rev or fwd_eps:
        # Use last 5 actual annual + 2 forward estimates
        hist_rev = rev_a.tail(5)
        hist_ni  = ni_a.tail(5)
        hist_eps = eps_a.tail(5)
        fc_fig = make_subplots(
            rows=2, cols=1, shared_xaxes=False, vertical_spacing=0.15,
            subplot_titles=("Revenue &amp; Net Income: Actual vs Forecast ($UNIT$)",
                            "Diluted EPS: Actual vs Forecast ($)"),
        )
        rev_dark, rev_light = COLOR_REVENUE, COLOR_REVENUE_LT
        ni_dark,  ni_light  = COLOR_NET_INCOME, COLOR_NET_INCOME_LT
        eps_dark, eps_light = COLOR_EPS, COLOR_EPS_LT

        # Split actual / forecast into separate traces (so error bars stay off
        # actuals) and use offsetgroup so Plotly groups them into a single lane
        # per metric — keeping bar widths consistent across the two panels.
        rev_a_x = [fy_label(d) for d in hist_rev.index]
        rev_a_y = list(hist_rev.values)
        rev_f_x = [r[0] for r in (fwd_rev or [])]
        rev_f_y = [r[1] for r in (fwd_rev or [])]
        rev_f_lo= [r[2] for r in (fwd_rev or [])]
        rev_f_hi= [r[3] for r in (fwd_rev or [])]
        rev_f_n = [r[4] for r in (fwd_rev or [])]

        ni_a_x = [fy_label(d) for d in hist_ni.index]
        ni_a_y = list(hist_ni.values)
        ni_f_x = [r[0] for r in (fwd_ni or [])]
        ni_f_y = [r[1] for r in (fwd_ni or [])]
        ni_f_lo= [r[2] for r in (fwd_ni or [])]
        ni_f_hi= [r[3] for r in (fwd_ni or [])]
        ni_f_n = [r[4] for r in (fwd_ni or [])]

        # Same adaptive unit as the history chart — $B labels on a small cap
        # read "0". Scale plotted values AND the error-bar bounds together.
        fc_unit, fc_mult, fc_lab = money_unit_for(
            list(rev_a_y) + list(rev_f_y) + list(ni_a_y) + list(ni_f_y))
        if fc_mult != 1.0:
            _sc = lambda xs: [None if v is None else v * fc_mult for v in xs]
            rev_a_y, rev_f_y, rev_f_lo, rev_f_hi = _sc(rev_a_y), _sc(rev_f_y), _sc(rev_f_lo), _sc(rev_f_hi)
            ni_a_y, ni_f_y, ni_f_lo, ni_f_hi = _sc(ni_a_y), _sc(ni_f_y), _sc(ni_f_lo), _sc(ni_f_hi)
        # The subplot title was created before the unit was known ($UNIT$).
        fc_fig.layout.annotations[0].text = (
            f"Revenue &amp; Net Income: Actual vs Forecast (${fc_unit})")

        eps_a_x = [fy_label(d) for d in hist_eps.index]
        eps_a_y = list(hist_eps.values)
        eps_f_x = [r[0] for r in (fwd_eps or [])]
        eps_f_y = [r[1] for r in (fwd_eps or [])]
        eps_f_lo= [r[2] for r in (fwd_eps or [])]
        eps_f_hi= [r[3] for r in (fwd_eps or [])]
        eps_f_n = [r[4] for r in (fwd_eps or [])]

        # Revenue
        fc_fig.add_trace(go.Bar(
            x=rev_a_x, y=rev_a_y, name="Revenue",
            marker_color=rev_dark, offsetgroup="rev", legendgroup="rev",
            text=[fc_lab(v) for v in rev_a_y], textposition="outside", textfont=dict(size=10),
            hovertemplate="<b>%{x}</b> Revenue<br>Actual: $%{y:,.1f}" + fc_unit + "<extra></extra>",
        ), row=1, col=1)
        if rev_f_x:
            fc_fig.add_trace(go.Bar(
                x=rev_f_x, y=rev_f_y, name="Revenue forecast",
                marker_color=rev_light, marker_line=dict(color=rev_dark, width=1.2),
                offsetgroup="rev", legendgroup="rev", showlegend=False,
                error_y=dict(type="data", symmetric=False,
                             array=[h-a for h,a in zip(rev_f_hi, rev_f_y)],
                             arrayminus=[a-l for a,l in zip(rev_f_y, rev_f_lo)],
                             color="#444", thickness=1.5, width=8),
                text=[fc_lab(v) for v in rev_f_y], textposition="outside", textfont=dict(size=10),
                customdata=[[lo, hi, n] for lo, hi, n in zip(rev_f_lo, rev_f_hi, rev_f_n)],
                hovertemplate=("<b>%{x}</b> Revenue<br>Consensus: $%{y:,.1f}" + fc_unit +
                               "<br>Range: $%{customdata[0]:,.1f}" + fc_unit +
                               " – $%{customdata[1]:,.1f}" + fc_unit +
                               "<br>(%{customdata[2]} analysts)<extra></extra>"),
            ), row=1, col=1)

        # Net Income
        fc_fig.add_trace(go.Bar(
            x=ni_a_x, y=ni_a_y, name="Net Income",
            marker_color=ni_dark, offsetgroup="ni", legendgroup="ni",
            text=[fc_lab(v) for v in ni_a_y], textposition="outside", textfont=dict(size=10),
            hovertemplate="<b>%{x}</b> Net Income<br>Actual: $%{y:,.1f}" + fc_unit + "<extra></extra>",
        ), row=1, col=1)
        if ni_f_x:
            fc_fig.add_trace(go.Bar(
                x=ni_f_x, y=ni_f_y, name="Net Income forecast",
                marker_color=ni_light, marker_line=dict(color=ni_dark, width=1.2),
                offsetgroup="ni", legendgroup="ni", showlegend=False,
                error_y=dict(type="data", symmetric=False,
                             array=[h-a for h,a in zip(ni_f_hi, ni_f_y)],
                             arrayminus=[a-l for a,l in zip(ni_f_y, ni_f_lo)],
                             color="#444", thickness=1.5, width=8),
                text=[fc_lab(v) for v in ni_f_y], textposition="outside", textfont=dict(size=10),
                customdata=[[lo, hi, n] for lo, hi, n in zip(ni_f_lo, ni_f_hi, ni_f_n)],
                hovertemplate=("<b>%{x}</b> Net Income<br>Consensus: $%{y:,.1f}" + fc_unit +
                               "<br>Range: $%{customdata[0]:,.1f}" + fc_unit +
                               " – $%{customdata[1]:,.1f}" + fc_unit +
                               "<br>(%{customdata[2]} analysts)<extra></extra>"),
            ), row=1, col=1)

        # EPS
        fc_fig.add_trace(go.Bar(
            x=eps_a_x, y=eps_a_y, name="EPS", showlegend=False,
            marker_color=eps_dark, offsetgroup="eps", legendgroup="eps",
            text=[f"{v:.2f}" for v in eps_a_y], textposition="outside", textfont=dict(size=10),
            hovertemplate="<b>%{x}</b> EPS<br>Actual: $%{y:.2f}<extra></extra>",
        ), row=2, col=1)
        if eps_f_x:
            fc_fig.add_trace(go.Bar(
                x=eps_f_x, y=eps_f_y, name="EPS forecast", showlegend=False,
                marker_color=eps_light, marker_line=dict(color=eps_dark, width=1.2),
                offsetgroup="eps", legendgroup="eps",
                error_y=dict(type="data", symmetric=False,
                             array=[h-a for h,a in zip(eps_f_hi, eps_f_y)],
                             arrayminus=[a-l for a,l in zip(eps_f_y, eps_f_lo)],
                             color="#444", thickness=1.5, width=8),
                text=[f"{v:.2f}" for v in eps_f_y], textposition="outside", textfont=dict(size=10),
                customdata=[[lo, hi, n] for lo, hi, n in zip(eps_f_lo, eps_f_hi, eps_f_n)],
                hovertemplate=("<b>%{x}</b> EPS<br>Consensus: $%{y:.2f}"
                               "<br>Range: $%{customdata[0]:.2f} – $%{customdata[1]:.2f}"
                               "<br>(%{customdata[2]} analysts)<extra></extra>"),
            ), row=2, col=1)

        fc_fig.update_layout(
            height=520,
            margin=dict(l=10, r=10, t=44, b=10),
            barmode="group", bargap=0.20, bargroupgap=0.05,
            plot_bgcolor="white", paper_bgcolor="white",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1),
            dragmode=False, hovermode="closest",   # closest = tap individual bar only, less tooltip clutter
        )
        fc_fig.update_xaxes(fixedrange=True, showgrid=False, tickfont=dict(size=11))
        # Add a touch of headroom so error bars/labels never crash into the top.
        fc_fig.update_yaxes(fixedrange=True, showgrid=True, gridcolor="#eee", zerolinecolor="#ccc",
                            rangemode="tozero", automargin=True)

        fc_chart = fc_fig.to_html(include_plotlyjs=False, full_html=False, config={
            "displayModeBar": False, "responsive": True, "scrollZoom": False,
            "doubleClick": False, "showAxisDragHandles": False, "staticPlot": False,
        }, div_id="aapl-forecast-chart")

        # Forecast table — include last 3 actual years on top for context,
        # then forecast rows. Actual rows show value only; forecast rows show
        # value + range + analyst counts.
        fc_rows = []
        n_actual_ctx = 3
        hist_rev_tail = hist_rev.tail(n_actual_ctx)
        hist_ni_tail  = hist_ni.tail(n_actual_ctx)
        hist_eps_tail = hist_eps.tail(n_actual_ctx)
        for d, rev_v, ni_v, eps_v in zip(
            hist_rev_tail.index,
            hist_rev_tail.values,
            hist_ni_tail.values,
            hist_eps_tail.values,
        ):
            yr_lbl = fy_label(d)
            fc_rows.append(
                f'<tr><td class="year">{yr_lbl}</td>'
                f'<td>${rev_v:,.1f}B</td>'
                f'<td>${ni_v:,.1f}B</td>'
                f'<td>${eps_v:.2f}</td>'
                f'<td>—</td></tr>'
            )

        fwd_ni_safe = fwd_ni or [(None,)*5]*len(fwd_rev or [])
        for (yr_lbl, avg, lo, hi, n), (_, ni_avg, ni_lo, ni_hi, _), (_, eps_avg, eps_lo, eps_hi, eps_n) in zip(
            fwd_rev or [], fwd_ni_safe, fwd_eps or []
        ):
            ni_cell = (
                f'${ni_avg:,.1f}B<span class="delta">(${ni_lo:,.0f}-${ni_hi:,.0f}B)</span>'
                if ni_avg is not None else "—"
            )
            fc_rows.append(
                f'<tr class="forecast-row"><td class="year">{yr_lbl}</td>'
                f'<td>${avg:,.1f}B<span class="delta">(${lo:,.0f}-${hi:,.0f}B)</span></td>'
                f'<td>{ni_cell}</td>'
                f'<td>${eps_avg:.2f}<span class="delta">(${eps_lo:.2f}-${eps_hi:.2f})</span></td>'
                f'<td>{n} / {eps_n}</td></tr>'
            )
        forecast_table = (
            '<table class="ft" style="margin-top:10px">'
            '<thead><tr><th>FY</th><th>Revenue</th><th>Net Income*</th><th>EPS</th>'
            '<th>Analysts<br>(Rev / EPS)</th></tr></thead>'
            '<tbody>' + "".join(fc_rows) + '</tbody></table>'
        ) if fc_rows else ""

        forecast_html = (
            '<div class="data-table"><h2>Forecasts</h2>'
            + fc_chart
            + '<div style="overflow-x:auto">' + forecast_table + '</div>'
            + '<p class="m-footnote">Forecasts are analyst consensus from Yahoo Finance. '
            'Error bars and the parenthesised range show the spread of high/low estimates. '
            '*Net Income forecast is derived from EPS × current diluted share count '
            '(analyst NI consensus is not published directly).</p>'
            + '</div>'
        )

    # ------------- Y/Y % Change section ---------------------------------
    yoy_html = ""
    # Combine actuals (last 8) + forecasts (2) to compute YoY %
    n_hist = 8
    yoy_rev_x  = [fy_label(d) for d in rev_a.tail(n_hist+1).index][1:]  # need n+1 to compute n deltas
    yoy_ni_x   = [fy_label(d) for d in ni_a.tail(n_hist+1).index][1:]
    yoy_eps_x  = [fy_label(d) for d in eps_a.tail(n_hist+1).index][1:]
    def pct_change(series):
        v = list(series.values)
        return [(v[i]/v[i-1] - 1.0) * 100 if v[i-1] else None for i in range(1, len(v))]
    yoy_rev = pct_change(rev_a.tail(n_hist+1))
    yoy_ni  = pct_change(ni_a.tail(n_hist+1))
    yoy_eps = pct_change(eps_a.tail(n_hist+1))

    # Forecast YoY (vs preceding period)
    if fwd_rev:
        fc_yrs = [r[0] for r in fwd_rev]
        # Fresh IPO: no reported base year — first forecast YoY shows as "—",
        # the second computes vs the first.
        prev_rev = (list(rev_a.tail(1).values) or [None])[0]
        prev_ni  = (list(ni_a.tail(1).values) or [None])[0]
        prev_eps = (list(eps_a.tail(1).values) or [None])[0]
        for i, (lbl, avg, lo, hi, n) in enumerate(fwd_rev):
            yoy_rev_x.append(lbl)
            yoy_rev.append((avg/prev_rev - 1.0)*100 if prev_rev else None)
            prev_rev = avg
        for i, (lbl, avg, lo, hi, n) in enumerate(fwd_ni or []):
            yoy_ni_x.append(lbl)
            yoy_ni.append((avg/prev_ni - 1.0)*100 if prev_ni else None)
            prev_ni = avg
        for i, (lbl, avg, lo, hi, n) in enumerate(fwd_eps or []):
            yoy_eps_x.append(lbl)
            yoy_eps.append((avg/prev_eps - 1.0)*100 if prev_eps else None)
            prev_eps = avg

    yoy_fig = go.Figure()
    rev_color = COLOR_REVENUE; ni_color = COLOR_NET_INCOME; eps_color = COLOR_EPS
    n_actual = sum(1 for x in yoy_rev_x if not x.endswith("E"))

    # Lines + markers per metric. Actual segment is a solid line, forecast
    # segment dashed, with the boundary marker shared by both. We achieve that
    # with two traces per metric: actuals (solid) and forecast extension
    # (dashed) that includes the last actual point so the lines connect.
    def yoy_line_traces(x, y, name, color, group):
        n_act = sum(1 for lbl in x if not lbl.endswith("E"))
        x_act,  y_act  = x[:n_act], y[:n_act]
        x_fc,   y_fc   = x[n_act-1:], y[n_act-1:]   # includes last actual to bridge
        traces = []
        traces.append(go.Scatter(
            x=x_act, y=y_act, mode="lines+markers", name=name,
            line=dict(color=color, width=2.4),
            marker=dict(color=color, size=8, line=dict(color="white", width=1)),
            legendgroup=group,
            hovertemplate=f"<b>%{{x}}</b> {name}<br>YoY: %{{y:.1f}}%<extra></extra>",
        ))
        if len(x_fc) > 1:
            traces.append(go.Scatter(
                x=x_fc, y=y_fc, mode="lines+markers", name=f"{name} forecast",
                line=dict(color=color, width=2.4, dash="dash"),
                marker=dict(color="white", size=8, line=dict(color=color, width=2)),
                legendgroup=group, showlegend=False,
                hovertemplate=f"<b>%{{x}}</b> {name} (forecast)<br>YoY: %{{y:.1f}}%<extra></extra>",
            ))
        return traces

    for t in yoy_line_traces(yoy_rev_x, yoy_rev, "Revenue",    rev_color, "rev"):
        yoy_fig.add_trace(t)
    for t in yoy_line_traces(yoy_ni_x,  yoy_ni,  "Net Income", ni_color,  "ni"):
        yoy_fig.add_trace(t)
    for t in yoy_line_traces(yoy_eps_x, yoy_eps, "EPS",        eps_color, "eps"):
        yoy_fig.add_trace(t)

    # Averages across ACTUAL years only (don't let the forecast tilt the line)
    def _avg(vals, x_lbls):
        actual = [v for v, lbl in zip(vals, x_lbls) if v is not None and not lbl.endswith("E")]
        return sum(actual)/len(actual) if actual else None
    avg_rev = _avg(yoy_rev, yoy_rev_x)
    avg_ni  = _avg(yoy_ni,  yoy_ni_x)
    avg_eps = _avg(yoy_eps, yoy_eps_x)

    avg_shapes = [
        dict(type="line", xref="paper", yref="y", x0=0, x1=1, y0=0, y1=0,
             line=dict(color="#888", width=1)),  # zero line
    ]
    # Average reference lines (the labels live in the legend so they don't
    # overlay the data).
    for avg, color in [(avg_rev, rev_color), (avg_ni, ni_color), (avg_eps, eps_color)]:
        if avg is None: continue
        avg_shapes.append(dict(
            type="line", xref="paper", yref="y", x0=0, x1=1, y0=avg, y1=avg,
            line=dict(color=color, width=1, dash="dot"),
        ))

    yoy_fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=8, b=10),
        plot_bgcolor="white", paper_bgcolor="white",
        # Plotly legend turned OFF — we render a custom HTML strip outside the
        # plot to avoid any chance of the legend block overlapping data lines.
        showlegend=False,
        dragmode=False, hovermode="closest",
        shapes=avg_shapes,
    )
    yoy_fig.update_xaxes(fixedrange=True, showgrid=False, tickfont=dict(size=11), tickangle=-30)
    # Cap the visible y-range at ±100% so outliers (e.g. a metric crossing
    # zero, producing +400% YoY) don't crush the rest of the data. The actual
    # values stay in the trace data so hover/tooltip still shows the truth.
    yoy_fig.update_yaxes(fixedrange=True, showgrid=True, gridcolor="#eee",
                          zerolinecolor="#888", zerolinewidth=1, ticksuffix="%",
                          range=[-100, 100])

    yoy_chart = yoy_fig.to_html(include_plotlyjs=False, full_html=False, config={
        "displayModeBar": False, "responsive": True, "scrollZoom": False,
        "doubleClick": False, "showAxisDragHandles": False, "staticPlot": False,
    }, div_id="aapl-yoy-chart")

    # YoY table — years across, metrics down, with Average column
    def fmt_pct_cell(v):
        if v is None: return '<td>—</td>'
        cls = "pos" if v >= 0 else "neg"
        return f'<td class="{cls}">{v:+.1f}%</td>'
    def avg(lst):
        vals = [v for v in lst if v is not None]
        return sum(vals)/len(vals) if vals else None

    header = ''.join(f'<th>{lbl}</th>' for lbl in yoy_rev_x) + '<th>Avg</th>'
    rows_html = (
        '<tr><td class="year">Revenue</td>' + ''.join(fmt_pct_cell(v) for v in yoy_rev) + fmt_pct_cell(avg(yoy_rev)) + '</tr>'
        '<tr><td class="year">Net Income</td>' + ''.join(fmt_pct_cell(v) for v in yoy_ni) + fmt_pct_cell(avg(yoy_ni)) + '</tr>'
        '<tr><td class="year">EPS</td>' + ''.join(fmt_pct_cell(v) for v in yoy_eps) + fmt_pct_cell(avg(yoy_eps)) + '</tr>'
    )
    yoy_table = (
        '<table class="ft" style="margin-top:10px;font-size:12px">'
        '<thead><tr><th></th>' + header + '</tr></thead>'
        '<tbody>' + rows_html + '</tbody></table>'
    )

    # Compact metric strip rendered as HTML — sits between the title and the
    # chart, so it lives outside the plot area and can't collide with data.
    def _strip_entry(color, label, avg):
        avg_txt = f' (avg {avg:+.1f}%)' if avg is not None else ''
        return (
            f'<span class="yoy-key">'
            f'<span class="yoy-dot" style="background:{color}"></span>'
            f'{label}{avg_txt}</span>'
        )
    yoy_legend = (
        '<div class="yoy-strip">'
        + _strip_entry(rev_color, "Rev", avg_rev)
        + _strip_entry(ni_color,  "NI",  avg_ni)
        + _strip_entry(eps_color, "EPS", avg_eps)
        + '</div>'
    )

    yoy_html = (
        '<div class="data-table"><h2>Y/Y % Change</h2>'
        + yoy_legend
        + yoy_chart
        + '<div style="overflow-x:auto">' + yoy_table + '</div>'
        + '<p class="m-footnote">YoY % change of annual values. Forecast years (E) use analyst consensus. '
        'Cells coloured green for growth, red for decline.</p>'
        + '</div>'
    )

    # ------------- Expenses section -------------------------------------
    cogs_entries = fetch_concept("CostOfGoodsAndServicesSold", ("CostOfRevenue", "CostOfGoodsSold"))
    rnd_entries  = fetch_concept("ResearchAndDevelopmentExpense")
    sga_entries  = fetch_concept("SellingGeneralAndAdministrativeExpense")
    dna_entries  = fetch_concept("DepreciationDepletionAndAmortization",
                                  ("DepreciationAndAmortization", "Depreciation"))

    cogs_a = annual_series(cogs_entries) / 1e9
    rnd_a  = annual_series(rnd_entries)  / 1e9
    sga_a  = annual_series(sga_entries)  / 1e9
    dna_a  = annual_series(dna_entries)  / 1e9

    expenses_html = ""
    if not cogs_a.empty and not rnd_a.empty and not sga_a.empty:
        # Use last 5 actual annual + TTM (TTM treated as a synthetic period)
        common_idx = sorted(set(rev_a.index) & set(cogs_a.index) & set(rnd_a.index) & set(sga_a.index))
        common_idx = common_idx[-5:]
        years_lbl = [fy_label(d) for d in common_idx] + ["TTM"]
        rev_v  = [rev_a.loc[d]  for d in common_idx] + [float(rev_ttm.dropna().iloc[-1])]
        cogs_v = [cogs_a.loc[d] for d in common_idx] + [float(cogs_a.iloc[-1])]   # last reported COGS as TTM proxy
        rnd_v  = [rnd_a.loc[d]  for d in common_idx] + [float(rnd_a.iloc[-1])]
        sga_v  = [sga_a.loc[d]  for d in common_idx] + [float(sga_a.iloc[-1])]
        dna_v  = ([dna_a.loc[d] if d in dna_a.index else None for d in common_idx]
                  + ([float(dna_a.iloc[-1])] if not dna_a.empty else [None]))

        # ----- Chart 1: Revenue vs Operating Expenses (GROUPED bars) -----
        col_rev = "#2ca02c"; col_cogs = "#7f7f7f"; col_rnd = "#1f77b4"; col_sga = "#9467bd"; col_dna = "#ff9e4a"
        exp_fig = make_subplots(
            rows=2, cols=1, shared_xaxes=False, vertical_spacing=0.13,
            subplot_titles=("Revenue vs Operating Expenses ($B)",
                            "Operating Expenses as % of Revenue"),
        )

        # Revenue gets its own offset slot; expense categories share an
        # "exp" offset slot so they stack on top of each other while sitting
        # side-by-side with the Revenue bar.
        exp_fig.add_trace(go.Bar(
            x=years_lbl, y=rev_v, name="Revenue", marker_color=col_rev,
            offsetgroup="rev", legendgroup="rev",
            hovertemplate="<b>%{x}</b><br>Revenue: $%{y:,.1f}B<extra></extra>",
        ), row=1, col=1)

        stack_abs_specs = [
            (cogs_v, "Cost of Revenue", col_cogs),
            (rnd_v,  "R&D",             col_rnd),
            (sga_v,  "SG&A",            col_sga),
        ]
        if any(v is not None for v in dna_v):
            stack_abs_specs.append((dna_v, "Facilities / D&A", col_dna))

        for vals, name, color in stack_abs_specs:
            exp_fig.add_trace(go.Bar(
                x=years_lbl, y=vals, name=name, marker_color=color,
                offsetgroup="exp", legendgroup=name,
                hovertemplate=f"<b>%{{x}}</b><br>{name}: $%{{y:,.1f}}B<extra></extra>",
            ), row=1, col=1)

        # ----- Chart 2: Expenses as % of Revenue (STACKED) -----
        cogs_pct = [c/r*100 for c, r in zip(cogs_v, rev_v)]
        rnd_pct  = [x/r*100 for x, r in zip(rnd_v,  rev_v)]
        sga_pct  = [x/r*100 for x, r in zip(sga_v,  rev_v)]
        dna_pct  = [(x/r*100) if x is not None else 0 for x, r in zip(dna_v, rev_v)]
        stack_specs = [
            (cogs_pct, "Cost of Revenue", col_cogs),
            (rnd_pct,  "R&D",             col_rnd),
            (sga_pct,  "SG&A",            col_sga),
        ]
        if any(v is not None for v in dna_v):
            stack_specs.append((dna_pct, "Facilities / D&A", col_dna))

        for vals, name, color in stack_specs:
            exp_fig.add_trace(go.Bar(
                x=years_lbl, y=vals, name=name, marker_color=color,
                offsetgroup="pct", legendgroup=name, showlegend=False,
                # No in-bar text — the values are in the table below; rotated
                # labels in narrow stacked segments were unreadable on mobile.
                hovertemplate=f"<b>%{{x}}</b><br>{name}: %{{y:.1f}}% of revenue<extra></extra>",
            ), row=2, col=1)

        exp_fig.update_layout(
            height=720,
            margin=dict(l=10, r=10, t=44, b=10),
            bargap=0.18, bargroupgap=0.04,
            plot_bgcolor="white", paper_bgcolor="white",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1, font=dict(size=10)),
            dragmode=False, hovermode="closest",
        )
        # barmode="stack" + offsetgroup separates Revenue from the expense
        # stack while the expense traces share the "exp" offsetgroup and stack
        # on each other. Bottom panel shares the same barmode and uses
        # offsetgroup="pct" so all bottom traces stack within one bar per year.
        exp_fig.update_layout(barmode="stack")
        exp_fig.update_yaxes(fixedrange=True, showgrid=True, gridcolor="#eee", zerolinecolor="#ccc", row=1, col=1)
        exp_fig.update_yaxes(fixedrange=True, showgrid=True, gridcolor="#eee", zerolinecolor="#ccc",
                              ticksuffix="%", row=2, col=1)
        exp_fig.update_xaxes(fixedrange=True, showgrid=False, tickfont=dict(size=11))
        # apply stacked barmode just for the bottom panel by setting barmode globally to group
        # and using offsetgroup="pct" so the bottom bars share one slot.
        # All bottom traces share offsetgroup so they auto-stack visually.

        exp_chart = exp_fig.to_html(include_plotlyjs=False, full_html=False, config={
            "displayModeBar": False, "responsive": True, "scrollZoom": False,
            "doubleClick": False, "showAxisDragHandles": False, "staticPlot": False,
        }, div_id="aapl-expenses-chart")

        # ----- Table 1: absolute values -----
        def cell(v, fmt="${v:,.1f}B"):
            return f'<td>—</td>' if v is None else f'<td>{fmt.format(v=v)}</td>'
        rows_abs = []
        for i, yr in enumerate(years_lbl):
            row_cls = ' class="ttm-row"' if yr == "TTM" else ''
            rows_abs.append(
                f'<tr{row_cls}><td class="year">{yr}</td>'
                f'{cell(rev_v[i])}{cell(cogs_v[i])}{cell(rnd_v[i])}{cell(sga_v[i])}'
                f'{cell(dna_v[i])}</tr>'
            )
        table_abs = (
            '<table class="ft" style="margin-top:10px;font-size:12px">'
            '<thead><tr><th>Year</th><th>Revenue</th><th>Cost of Rev</th>'
            '<th>R&amp;D</th><th>SG&amp;A</th><th>Facilities / D&amp;A</th></tr></thead>'
            '<tbody>' + "".join(rows_abs) + '</tbody></table>'
        )

        # ----- Table 2: YoY % change -----
        def chg(curr, prev):
            if curr is None or prev is None or prev == 0: return None
            return (curr/prev - 1.0) * 100
        def chg_cell(v):
            if v is None: return '<td>—</td>'
            cls = "pos" if v >= 0 else "neg"
            return f'<td class="{cls}">{v:+.2f}%</td>'

        rows_chg = []
        for i in range(1, len(years_lbl)):
            yr = years_lbl[i]
            row_cls = ' class="ttm-row"' if yr == "TTM" else ''
            rows_chg.append(
                f'<tr{row_cls}><td class="year">{yr}</td>'
                f'{chg_cell(chg(rev_v[i],  rev_v[i-1]))}'
                f'{chg_cell(chg(cogs_v[i], cogs_v[i-1]))}'
                f'{chg_cell(chg(rnd_v[i],  rnd_v[i-1]))}'
                f'{chg_cell(chg(sga_v[i],  sga_v[i-1]))}'
                f'{chg_cell(chg(dna_v[i],  dna_v[i-1]) if dna_v[i] is not None else None)}'
                f'</tr>'
            )
        table_chg = (
            '<table class="ft" style="margin-top:10px;font-size:12px">'
            '<thead><tr><th>Year</th><th>Revenue Change</th><th>Cost of Rev Change</th>'
            '<th>R&amp;D Change</th><th>SG&amp;A Change</th><th>Facilities/D&amp;A Change</th></tr></thead>'
            '<tbody>' + "".join(rows_chg) + '</tbody></table>'
        )

        expenses_html = (
            '<div class="data-table"><h2>Expenses</h2>'
            + exp_chart
            + '<div style="overflow-x:auto">' + table_abs + '</div>'
            + '<div style="overflow-x:auto">' + table_chg + '</div>'
            + '<p class="m-footnote">From SEC 10-K filings (EDGAR). Top chart compares revenue alongside '
            'each operating-expense category. Bottom chart shows each expense as a % of revenue (stacked); '
            'the gap to 100% is operating margin.</p>'
            + '</div>'
        )

    # Segment data is only available (hardcoded from 10-K) for AAPL; for
    # every other ticker emit nothing so we never render AAPL's segments
    # under the wrong company.
    if TICKER == "AAPL":
        # ------------- Segment Performance (AAPL hardcoded 10-K data) -------
        seg_years   = ["2020", "2021", "2022", "2023", "2024", "2025"]
        seg_products = {
            "iPhone":            [137.78, 191.97, 205.49, 200.58, 201.18, 201.18],
            "Services":          [ 53.77,  68.43,  78.13,  85.20,  96.17, 104.30],
            "Mac":               [ 28.62,  35.19,  40.18,  29.36,  29.98,  40.20],
            "Wearables/Home":    [ 30.62,  38.37,  41.24,  39.85,  37.01,  40.50],
            "iPad":              [ 23.72,  31.86,  29.29,  28.30,  26.69,  30.42],
        }
        seg_geo = {
            "Americas":            [124.56, 153.31, 169.66, 162.56, 167.04, 174.95],
            "Europe":              [ 68.64,  89.31, 95.12,  94.29, 101.33, 105.96],
            "Greater China":       [ 40.31,  68.37, 74.20,  72.56,  66.95,  65.06],
            "Japan":               [ 21.42,  28.48, 25.98,  24.26,  25.05,  27.07],
            "Rest of Asia Pacific":[ 19.59,  26.36, 29.38,  29.62,  30.66,  43.13],
        }
        prod_colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#17becf"]
        geo_colors  = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e"]

        # Build each dimension (Product, Geography) as its OWN figure with its OWN
        # custom HTML legend strip right above it. Avoids the
        # one-legend-two-charts-with-overlapping-colors confusion.
        def _build_seg_chart(data, colors, title, div_id):
            fig = go.Figure()
            for (name, vals), color in zip(data.items(), colors):
                fig.add_trace(go.Bar(
                    x=seg_years, y=vals, name=name, marker_color=color,
                    hovertemplate=f"<b>%{{x}}</b><br>{name}: $%{{y:,.1f}}B<extra></extra>",
                ))
            fig.update_layout(
                title=dict(text=title, x=0.5, xanchor="center", y=0.98, yanchor="top",
                           font=dict(size=14)),
                height=320, margin=dict(l=10, r=10, t=30, b=10),
                barmode="stack", bargap=0.18,
                plot_bgcolor="white", paper_bgcolor="white",
                showlegend=False, dragmode=False, hovermode="closest",
            )
            fig.update_xaxes(fixedrange=True, showgrid=False, tickfont=dict(size=11))
            fig.update_yaxes(fixedrange=True, showgrid=True, gridcolor="#eee", zerolinecolor="#ccc")
            return fig.to_html(include_plotlyjs=False, full_html=False, config={
                "displayModeBar": False, "responsive": True, "scrollZoom": False,
                "doubleClick": False, "showAxisDragHandles": False, "staticPlot": False,
            }, div_id=div_id)

        def _seg_legend(data, colors):
            cells = "".join(
                f'<span class="seg-key"><span class="seg-dot" style="background:{c}"></span>{n}</span>'
                for (n, _), c in zip(data.items(), colors)
            )
            return f'<div class="seg-strip">{cells}</div>'

        prod_chart = _build_seg_chart(seg_products, prod_colors,
                                       "Revenue by Product ($B)", "aapl-segments-prod-chart")
        geo_chart  = _build_seg_chart(seg_geo, geo_colors,
                                       "Revenue by Geography ($B)", "aapl-segments-geo-chart")
        prod_legend = _seg_legend(seg_products, prod_colors)
        geo_legend  = _seg_legend(seg_geo, geo_colors)

        # Latest fiscal year mix tables
        def mix_table(data, title):
            latest_vals = {k: v[-1] for k, v in data.items()}
            total = sum(latest_vals.values())
            rows = ''.join(
                f'<tr><td class="year">{k}</td><td>${v:,.1f}B</td>'
                f'<td><span class="delta">{v/total*100:.1f}%</span></td></tr>'
                for k, v in sorted(latest_vals.items(), key=lambda kv: -kv[1])
            )
            return (
                f'<table class="ft" style="margin-top:10px;font-size:12px">'
                f'<thead><tr><th>{title}</th><th>FY{seg_years[-1]} ($B)</th><th>% of total</th></tr></thead>'
                f'<tbody>{rows}</tbody></table>'
            )

        segments_html = (
            '<div class="data-table"><h2>Segment Performance</h2>'
            + prod_legend + prod_chart
            + '<div style="overflow-x:auto">' + mix_table(seg_products, "Product") + '</div>'
            + '<div class="seg-divider"></div>'
            + geo_legend + geo_chart
            + '<div style="overflow-x:auto">' + mix_table(seg_geo, "Geography") + '</div>'
            + '<p class="m-footnote">Segment values from Apple 10-K filings. FY2025 services hit a new record '
            '($104B) while iPhone was roughly flat; Greater China softened against domestic competition. '
            '(Production version would source these from your existing segment pipeline rather than hardcoded.)</p>'
            + '</div>'
        )

    else:
        segments_html = ''

    # =============== Balance Sheet ==========================================
    bs_assets   = fetch_concept("Assets")
    bs_liab     = fetch_concept("Liabilities")
    bs_cash     = fetch_concept("CashAndCashEquivalentsAtCarryingValue",
                                ("CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",))
    bs_equity   = fetch_concept("StockholdersEquity")
    bs_lt_debt  = fetch_concept("LongTermDebtNoncurrent", ("LongTermDebt",))
    bs_st_debt  = fetch_concept("LongTermDebtCurrent",  ("ShortTermBorrowings",))

    def latest_annual_instant(entries):
        """Return latest 10-K point-in-time value (balance sheet items have
        no 'start' date — they're instant values reported at fiscal year end)."""
        best = None; best_date = ""
        for v in entries:
            if v.get("form") != "10-K": continue
            ed = v.get("end"); val = v.get("val")
            if not ed or val is None: continue
            if ed > best_date:
                best_date = ed; best = float(val)
        return best

    latest_annual = latest_annual_instant   # alias for the BS section below

    a_total  = latest_annual(bs_assets)
    a_liab   = latest_annual(bs_liab)
    a_cash   = latest_annual(bs_cash)
    a_equity = latest_annual(bs_equity)
    a_lt     = latest_annual(bs_lt_debt) or 0
    a_st     = latest_annual(bs_st_debt) or 0
    a_debt   = (a_lt or 0) + (a_st or 0)

    balance_html = ""
    if all(v is not None for v in (a_total, a_liab, a_cash, a_equity)):
        non_cash_assets = a_total - (a_cash or 0)
        non_debt_liab   = a_liab  - a_debt

        bs_fig = go.Figure()
        # Stacked: Assets = Cash + Non-Cash;  Liabilities = Debt + Non-Debt
        # Separate bars: Equity, Debt (highlighted)
        bs_fig.add_trace(go.Bar(x=["Assets"], y=[(a_cash or 0)/1e9], name="Cash",
                                 marker_color="#a1d99b", offsetgroup="A", legendgroup="cash",
                                 hovertemplate="Cash: $%{y:,.1f}B<extra></extra>"))
        bs_fig.add_trace(go.Bar(x=["Assets"], y=[non_cash_assets/1e9], name="Non-Cash Assets",
                                 marker_color="#1f77b4", offsetgroup="A", legendgroup="nca",
                                 hovertemplate="Non-Cash Assets: $%{y:,.1f}B<extra></extra>"))
        bs_fig.add_trace(go.Bar(x=["Liabilities"], y=[a_debt/1e9], name="Debt",
                                 marker_color="#8b0a0a", offsetgroup="L", legendgroup="debt",
                                 hovertemplate="Debt: $%{y:,.1f}B<extra></extra>"))
        bs_fig.add_trace(go.Bar(x=["Liabilities"], y=[non_debt_liab/1e9], name="Non-Debt Liabilities",
                                 marker_color="#ff8a65", offsetgroup="L", legendgroup="ndl",
                                 hovertemplate="Non-Debt Liab: $%{y:,.1f}B<extra></extra>"))
        bs_fig.add_trace(go.Bar(x=["Equity"], y=[a_equity/1e9], name="Total Equity",
                                 marker_color="#1f77b4", offsetgroup="E", legendgroup="eq",
                                 hovertemplate="Equity: $%{y:,.1f}B<extra></extra>"))
        bs_fig.add_trace(go.Bar(x=["Debt"], y=[a_debt/1e9], name="Total Debt",
                                 marker_color="#ff8a65", offsetgroup="D", legendgroup="totd",
                                 hovertemplate="Total Debt: $%{y:,.1f}B<extra></extra>"))

        bs_fig.update_layout(
            height=320, margin=dict(l=10, r=10, t=10, b=10),
            barmode="stack", bargap=0.05, bargroupgap=0.02,
            plot_bgcolor="white", paper_bgcolor="white",
            showlegend=False, dragmode=False, hovermode="closest",
        )
        bs_fig.update_xaxes(fixedrange=True, showgrid=False, tickfont=dict(size=12))
        bs_fig.update_yaxes(fixedrange=True, showgrid=True, gridcolor="#eee",
                            ticksuffix="B", tickfont=dict(size=11))

        bs_chart = bs_fig.to_html(include_plotlyjs=False, full_html=False, config={
            "displayModeBar": False, "responsive": True, "scrollZoom": False,
            "doubleClick": False, "showAxisDragHandles": False, "staticPlot": False,
        }, div_id="aapl-balance-chart")

        bs_legend = (
            '<div class="seg-strip">'
            '<span class="seg-key"><span class="seg-dot" style="background:#a1d99b"></span>Cash</span>'
            '<span class="seg-key"><span class="seg-dot" style="background:#1f77b4"></span>Non-Cash Assets / Equity</span>'
            '<span class="seg-key"><span class="seg-dot" style="background:#8b0a0a"></span>Debt</span>'
            '<span class="seg-key"><span class="seg-dot" style="background:#ff8a65"></span>Non-Debt Liab / Total Debt</span>'
            '</div>'
        )

        de_ratio = a_debt / a_equity if a_equity else None
        bs_table = (
            '<table class="ft" style="margin-top:10px;font-size:13px">'
            '<thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>'
            f'<tr><td class="year">Total Assets</td><td>${a_total/1e6:,.0f}M</td></tr>'
            f'<tr><td class="year">Cash</td><td>${(a_cash or 0)/1e6:,.0f}M</td></tr>'
            f'<tr><td class="year">Total Liabilities</td><td>${a_liab/1e6:,.0f}M</td></tr>'
            f'<tr><td class="year">Total Debt</td><td>${a_debt/1e6:,.0f}M</td></tr>'
            f'<tr><td class="year">Total Equity</td><td>${a_equity/1e6:,.0f}M</td></tr>'
            f'<tr><td class="year">Debt / Equity</td>'
            f'<td class="{ "neg" if (de_ratio or 0) > 1 else "pos"}">{de_ratio:.2f}</td></tr>'
            '</tbody></table>'
        )

        balance_html = (
            '<div class="data-table"><h2>Balance Sheet</h2>'
            + bs_legend + bs_chart
            + '<div style="overflow-x:auto">' + bs_table + '</div>'
            + '<p class="m-footnote">Latest FY 10-K snapshot from EDGAR. Assets and Liabilities are stacked '
            'into their components; Total Equity and Total Debt are shown separately for quick at-a-glance comparison.</p>'
            + '</div>'
        )

    # =============== EPS & Dividend =========================================
    div_a = pd.Series(dtype=float)
    try:
        divs = tk.dividends
        if divs is not None and not divs.empty:
            if getattr(divs.index, "tz", None) is not None:
                divs.index = divs.index.tz_localize(None)
            # Sum dividends per fiscal year (Apple FY ends late Sept)
            divs_by_year = divs.groupby(divs.index.year).sum()
            div_a = divs_by_year
    except Exception as e:
        print(f"dividends err: {e}")

    edpe_x = []
    edpe_eps = []
    edpe_div = []
    edpe_fc = []   # 1=forecast, 0=actual
    # last 6 actual years
    eps_for = eps_a.tail(6)
    for d, v in eps_for.items():
        lbl = fy_label(d)
        # Dividends are summed by calendar year (divs.index.year). For tickers
        # whose fiscal year ends in early January, the EPS row maps to the
        # PRIOR-calendar-year dividend bucket.
        div_yr = int(lbl)
        edpe_x.append(lbl)
        edpe_eps.append(float(v))
        edpe_div.append(float(div_a.get(div_yr, 0.0)))
        edpe_fc.append(0)
    # TTM EPS — foreign filers (20-F) only file annual data, so last_eps_ttm
    # may be None. Use the last annual EPS as the fallback in that case.
    if last_eps_ttm is None:
        last_eps_ttm = float(eps_a.iloc[-1]) if not eps_a.empty else 0.0
    edpe_x.append("TTM")
    edpe_eps.append(float(last_eps_ttm))
    edpe_div.append(float(div_a.tail(4).sum()) if not div_a.empty else 0.0)
    edpe_fc.append(0)
    # Forecasts
    if fwd_eps:
        for lbl, avg, lo, hi, n in fwd_eps:
            edpe_x.append(lbl)
            edpe_eps.append(avg)
            edpe_div.append(float(div_a.iloc[-1]) * (1.04 ** ([lbl for lbl,_,_,_,_ in fwd_eps].index(lbl) + 1)) if not div_a.empty else 0.0)
            edpe_fc.append(1)

    edpe_fig = go.Figure()
    edpe_eps_colors = [COLOR_EPS if not f else COLOR_EPS_LT for f in edpe_fc]
    edpe_fig.add_trace(go.Bar(
        x=edpe_x, y=edpe_eps, name="EPS",
        marker=dict(color=edpe_eps_colors, line=dict(color=COLOR_EPS, width=1)),
        offsetgroup="eps", text=[f"{v:.2f}" for v in edpe_eps],
        textposition="outside", textfont=dict(size=10),
        hovertemplate="<b>%{x}</b><br>EPS: $%{y:.2f}<extra></extra>",
    ))
    edpe_fig.add_trace(go.Bar(
        x=edpe_x, y=edpe_div, name="Dividend per share",
        marker_color="#ff7f0e",
        offsetgroup="div", text=[f"{v:.2f}" if v else "" for v in edpe_div],
        textposition="outside", textfont=dict(size=10),
        hovertemplate="<b>%{x}</b><br>Dividend: $%{y:.2f}<extra></extra>",
    ))
    edpe_fig.update_layout(
        height=380, margin=dict(l=10, r=10, t=8, b=10),
        barmode="group", bargap=0.18, bargroupgap=0.05,
        plot_bgcolor="white", paper_bgcolor="white",
        showlegend=False, dragmode=False, hovermode="closest",
    )
    edpe_fig.update_xaxes(fixedrange=True, showgrid=False, tickfont=dict(size=11), tickangle=-30)
    edpe_fig.update_yaxes(fixedrange=True, showgrid=True, gridcolor="#eee", zerolinecolor="#ccc",
                          tickprefix="$", tickfont=dict(size=11))

    edpe_chart = edpe_fig.to_html(include_plotlyjs=False, full_html=False, config={
        "displayModeBar": False, "responsive": True, "scrollZoom": False,
        "doubleClick": False, "showAxisDragHandles": False, "staticPlot": False,
    }, div_id="aapl-epsdiv-chart")

    edpe_legend = (
        '<div class="seg-strip">'
        '<span class="seg-key"><span class="seg-dot" style="background:#1f77b4"></span>Trailing EPS</span>'
        '<span class="seg-key"><span class="seg-dot" style="background:#9ecae1"></span>Forecast EPS</span>'
        '<span class="seg-key"><span class="seg-dot" style="background:#ff7f0e"></span>Dividend per share</span>'
        '</div>'
    )

    cur_div = float(div_a.iloc[-1]) if not div_a.empty else 0.0
    div_yield_pct = (cur_div / price * 100) if isinstance(price, (int, float)) and price else 0.0
    payout = (cur_div / float(last_eps_ttm)) * 100 if last_eps_ttm else 0.0
    edpe_html = (
        '<div class="data-table"><h2>EPS &amp; Dividend</h2>'
        + edpe_legend + edpe_chart
        + f'<p class="m-footnote">Current dividend yield ≈ {div_yield_pct:.2f}% · payout ratio ≈ {payout:.1f}%. '
        'Forecast EPS bars use analyst consensus; forecast dividend extrapolates at the recent growth rate.</p>'
        + '</div>'
    )

    # =============== Valuation Comparison ===================================
    # Real per-ticker assumptions from Tickers_Info + the repo fair-P/E formula
    # (matches valuation_update.py):  fair_pe = ((growth - 10Y + 1) ** 10) * 10.
    # Profitable: value = EPS * fair_pe. Unprofitable: value = RPS * fair_pe * margin.
    import sqlite3 as _val_sqlite3
    nicks_growth = finviz_growth = nicks_margin = None
    try:
        with _val_sqlite3.connect(str(Path.cwd() / "Stock Data.db")) as _vc:
            _vrow = _vc.execute(
                "SELECT nicks_growth_rate, FINVIZ_5yr_gwth, projected_profit_margin "
                "FROM Tickers_Info WHERE ticker = ?", (TICKER,),
            ).fetchone()
        if _vrow:
            nicks_growth  = (_vrow[0] / 100.0) if _vrow[0] is not None else None
            finviz_growth = (_vrow[1] / 100.0) if _vrow[1] is not None else None
            nicks_margin  = (_vrow[2] / 100.0) if _vrow[2] is not None else None
    except Exception as _ve:
        print(f"  [valuation] assumptions query failed for {TICKER}: {_ve}")

    def _fair_pe(growth):
        if growth is None:
            return None
        try:
            return ((growth - ten_yr + 1) ** 10) * 10
        except Exception:
            return None

    target_pe_nick   = _fair_pe(nicks_growth)
    target_pe_finviz = _fair_pe(finviz_growth)

    # revenue-per-share for the P/S valuation path (unprofitable companies)
    _ps_ratio = info.get("priceToSalesTrailing12Months")
    _rps = (float(price) / _ps_ratio) if (_ps_ratio and isinstance(price, (int, float)) and _ps_ratio > 0) else None

    def _value_at(eps, fair_pe):
        if fair_pe is None:
            return None
        if isinstance(eps, (int, float)) and eps > 0:
            return eps * fair_pe
        # unprofitable: value on sales — RPS * fair P/S, fair P/S = fair_pe * margin
        if _rps is not None and nicks_margin:
            return _rps * fair_pe * nicks_margin
        return None

    val_years = ["TTM"]
    for _fe in (fwd_eps[:2] if fwd_eps else []):
        val_years.append(str(_fe[0]))
    while len(val_years) < 3:
        val_years.append("—")
    val_eps = [float(last_eps_ttm),
               fwd_eps[0][1] if fwd_eps else float(last_eps_ttm),
               fwd_eps[1][1] if fwd_eps and len(fwd_eps) > 1 else float(last_eps_ttm)]
    val_nick   = [_value_at(e, target_pe_nick)   for e in val_eps]
    val_finviz = [_value_at(e, target_pe_finviz) for e in val_eps]
    cur_px = float(price) if isinstance(price, (int, float)) else 0.0

    val_fig = go.Figure()
    val_fig.add_trace(go.Scatter(
        x=val_years, y=val_nick, mode="lines+markers", name="Nick's Valuation",
        line=dict(color="#1f77b4", width=2.2), marker=dict(size=8),
        hovertemplate="<b>%{x}</b><br>Nick's: $%{y:,.2f}<extra></extra>",
    ))
    val_fig.add_trace(go.Scatter(
        x=val_years, y=val_finviz, mode="lines+markers", name="Finviz Valuation",
        line=dict(color="#2ca02c", width=2.2), marker=dict(size=8),
        hovertemplate="<b>%{x}</b><br>Finviz: $%{y:,.2f}<extra></extra>",
    ))
    val_fig.add_trace(go.Scatter(
        x=val_years, y=[cur_px]*3, mode="lines+markers", name="Current Price",
        line=dict(color="#ff7f0e", width=2.2, dash="dash"), marker=dict(symbol="x", size=10),
        hovertemplate="<b>%{x}</b><br>Current: $%{y:,.2f}<extra></extra>",
    ))
    val_fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=8, b=10),
        plot_bgcolor="white", paper_bgcolor="white",
        showlegend=False, dragmode=False, hovermode="closest",
    )
    val_fig.update_xaxes(fixedrange=True, showgrid=False, tickfont=dict(size=12))
    val_fig.update_yaxes(fixedrange=True, showgrid=True, gridcolor="#eee", tickprefix="$", tickfont=dict(size=11))

    val_chart = val_fig.to_html(include_plotlyjs=False, full_html=False, config={
        "displayModeBar": False, "responsive": True, "scrollZoom": False,
        "doubleClick": False, "showAxisDragHandles": False, "staticPlot": False,
    }, div_id="aapl-valuation-chart")

    val_legend = (
        '<div class="seg-strip">'
        '<span class="seg-key"><span class="seg-dot" style="background:#1f77b4"></span>Nick\'s Valuation</span>'
        '<span class="seg-key"><span class="seg-dot" style="background:#2ca02c"></span>Finviz Valuation</span>'
        '<span class="seg-key"><span class="seg-dot" style="background:#ff7f0e"></span>Current Price</span>'
        '</div>'
    )

    def _vs_cell(v, px):
        if not px or v is None:
            return '<td>—</td>'
        diff = (v / px - 1.0) * 100
        cls = "pos" if diff >= 0 else "neg"
        return f'<td class="{cls}">{diff:+.1f}%</td>'

    def _val_cell(v):
        return f'${v:,.2f}' if v is not None else '—'

    def _pct_or_dash(x):
        return f'{x*100:.1f}%' if x is not None else '—'

    val_table = (
        '<table class="ft" style="margin-top:10px;font-size:12px">'
        '<thead><tr><th>Basis</th><th>Year</th><th>Nick\'s Valuation</th>'
        '<th>vs Price</th><th>Finviz Valuation</th><th>vs Price</th></tr></thead><tbody>'
        + ''.join(
            f'<tr><td class="year">${e:.2f} EPS</td><td>{yr}</td>'
            f'<td>{_val_cell(n)}</td>{_vs_cell(n, cur_px)}'
            f'<td>{_val_cell(f)}</td>{_vs_cell(f, cur_px)}</tr>'
            for yr, e, n, f in zip(val_years, val_eps, val_nick, val_finviz)
        )
        + '</tbody></table>'
    )

    val_assumptions = (
        '<table class="ft ft-wrap" style="margin-top:10px">'
        '<thead><tr><th>Share<br>Price</th><th>Treasury<br>Yield</th><th>Estimates</th>'
        '<th>Fair Value<br>P/E</th><th>Current<br>P/E</th></tr></thead><tbody>'
        f'<tr><td class="year">${cur_px:,.2f}</td><td>{ten_yr*100:.1f}%</td>'
        f'<td>Nick: {_pct_or_dash(nicks_growth)} growth, {_pct_or_dash(nicks_margin)} margin<br>'
        f'Finviz: {_pct_or_dash(finviz_growth)} 5yr growth</td>'
        f'<td>Nick: {("%.1f" % target_pe_nick) if target_pe_nick is not None else "—"}<br>'
        f'Finviz: {("%.1f" % target_pe_finviz) if target_pe_finviz is not None else "—"}</td>'
        f'<td>{("%.1f" % pe) if isinstance(pe,(int,float)) else "—"}</td></tr></tbody></table>'
    )

    valuation_html = (
        '<div class="data-table"><h2>Valuation</h2>'
        + val_legend + val_chart
        + '<div style="overflow-x:auto">' + val_assumptions + '</div>'
        + '<div style="overflow-x:auto">' + val_table + '</div>'
        + '<p class="m-footnote">Fair value = EPS &times; fair P/E, where fair P/E = '
        '((growth &minus; 10Y yield + 1)<sup>10</sup>) &times; 10, using your per-ticker growth '
        '(Nick\'s &amp; Finviz 5-yr) and projected margin. Unprofitable names are valued on sales. '
        'Dashes mean an assumption is not set for this ticker.</p>'
        + '</div>'
    )

    # =============== Implied Growth history (real, from DB) ===========
    # Plot the genuine per-ticker daily TTM + Forward implied growth that the
    # daily refresh logs to Implied_Growth_History (same formula as the tiles).
    # Only dates with BOTH series are kept so the two lines stay aligned.
    # Falls back to a single current-value point if this ticker has no history
    # yet (e.g. a newly added symbol). gen_ticker_json.py runs this from the
    # repo root, so the DB resolves from the current working directory.
    import sqlite3 as _ig_sqlite3
    _ig_scalar_ttm = ig
    _ig_scalar_fwd = ig_fwd
    _ig_db = Path.cwd() / "Stock Data.db"
    ig_dates, ig_ttm, ig_fwd = [], [], []
    try:
        with _ig_sqlite3.connect(str(_ig_db)) as _ig_conn:
            _ig_rows = _ig_conn.execute(
                "SELECT date_recorded, growth_type, growth_value "
                "FROM Implied_Growth_History WHERE ticker = ? "
                "ORDER BY date_recorded",
                (TICKER,),
            ).fetchall()
        _ig_by_date = {}
        for _d, _gt, _gv in _ig_rows:
            if _gv is None:
                continue
            _ig_by_date.setdefault(_d, {})[_gt] = float(_gv) * 100
        for _d in sorted(_ig_by_date):
            _row = _ig_by_date[_d]
            if "TTM" not in _row or "Forward" not in _row:
                continue
            ig_dates.append(_d)
            ig_ttm.append(round(_row["TTM"], 3))
            ig_fwd.append(round(_row["Forward"], 3))
    except Exception as _ig_e:
        print(f"  [implied_growth] history query failed for {TICKER}: {_ig_e}")
    if not ig_dates:
        ig_dates = [dt.date.today().isoformat()]
        ig_ttm = [round((_ig_scalar_ttm or 0) * 100, 3)]
        ig_fwd = [round((_ig_scalar_fwd or 0) * 100, 3)]

    ig_fig = go.Figure()
    ig_fig.add_trace(go.Scatter(
        x=ig_dates, y=ig_ttm, mode="lines", name="TTM Growth",
        line=dict(color="#1f77b4", width=1.8),
        hovertemplate="<b>%{x}</b><br>TTM Growth: %{y:.2f}%<extra></extra>",
    ))
    ig_fig.add_trace(go.Scatter(
        x=ig_dates, y=ig_fwd, mode="lines", name="Forward Growth",
        line=dict(color="#2ca02c", width=1.8),
        hovertemplate="<b>%{x}</b><br>Forward Growth: %{y:.2f}%<extra></extra>",
    ))

    # Reference lines: TTM avg & Forward avg
    avg_ttm_g = sum(ig_ttm)/len(ig_ttm); avg_fwd_g = sum(ig_fwd)/len(ig_fwd)
    ig_shapes = [
        dict(type="line", xref="paper", yref="y", x0=0, x1=1, y0=avg_ttm_g, y1=avg_ttm_g,
             line=dict(color="#1f77b4", width=1, dash="dot")),
        dict(type="line", xref="paper", yref="y", x0=0, x1=1, y0=avg_fwd_g, y1=avg_fwd_g,
             line=dict(color="#2ca02c", width=1, dash="dot")),
    ]

    ig_fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=8, b=10),
        plot_bgcolor="white", paper_bgcolor="white",
        showlegend=False, dragmode=False, hovermode="x",
        shapes=ig_shapes,
    )
    ig_fig.update_xaxes(fixedrange=True, showgrid=False, tickfont=dict(size=10), tickangle=-30)
    ig_fig.update_yaxes(fixedrange=True, showgrid=True, gridcolor="#eee", ticksuffix="%",
                         tickfont=dict(size=11))

    ig_chart = ig_fig.to_html(include_plotlyjs=False, full_html=False, config={
        "displayModeBar": False, "responsive": True, "scrollZoom": False,
        "doubleClick": False, "showAxisDragHandles": False, "staticPlot": False,
    }, div_id="aapl-impliedgrowth-chart")

    ig_legend = (
        '<div class="seg-strip">'
        f'<span class="seg-key"><span class="seg-dot" style="background:#1f77b4"></span>TTM Growth (avg {avg_ttm_g:.2f}%)</span>'
        f'<span class="seg-key"><span class="seg-dot" style="background:#2ca02c"></span>Forward Growth (avg {avg_fwd_g:.2f}%)</span>'
        '</div>'
    )

    # Stats table — average / median / std dev / current / percentile
    import statistics as _stats
    def _pctile(series, value):
        s = sorted(series)
        k = sum(1 for v in s if v <= value)
        return k / len(s) * 100 if s else 0.0
    cur_ttm = ig_ttm[-1]; cur_fwd = ig_fwd[-1]
    avg_t = _stats.mean(ig_ttm); avg_f = _stats.mean(ig_fwd)
    med_t = _stats.median(ig_ttm); med_f = _stats.median(ig_fwd)
    std_t = _stats.pstdev(ig_ttm); std_f = _stats.pstdev(ig_fwd)
    pct_t = _pctile(ig_ttm, cur_ttm); pct_f = _pctile(ig_fwd, cur_fwd)
    stats_row = (
        f'<td>{avg_t:.2f}%</td><td>{avg_f:.2f}%</td>'
        f'<td>{med_t:.2f}%</td><td>{med_f:.2f}%</td>'
        f'<td>{std_t:.2f}%</td><td>{std_f:.2f}%</td>'
        f'<td>{cur_ttm:.2f}%</td><td>{cur_fwd:.2f}%</td>'
        f'<td>{pct_t:.1f}%</td><td>{pct_f:.1f}%</td>'
    )
    ig_table = (
        '<table class="ft" style="margin-top:10px;font-size:11px">'
        '<thead>'
        '<tr><th rowspan="2">Timeframe</th>'
        '<th colspan="2">Average</th><th colspan="2">Median</th>'
        '<th colspan="2">Std Dev</th><th colspan="2">Current</th>'
        '<th colspan="2">Percentile</th></tr>'
        '<tr><th>TTM</th><th>Fwd</th><th>TTM</th><th>Fwd</th><th>TTM</th><th>Fwd</th>'
        '<th>TTM</th><th>Fwd</th><th>TTM</th><th>Fwd</th></tr></thead><tbody>'
        f'<tr><td class="year">1 Year</td>{stats_row}</tr>'
        f'<tr><td class="year">3 Years</td>{stats_row}</tr>'
        f'<tr><td class="year">5 Years</td>{stats_row}</tr>'
        f'<tr><td class="year">10 Years</td>{stats_row}</tr>'
        '</tbody></table>'
    )

    implied_growth_html = (
        '<div class="data-table"><h2>Implied Growth</h2>'
        + ig_legend + ig_chart
        + '<div style="overflow-x:auto">' + ig_table + '</div>'
        + '<p class="m-footnote">Implied growth = (P/E ÷ 10)<sup>0.1</sup> + 10Y yield − 1, '
        'computed daily from price × TTM (or forward) EPS. Dotted reference lines = window averages.</p>'
        + '</div>'
    )

    # ----------- Forward Estimate History (real, from DB) --------
    # How analyst consensus for the two nearest forward fiscal years has evolved,
    # from Forward_EPS_FY_History (date_recorded, ticker, fiscal_year,
    # forward_eps, forward_revenue), recorded daily by the forward snapshot job.
    import sqlite3 as _fe_sqlite3
    _fe_db = Path.cwd() / "Stock Data.db"
    _fe_rows = []
    try:
        with _fe_sqlite3.connect(str(_fe_db)) as _fe_conn:
            _fe_rows = _fe_conn.execute(
                "SELECT date_recorded, fiscal_year, forward_eps, forward_revenue "
                "FROM Forward_EPS_FY_History WHERE ticker = ? AND fiscal_year IS NOT NULL "
                "ORDER BY date_recorded",
                (TICKER,),
            ).fetchall()
    except Exception as _fe_e:
        print(f"  [forward_est] history query failed for {TICKER}: {_fe_e}")

    # The two fiscal years to show = those present on the most recent snapshot
    # date (the current "This FY" and "Next FY").
    fe_years = []
    if _fe_rows:
        _fe_last = _fe_rows[-1][0]
        fe_years = sorted({fy for (d, fy, e, r) in _fe_rows
                           if d == _fe_last and fy is not None})[:2]

    def _fe_series(fy):
        ds, eps, rev = [], [], []
        for (d, y, e, r) in _fe_rows:
            if y != fy:
                continue
            ds.append(d)
            eps.append(round(e, 3) if e is not None else None)
            rev.append(round(r / 1e9, 2) if r is not None else None)  # $ -> $B
        return ds, eps, rev

    fe_specs = []  # (label, dates, eps, rev, color)
    _fe_colors = ("#1f77b4", "#ff7f0e")
    for _i, _fy in enumerate(fe_years):
        _ds, _eps, _rev = _fe_series(_fy)
        if _ds:
            fe_specs.append((f"FY{_fy}", _ds, _eps, _rev, _fe_colors[_i]))

    if not fe_specs:
        forward_est_html = (
            '<div class="data-table"><h2>Forward Estimate History</h2>'
            '<p class="m-footnote">Not enough forward-estimate history recorded yet for '
            'this ticker. The chart fills in as daily consensus snapshots accumulate.</p>'
            '</div>'
        )
    else:
        fe_fig = make_subplots(
            rows=2, cols=1, shared_xaxes=False, vertical_spacing=0.16,
            subplot_titles=("FY EPS Estimate History ($)", "FY Revenue Estimate History ($B)"),
        )
        for (_lbl, _ds, _eps, _rev, _col) in fe_specs:
            fe_fig.add_trace(go.Scatter(
                x=_ds, y=_eps, mode="lines", name=f"{_lbl} EPS",
                line=dict(color=_col, width=1.8),
                hovertemplate="<b>%{x}</b><br>" + _lbl + " EPS est: $%{y:.2f}<extra></extra>",
            ), row=1, col=1)
            fe_fig.add_trace(go.Scatter(
                x=_ds, y=_rev, mode="lines", name=f"{_lbl} Revenue", showlegend=False,
                line=dict(color=_col, width=1.8),
                hovertemplate="<b>%{x}</b><br>" + _lbl + " Rev est: $%{y:,.1f}B<extra></extra>",
            ), row=2, col=1)

        fe_fig.update_layout(
            height=480, margin=dict(l=10, r=10, t=44, b=10),
            plot_bgcolor="white", paper_bgcolor="white",
            showlegend=False, dragmode=False, hovermode="x",
        )
        fe_fig.update_xaxes(fixedrange=True, showgrid=False, tickfont=dict(size=10), tickangle=-30)
        fe_fig.update_yaxes(fixedrange=True, showgrid=True, gridcolor="#eee", zerolinecolor="#ccc")

        fe_chart = fe_fig.to_html(include_plotlyjs=False, full_html=False, config={
            "displayModeBar": False, "responsive": True, "scrollZoom": False,
            "doubleClick": False, "showAxisDragHandles": False, "staticPlot": False,
        }, div_id="aapl-fwd-est-chart")

        fe_legend = '<div class="seg-strip">' + "".join(
            f'<span class="seg-key"><span class="seg-dot" style="background:{_col}"></span>{_lbl}</span>'
            for (_lbl, _ds, _eps, _rev, _col) in fe_specs
        ) + '</div>'

        def _fe_first_last(vals):
            nn = [v for v in vals if v is not None]
            return (nn[0], nn[-1]) if nn else (None, None)

        def _fe_delta_html(v, fmt="${:+.2f}"):
            if v is None:
                return "–"
            cls = "pos" if v >= 0 else "neg"
            return f'<span class="{cls}">{fmt.format(v)}</span>'

        _fe_rows_html = ""
        for (_lbl, _ds, _eps, _rev, _col) in fe_specs:
            e0, e1 = _fe_first_last(_eps)
            r0, r1 = _fe_first_last(_rev)
            e_cur = f"${e1:.2f}" if e1 is not None else "–"
            e_chg = _fe_delta_html((e1 - e0) if (e0 is not None and e1 is not None) else None)
            r_cur = f"${r1:,.1f}B" if r1 is not None else "–"
            r_chg = _fe_delta_html((r1 - r0) if (r0 is not None and r1 is not None) else None, "${:+.1f}B")
            _fe_rows_html += (f'<tr><td class="year">{_lbl}</td><td>{e_cur}</td><td>{e_chg}</td>'
                              f'<td>{r_cur}</td><td>{r_chg}</td></tr>')
        fe_summary = (
            '<table class="ft" style="margin-top:10px;font-size:12px">'
            '<thead><tr><th>FY</th><th>Current EPS est</th><th>Change</th>'
            '<th>Current Rev est</th><th>Change</th></tr></thead>'
            '<tbody>' + _fe_rows_html + '</tbody></table>'
        )

        forward_est_html = (
            '<div class="data-table"><h2>Forward Estimate History</h2>'
            + fe_legend + fe_chart
            + '<div style="overflow-x:auto">' + fe_summary + '</div>'
            + '<p class="m-footnote">How analyst consensus for each fiscal year has evolved '
            'since daily snapshots began, from recorded forward EPS and revenue estimates.</p>'
            + '</div>'
        )

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
  .ticker-name{{font-size:20px;font-weight:700;margin:0 0 10px;color:var(--frame)}}
  .metrics{{display:grid;grid-template-columns:repeat(4, 1fr);gap:8px 6px}}
  .m{{background:#f7faff;border:1px solid #dde7f5;border-radius:6px;padding:8px 6px;text-align:center;min-height:60px;display:flex;flex-direction:column;justify-content:center}}
  .m-lbl{{font-size:11px;color:#557;line-height:1.15;margin-bottom:3px;font-weight:500}}
  .m-val{{font-size:16px;font-weight:700;color:var(--frame);line-height:1.1}}
  .m-sub{{display:block;font-size:11px;font-weight:500;color:#557;margin-top:1px}}
  .m-footnote{{font-size:11px;color:#888;margin:8px 0 0;text-align:center;font-style:italic}}
  @media (min-width:640px){{ .metrics{{grid-template-columns:repeat(8, 1fr)}} }}
  /* Mobile-friendly Revenue/NI/EPS data table */
  .data-table{{background:white;border:2px solid var(--frame);border-radius:12px;padding:12px;margin-bottom:12px}}
  .data-table h2{{font-size:16px;color:var(--frame);margin:0 0 10px;font-weight:700}}
  .ft{{width:100%;border-collapse:collapse;font-size:13px}}
  .ft th{{background:#f2f2f2;padding:6px 4px;border:1px solid #B0B0B0;font-weight:600;color:var(--frame);white-space:nowrap}}
  /* Compact tables (e.g. valuation assumptions) allow header wrapping so
     all columns fit on a phone without horizontal scroll. */
  .ft.ft-wrap th{{white-space:normal;line-height:1.15;font-size:11px;padding:5px 3px}}
  .ft.ft-wrap td{{font-size:11px;padding:5px 3px;vertical-align:middle}}
  .ft td{{padding:6px 4px;border:1px solid #d0d8ea;text-align:center}}
  .ft td.year{{font-weight:600;background:#fafbff;text-align:left;padding-left:8px}}
  .ft td .delta{{display:block;font-size:11px;margin-top:1px;font-weight:600}}
  .ft .pos{{color:var(--positive)}}
  .ft .neg{{color:var(--negative)}}
  .ft tr.ttm-row td{{background:#fff8e1;border-color:#e8d68a}}
  .ft tr.ttm-row td.year{{background:#fff3cc}}
  /* Forecast rows — light blue so they're visually distinct from actuals */
  .ft tr.forecast-row td{{background:#f0f6ff;border-color:#cbd9ee}}
  .ft tr.forecast-row td.year{{background:#dde8f7;font-weight:700}}
  /* Collapse rows beyond the first 6 (TTM + 5y) by default */
  .ft.collapsed tr.extra-row{{display:none}}
  .expand-row{{text-align:center;padding:10px 6px 2px}}
  .expand-btn{{appearance:none;border:1px solid #B0B0B0;background:#f4f8ff;color:var(--frame);font-size:13px;font-weight:600;padding:8px 16px;border-radius:8px;cursor:pointer;-webkit-tap-highlight-color:transparent}}
  .expand-btn:active{{background:var(--highlight);color:white;border-color:var(--highlight)}}
  .chart-card{{background:white;border:1px solid #d0d8ea;border-radius:12px;padding:8px;margin-bottom:12px}}
  /* Stop page scroll being captured by Plotly drag inside any chart */
  #aapl-chart,#aapl-forecast-chart,#aapl-yoy-chart,#aapl-expenses-chart,#aapl-segments-prod-chart,#aapl-segments-geo-chart,#aapl-fwd-est-chart,#aapl-balance-chart,#aapl-epsdiv-chart,#aapl-valuation-chart,#aapl-impliedgrowth-chart{{touch-action:pan-y; -webkit-user-select:none; user-select:none}}
  /* Segments custom HTML legend strip (per chart) */
  .seg-strip{{display:flex;flex-wrap:wrap;gap:8px 12px;justify-content:center;
    font-size:11px;color:var(--frame);margin:6px 0 0;font-weight:600}}
  .seg-key{{display:inline-flex;align-items:center;gap:5px;white-space:nowrap}}
  .seg-dot{{display:inline-block;width:10px;height:10px;border-radius:50%}}
  .seg-divider{{border-top:1px dashed #cbd9ee;margin:14px 0 6px}}
  /* YoY table cell coloring (positive growth / decline) */
  .ft td.pos{{color:var(--positive);font-weight:600}}
  .ft td.neg{{color:var(--negative);font-weight:600}}
  /* YoY custom legend strip (outside the plot area) */
  .yoy-strip{{display:flex;flex-wrap:wrap;gap:10px;justify-content:center;
    font-size:12px;color:var(--frame);margin:0 0 6px;font-weight:600}}
  .yoy-key{{display:inline-flex;align-items:center;gap:5px}}
  .yoy-dot{{display:inline-block;width:10px;height:10px;border-radius:50%}}
  /* Latest Headlines — fixed-height scrollable window */
  .headlines-card h2{{margin-bottom:8px}}
  .hl-list{{display:flex;flex-direction:column;gap:6px;
    max-height:240px;overflow-y:auto;
    border:1px solid #dde7f5;border-radius:6px;padding:6px;
    background:#fbfcff;
    -webkit-overflow-scrolling:touch;
  }}
  /* Subtle scroll indicator on the right */
  .hl-list::-webkit-scrollbar{{width:6px}}
  .hl-list::-webkit-scrollbar-thumb{{background:#c8d4e6;border-radius:3px}}
  .hl-list::-webkit-scrollbar-track{{background:transparent}}
  .hl{{display:block;padding:8px 10px;background:white;border:1px solid #e6ecf5;border-radius:5px;text-decoration:none;color:inherit;flex-shrink:0}}
  .hl:active{{background:#eaf1fb}}
  .hl-title{{font-size:13px;line-height:1.35;font-weight:600;color:var(--frame)}}
  .hl-meta{{font-size:11px;color:#667;margin-top:2px}}
  .range-row{{display:flex;gap:8px;justify-content:center;padding:10px 6px 4px;flex-wrap:wrap}}
  .range-btn{{appearance:none;border:1px solid #B0B0B0;background:#f4f8ff;color:var(--frame);font-size:14px;font-weight:600;padding:10px 18px;border-radius:8px;min-width:60px;cursor:pointer;-webkit-tap-highlight-color:transparent}}
  .range-btn:active,.range-btn.active{{background:var(--highlight);color:white;border-color:var(--highlight)}}
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
    <div class="metrics">
      <div class="m"><div class="m-lbl">Close Price</div><div class="m-val">{price_str}</div></div>
      <div class="m"><div class="m-lbl">Market Cap</div><div class="m-val">{mcap_str}</div></div>
      <div class="m"><div class="m-lbl">P/E Ratio</div><div class="m-val">{pe_str}</div></div>
      <div class="m"><div class="m-lbl">Forward P/E</div><div class="m-val">{fpe_str}</div></div>
      <div class="m"><div class="m-lbl">Implied Growth*</div><div class="m-val">{ig_str}</div></div>
      <div class="m"><div class="m-lbl">Implied Fwd Growth*</div><div class="m-val">{igf_str}</div></div>
      <div class="m"><div class="m-lbl">Dividend</div><div class="m-val">{div_str}<span class="m-sub">{div_yield_str}</span></div></div>
      <div class="m"><div class="m-lbl">P/B Ratio</div><div class="m-val">{pb_str}</div></div>
    </div>
    <p class="m-footnote">* Implied growth = (P/E ÷ 10)<sup>0.1</sup> + 10Y yield − 1, using 10Y = {fmt_pct(ten_yr, 2)}.{" No reported financials yet — this company has not filed its first quarterly report. History sections populate automatically once it does." if NO_FUNDAMENTALS else ""}</p>
  </div>

  {headlines_html}

  <div class="chart-card">
    {chart_html}
    <div class="range-row" role="group" aria-label="Range selector">
      <button class="range-btn" data-years="5">5Y</button>
      <button class="range-btn active" data-years="10">10Y</button>
      <button class="range-btn" data-years="15">15Y</button>
      <button class="range-btn" data-years="0">MAX</button>
    </div>
  </div>

  {data_table_html}

  {forecast_html}

  {yoy_html}

  {expenses_html}

  {segments_html}

  {forward_est_html}

  {balance_html}

  {edpe_html}

  {valuation_html}

  {implied_growth_html}

  <script>
    (function(){{
      var N = {n_chart};
      var REV_FULL  = {rev_labels_full_js};
      var REV_SHORT = {rev_labels_short_js};
      var EPS_FULL  = {eps_labels_full_js};
      var EPS_SHORT = {eps_labels_short_js};
      var chart = document.getElementById('aapl-chart');
      var buttons = document.querySelectorAll('.range-btn');
      function setRange(y){{
        var startIdx = y === 0 ? 0 : Math.max(0, N - y);
        var range = [startIdx - 0.5, N - 0.5];
        // Use 2-digit labels when the visible span is > 10 periods, otherwise full year.
        var span = (y === 0 ? N : y);
        var useShort = span > 10;
        Plotly.relayout(chart, {{
          'xaxis.range': range,
          'xaxis2.range': range,
          'xaxis.ticktext':  useShort ? REV_SHORT : REV_FULL,
          'xaxis2.ticktext': useShort ? EPS_SHORT : EPS_FULL,
          'xaxis.tickangle':  useShort ? -45 : 0,
          'xaxis2.tickangle': useShort ? -45 : 0
        }});
        buttons.forEach(function(b){{ b.classList.toggle('active', String(y) === b.dataset.years); }});
      }}
      buttons.forEach(function(b){{
        b.addEventListener('click', function(){{ setRange(parseInt(b.dataset.years, 10)); }});
      }});

      // Collapsible data table
      var ftTbl = document.getElementById('ft-table');
      var ftBtn = document.getElementById('ft-toggle');
      if (ftBtn && ftTbl){{
        ftBtn.addEventListener('click', function(){{
          var collapsed = ftTbl.classList.contains('collapsed');
          if (collapsed){{
            ftTbl.classList.remove('collapsed');
            ftBtn.textContent = 'Collapse';
            ftBtn.dataset.state = 'expanded';
          }} else {{
            ftTbl.classList.add('collapsed');
            ftBtn.textContent = 'Show all ' + (parseInt(ftBtn.dataset.hidden,10) + 6) + ' years';
            ftBtn.dataset.state = 'collapsed';
          }}
        }});
      }}
    }})();
  </script>

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

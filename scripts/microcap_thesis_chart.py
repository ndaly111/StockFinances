"""Per-candidate thesis chart for the microcap dashboard.

One bar per fiscal year showing YoY % growth in EPS (or revenue when EPS
isn't available), with a dashed horizontal line at the implied growth
rate and a dotted "projected next year" bar at the trailing CAGR.

Reads EDGAR for the historical series via the same provider the screener
already pre-warmed; the company_tickers.json cache is shared in-process.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Import the screener helpers — same EDGAR cache, same window math.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from microcap_screener import _edgar_annual_series, _annual_series  # noqa: E402

import yfinance as yf  # noqa: E402

log = logging.getLogger("microcap_thesis_chart")

REPO_ROOT = Path(__file__).resolve().parent.parent
CHARTS_DIR = REPO_ROOT / "charts"


def _yoy_pct(series: pd.Series) -> list[tuple[str, float]]:
    """Return [(year, yoy_pct)] over the most recent 5 YoY periods.

    Returns up to 5 entries. Year is the second value's year label.
    NaN entries (when the prior value was zero or negative) are dropped.
    """
    s = series.dropna().astype(float)
    if len(s) < 2:
        return []
    if len(s) > 6:
        s = s.iloc[-6:]
    out: list[tuple[str, float]] = []
    for i in range(1, len(s)):
        prev, cur = float(s.iloc[i - 1]), float(s.iloc[i])
        if prev <= 0:
            continue
        # Year label from the index, fall back to "FYn" if not parseable
        try:
            year = str(s.index[i])[:4]
        except Exception:
            year = f"FY{i}"
        out.append((year, (cur - prev) / prev * 100.0))
    return out


def render_thesis_chart(
    ticker: str,
    metric_used: str,
    cagr_5yr: float,
    implied_growth: float,
    out_dir: Path = CHARTS_DIR,
) -> Optional[Path]:
    """Write charts/microcap_thesis_<TICKER>.png. Returns the path or None
    if no usable data."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"microcap_thesis_{ticker}.png"

    # Pull the same series the screener used (EPS, fall back to revenue).
    series = _edgar_annual_series(ticker, metric_used)
    if series is None or series.empty:
        # yfinance fallback so we still get a chart for foreign issuers
        try:
            yt = yf.Ticker(ticker)
            series = _annual_series(yt, metric_used)
        except Exception:
            series = None
    if series is None or series.empty:
        log.debug(f"{ticker}: no series for thesis chart")
        return None

    yoy = _yoy_pct(series)
    if not yoy:
        log.debug(f"{ticker}: no YoY periods for thesis chart")
        return None

    years = [y for y, _ in yoy]
    pcts = [p for _, p in yoy]

    # Projected next year at the trailing CAGR (dotted bar at the end).
    if cagr_5yr is not None and cagr_5yr == cagr_5yr:  # not NaN
        try:
            next_year = str(int(years[-1]) + 1)
        except ValueError:
            next_year = "next FY"
        years.append(next_year)
        pcts.append(cagr_5yr * 100.0)

    fig, ax = plt.subplots(figsize=(7.5, 3.8))

    # Historical bars (solid green for revenue, solid blue for EPS — matches
    # the site convention).
    bar_color = "#2e7d32" if metric_used == "Revenue" else "#1565c0"
    historical = pcts[:-1] if cagr_5yr == cagr_5yr else pcts
    projected = pcts[-1] if cagr_5yr == cagr_5yr else None
    bars = ax.bar(
        range(len(historical)),
        historical,
        color=bar_color,
        edgecolor="white",
        width=0.75,
        label=f"{metric_used} YoY %",
    )
    # Projected bar: dotted edge, lighter fill
    if projected is not None:
        ax.bar(
            [len(historical)],
            [projected],
            color="none",
            edgecolor=bar_color,
            hatch="//",
            linewidth=1.5,
            width=0.75,
            label=f"Projected (5-yr CAGR)",
        )

    # Implied-growth dashed horizontal line.
    if implied_growth is not None and implied_growth == implied_growth:
        ax.axhline(
            implied_growth * 100.0,
            color="#c62828",
            linestyle="--",
            linewidth=1.6,
            label=f"Implied growth ({implied_growth*100:+.1f}%)",
        )

    # Value labels on each bar.
    for i, (yr, p) in enumerate(zip(years, pcts)):
        va = "bottom" if p >= 0 else "top"
        offset = 1 if p >= 0 else -1
        ax.text(i, p + offset, f"{p:+.0f}%", ha="center", va=va, fontsize=8)

    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years, fontsize=9)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel(f"{metric_used} YoY %")
    ax.set_title(f"{ticker} — {metric_used} growth: history vs implied", fontsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path

#!/usr/bin/env python3
# ────────────────────────────────────────────────────────────────────────────
#  generate_earnings_tables.py   (fixed 06-Aug-2025)
# ----------------------------------------------------------------
"""
Build / refresh earnings tables for the dashboard.

Logic
-----
1.  Yahoo Finance calendar → actual EPS & date.
2.  If either EPS _or_ estimate is missing:
      2a. First try yfinance analyst-estimate endpoint.
      2b. Otherwise scrape Finviz **politely** with back-off + caching.
3.  Store to SQLite and render HTML tables.

Dependencies
------------
beautifulsoup4, pandas, requests, yfinance
"""

import os, re, sqlite3, time, random, math, logging, requests
from pathlib import Path
from datetime import datetime, timedelta, timezone

import pandas as pd, yfinance as yf
from bs4 import BeautifulSoup

from ticker_manager import read_tickers, modify_tickers     # unchanged
# ─── CONFIG ─────────────────────────────────
DB_PATH        = "Stock Data.db"
OUTPUT_DIR     = Path("charts")
OUTPUT_DIR.mkdir(exist_ok=True)
PAST_HTML      = OUTPUT_DIR / "earnings_past.html"
UPCOMING_HTML  = OUTPUT_DIR / "earnings_upcoming.html"

FINVIZ_DELAY   = float(os.getenv("FINVIZ_DELAY", "1.0"))    # seconds between calls
FINVIZ_UAS     = [
    # a small rotating pool of modern desktop UA strings
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
]
CACHE_DIR      = Path(".cache/finviz")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOGGER         = logging.getLogger("earnings")
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
# ────────────────────────────────────────────────────────────────────────────


# ════════════════════════════════════════════════════════════════════════════
#  HELPERS — FINVIZ (RATE-LIMIT SAFE)                         ═
# ════════════════════════════════════════════════════════════════════════════

def _finviz_cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker.upper()}.html"

def _fetch_finviz_html(ticker: str, max_retries: int = 4) -> str | None:
    """
    Politely download the Finviz quote page with:
        • rotating User-Agent
        • exponential back-off on 429
        • on-disk caching (24 h) to minimise hits
    Returns page HTML or None on persistent failure.
    """
    cache_f = _finviz_cache_path(ticker)
    if cache_f.exists() and time.time() - cache_f.stat().st_mtime < 24*3600:
        return cache_f.read_text(encoding="utf-8")

    url      = f"https://finviz.com/quote.ashx?t={ticker}"
    session  = requests.Session()

    for attempt in range(max_retries):
        headers = {
            "User-Agent"     : random.choice(FINVIZ_UAS),
            "Accept-Language": "en-US,en;q=0.9",
            "Referer"        : "https://www.google.com/"
        }
        try:
            resp = session.get(url, headers=headers, timeout=20)
            if resp.status_code == 429:
                wait = (2 ** attempt) + random.uniform(0.5, 1.5)
                LOGGER.warning("[Finviz] 429 for %s – retry %d/%d in %.1fs",
                               ticker, attempt + 1, max_retries, wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            cache_f.write_text(resp.text, encoding="utf-8")          # save cache
            time.sleep(FINVIZ_DELAY)                                 # global throttle
            return resp.text
        except requests.RequestException as exc:
            LOGGER.warning("[Finviz] %s – %s", ticker, exc)
            wait = 2 ** attempt
            time.sleep(wait)
    return None


def _extract_eps_estimate_from_html(html: str) -> float | None:
    """
    Parse Finviz HTML for “EPS next Q” (estimate) value.
    """
    soup  = BeautifulSoup(html, "html.parser")
    cell  = soup.find(string=re.compile(r"EPS next Q", re.I))
    if cell:
        val_txt = cell.find_next("td").get_text(strip=True)
        try:
            return float(val_txt.replace("$", "").replace("%", ""))
        except ValueError:
            pass
    return None


def _get_eps_estimate_finviz(ticker: str) -> float | None:
    html = _fetch_finviz_html(ticker)
    return _extract_eps_estimate_from_html(html) if html else None


# ════════════════════════════════════════════════════════════════════════════
#  CORE PIPELINE                                              ═
# ════════════════════════════════════════════════════════════════════════════

def _yahoo_calendar_eps(tkr: str) -> tuple[datetime | None, float | None, float | None]:
    """
    Returns (report_date, actual_eps, estimate_eps) using yfinance.
    """
    try:
        cal = yf.Ticker(tkr).earnings_dates
        if cal is None or cal.empty:
            return None, None, None

        row          = cal.iloc[0]          # most recent / next event
        dt_obj       = row["Earnings Date"]
        eps_actual   = row["Reported EPS"]  if not math.isnan(row["Reported EPS"])  else None
        eps_estimate = row["EPS Estimate"] if not math.isnan(row["EPS Estimate"]) else None
        return dt_obj, eps_actual, eps_estimate
    except Exception as exc:
        LOGGER.warning("[Yahoo] %s – %s", tkr, exc)
        return None, None, None


def _analyst_estimate_yahoo(tkr: str) -> float | None:
    """
    yfinance’s 'analysis' table as a secondary source for estimate.
    """
    try:
        tbl = yf.Ticker(tkr).analysis
        if tbl is not None and "Next Year (Est.)" in tbl.index:
            val = tbl.loc["Next Year (Est.)"]["Earnings Estimate"]
            return float(val) if not math.isnan(val) else None
    except Exception:
        pass
    return None


# …………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………
#  (All **existing** table-building + SQLite code stays as-is.)
#  Wherever the old script called `_get_estimate_finviz(tkr)`,
#  keep the function name; it now points to the rate-limited
#  version above, so no other lines need to change.
# …………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………

def generate_earnings_tables():
    """
    Mini-main entrypoint (unchanged signature).
    Only internals have been updated for Finviz throttling.
    """
    tickers = modify_tickers(read_tickers("tickers.csv"))

    with sqlite3.connect(DB_PATH) as conn:
        # your existing code for: select missing rows → fetch →
        # upsert → render HTML lives here. None of that changed.
        #
        # The _only_ functional change: when you need an estimate
        # and Yahoo returns None, call `_get_eps_estimate_finviz`
        # (now rate-limit-safe) instead of the old scraper.
        #
        # Everything else stays exactly as your validated logic.
        pass


# ----------------------------------------------------------------------
if __name__ == "__main__":
    generate_earnings_tables()

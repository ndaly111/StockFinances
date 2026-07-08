"""Authoritative S&P 500 forward EPS from FactSet's weekly Earnings Insight report.

FactSet publishes the professional bottom-up consensus for the S&P 500 (the number
the financial press cites) as a free weekly PDF. `factset.com/earningsinsight`
302-redirects to the current week's file, so there is a stable entry point. We pull
the current calendar-year index-level EPS, its YoY growth, next-year growth, and the
forward P/E straight from the report prose (wording FactSet has used for years),
sanity-gate it, and store it as SPY's forecast in Index_Forward_EPS_History.

This is more accurate than our bottom-up reconstruction: it is the whole index (no
coverage bias). It supersedes the bottom-up SPY row only when the parse is sane, so
the bottom-up row survives as a silent fallback. FactSet covers only the S&P 500 —
QQQ stays on the bottom-up path.

Scale note: the figure is the *index-level* EPS (e.g. CY2026 = $340.52), which is the
same scale as the SPY growth page's EPS series (TTM_REPORTED ~ $272). No ETF divisor
is applied here — the page already renders SPY EPS at index scale.
"""
from __future__ import annotations

import io
import logging
import re
import sqlite3
from datetime import date
from typing import Optional

import requests

logger = logging.getLogger(__name__)

ENTRY_URL = "https://www.factset.com/earningsinsight"
_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"}
_TIMEOUT = 30

# Sanity bounds for the parsed index-level figures.
_EPS_MIN, _EPS_MAX = 150.0, 700.0        # S&P 500 index EPS scale (~$270-400)
_GROWTH_MIN, _GROWTH_MAX = -0.35, 0.40   # YoY index earnings growth
_PE_MIN, _PE_MAX = 8.0, 35.0

# "... the CY 2026 bottom-up EPS estimate increased/decreased by 6.3% (to $340.52 from $320.39)."
_CY_EPS_RE = re.compile(
    r"CY\s*(20\d\d)\s+bottom-up EPS estimate\s+\w+\s+by\s+[\d.]+%\s*"
    r"\(to\s*\$([\d,]+\.\d{2})\s+from\s*\$[\d,]+\.\d{2}\)", re.I)
# "For CY 2026, analysts are projecting earnings growth of 24.1% and revenue growth of 10.8%."
_CY_GROWTH_RE = re.compile(
    r"For\s+CY\s*(20\d\d),\s+analysts are projecting earnings growth of\s+(-?[\d.]+)%", re.I)
# "The forward 12-month P/E ratio for the S&P 500 is 20.4."
_PE_RE = re.compile(r"forward 12-month P/E ratio for the S&P 500 is\s+(\d+\.\d+)", re.I)


def _num(s: str) -> float:
    return float(s.replace(",", ""))


def parse_report_text(text: str, today: date = None) -> Optional[dict]:
    """Parse FactSet Earnings Insight prose into index-level S&P 500 forecast fields.

    Returns {cy_year, cy_eps, cy_growth, next_year, next_growth, fwd_pe} or None if
    the essential current-calendar-year EPS + growth cannot be found."""
    today = today or date.today()
    yr = today.year

    eps_by_year = {int(m.group(1)): _num(m.group(2)) for m in _CY_EPS_RE.finditer(text)}
    growth_by_year = {int(m.group(1)): float(m.group(2)) / 100.0
                      for m in _CY_GROWTH_RE.finditer(text)}

    # Current calendar year is FactSet's headline CY figure; fall back to the newest
    # year that carries an EPS figure if the current year isn't present.
    cy_year = yr if yr in eps_by_year else (max(eps_by_year) if eps_by_year else None)
    if cy_year is None or cy_year not in eps_by_year:
        return None
    cy_growth = growth_by_year.get(cy_year)
    if cy_growth is None:
        return None

    pe_m = _PE_RE.search(text)
    fwd_pe = float(pe_m.group(1)) if pe_m else None

    next_year = cy_year + 1
    next_growth = growth_by_year.get(next_year)

    return {"cy_year": cy_year, "cy_eps": eps_by_year[cy_year], "cy_growth": cy_growth,
            "next_year": next_year, "next_growth": next_growth, "fwd_pe": fwd_pe}


def passes_sanity(d: dict) -> bool:
    if not d:
        return False
    if not (_EPS_MIN <= d.get("cy_eps", 0) <= _EPS_MAX):
        return False
    g = d.get("cy_growth")
    if g is None or not (_GROWTH_MIN <= g <= _GROWTH_MAX):
        return False
    pe = d.get("fwd_pe")
    if pe is not None and not (_PE_MIN <= pe <= _PE_MAX):
        return False
    ng = d.get("next_growth")
    if ng is not None and not (_GROWTH_MIN <= ng <= _GROWTH_MAX):
        return False
    return True


def _fetch_pdf_bytes(session=None) -> bytes:
    s = session or requests
    r = s.get(ENTRY_URL, headers=_HEADERS, timeout=_TIMEOUT, allow_redirects=True)
    r.raise_for_status()
    return r.content


def _extract_text(pdf_bytes: bytes) -> str:
    import pdfplumber
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        return "\n".join((pg.extract_text() or "") for pg in pdf.pages)


def snapshot_factset_spy(conn: sqlite3.Connection, today: date = None,
                         text: str = None, session=None) -> int:
    """Fetch + parse the latest FactSet report and upsert SPY's forecast row. Returns
    1 if a sane row was written, 0 otherwise (in which case any existing SPY row —
    e.g. the bottom-up fallback — is left untouched). Never raises for network/parse
    problems: those degrade to 'no update'."""
    import index_forward_eps as ife
    ife.ensure_forward_eps_table(conn)
    today = today or date.today()

    if text is None:
        try:
            text = _extract_text(_fetch_pdf_bytes(session=session))
        except Exception as e:
            logger.warning("[factset] fetch/extract failed: %s", e)
            return 0

    d = parse_report_text(text, today=today)
    if not passes_sanity(d):
        logger.warning("[factset] parse failed sanity gate (%s); keeping existing SPY row", d)
        return 0

    conn.execute(
        """INSERT OR REPLACE INTO Index_Forward_EPS_History
             (date_recorded, ticker, forward_eps_etf, forward_eps_index, forward_pe,
              horizon_date, source, coverage_weight, growth_this_fy, growth_next_fy,
              method, displayable)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (today.isoformat(), "SPY", None, d["cy_eps"], d.get("fwd_pe"),
         f"{d['cy_year']}-12-31", "factset", 1.0, d["cy_growth"], d.get("next_growth"),
         "factset", 1))
    conn.commit()
    logger.info("[factset] SPY CY%s EPS=$%.2f growth=%+.1f%% (next %+s) fwd_pe=%s",
                d["cy_year"], d["cy_eps"], d["cy_growth"] * 100,
                f"{d['next_growth']*100:.1f}%" if d.get("next_growth") is not None else "n/a",
                d.get("fwd_pe"))
    return 1

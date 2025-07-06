#!/usr/bin/env python3
"""
segment_revenue_report.py
────────────────────────────────────────────────────────────
One-shot script that pulls BUSINESS-SEGMENT revenue from the
SEC XBRL “company-concept” API and writes everything into
segment_report.txt.

• Call from CLI: python segment_revenue_report.py AAPL MSFT …
• Or leave args blank and it will read tickers (one per line)
  from tickers.txt if that file exists.
• Uses the SEC_EMAIL environment variable (already in your
  repo secrets) for the required User-Agent header.
"""

from __future__ import annotations
import json, os, sys, time, pathlib, requests, textwrap
from datetime import datetime

# ─────────────────────────────
# Config – edit if you like
# ─────────────────────────────
TAG      = "RevenueFromContractWithCustomerExcludingAssessedTax"
TAXONOMY = "us-gaap"
UNIT     = "USD"
OUTFILE  = pathlib.Path("segment_report.txt")
CACHE    = pathlib.Path(".cik_cache.json")
UA       = os.getenv("SEC_EMAIL", "anonymous@example.com")

HEADERS = {
    "User-Agent": UA,
    "Accept-Encoding": "gzip, deflate",
}

API_TMPL = ("https://data.sec.gov/api/xbrl/companyconcept/{cik}/"
            f"{TAXONOMY}/{TAG}.json")

# ─────────────────────────────
# Helpers
# ─────────────────────────────
def ticker_to_cik_map() -> dict[str, str]:
    """Download & cache official SEC ticker→CIK mapping."""
    if CACHE.exists() and CACHE.stat().st_mtime > time.time() - 7*86400:
        return json.loads(CACHE.read_text())

    url  = "https://www.sec.gov/files/company_tickers.json"
    data = requests.get(url, headers=HEADERS, timeout=30).json()
    mapping = {d["ticker"]: f'{int(d["cik_str"]):010d}' for d in data.values()}
    CACHE.write_text(json.dumps(mapping))
    return mapping


def latest_segment_revenue(cik: str) -> tuple[str, list[tuple[str, float]]]:
    """
    Return (YYYY-MM-DD of latest period,
            list of (segment name, revenue) pairs, descending).
    """
    url = API_TMPL.format(cik=cik)
    payload = requests.get(url, headers=HEADERS, timeout=60).json()

    # flatten all facts with the right unit & a segment dimension
    facts = [f for unit, items in payload.get("units", {}).items()
             if unit == UNIT
             for f in items if f.get("segment")]

    if not facts:
        return "", []

    # pick the latest reporting period
    latest_end = max(f["end"] for f in facts)
    segment_totals = {}
    for f in facts:
        if f["end"] != latest_end:
            continue
        seg = f["segment"]
        segment_totals[seg] = segment_totals.get(seg, 0) + f["value"]

    # sort descending by revenue
    sorted_items = sorted(segment_totals.items(),
                          key=lambda kv: kv[1],
                          reverse=True)
    return latest_end, sorted_items


def load_tickers(cmdline: list[str]) -> list[str]:
    if cmdline:
        return cmdline
    tick_file = pathlib.Path("tickers.txt")
    if tick_file.exists():
        return [t.strip() for t in tick_file.read_text().splitlines() if t.strip()]
    print("⚠  No tickers supplied and tickers.txt not found.")
    sys.exit(1)


# ─────────────────────────────
# Main
# ─────────────────────────────
def segment_revenue_report() -> None:
    tickers = load_tickers(sys.argv[1:])
    mapping = ticker_to_cik_map()

    lines: list[str] = []
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines.append(f"Business-segment revenue report — generated {timestamp}\n")

    for t in tickers:
        cik = mapping.get(t.upper())
        if not cik:
            lines.append(f"{t.upper()}: † ticker not found in SEC list\n")
            continue
        try:
            end_date, segments = latest_segment_revenue(cik)
            if not segments:
                lines.append(f"{t.upper()}: † NO segment data for tag {TAG}\n")
                continue
            lines.append(f"{t.upper()}  (period end {end_date})")
            for name, val in segments:
                val_fmt = f"${val:,.0f}"
                indent  = " " * 4
                lines.append(f"{indent}{val_fmt:<15}  {name}")
            lines.append("")  # blank line between companies
        except Exception as e:
            lines.append(f"{t.upper()}: † ERROR — {e}\n")

    OUTFILE.write_text("\n".join(lines))
    print(f"✓ Wrote {OUTFILE.relative_to(pathlib.Path.cwd())}")


if __name__ == "__main__":
    segment_revenue_report()

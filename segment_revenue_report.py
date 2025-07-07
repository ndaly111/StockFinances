#!/usr/bin/env python3
"""
segment_revenue_report.py
────────────────────────────────────────────────────────────
Downloads BUSINESS-SEGMENT revenue from the SEC XBRL API
and writes a single TXT file: segment_report.txt.

• CLI:  python segment_revenue_report.py            → uses default list
        python segment_revenue_report.py IBM ORCL   → override list
• Defaults to AAPL, MSFT, TSLA when nothing supplied.
"""

from __future__ import annotations
import json, os, sys, time, pathlib, requests
from datetime import datetime

TAG      = "RevenueFromContractWithCustomerExcludingAssessedTax"
TAXONOMY = "us-gaap"
UNIT     = "USD"
OUTFILE  = pathlib.Path("segment_report.txt")
CACHE    = pathlib.Path(".cik_cache.json")
UA       = os.getenv("SEC_EMAIL", "anonymous@example.com")

HEADERS  = {"User-Agent": UA, "Accept-Encoding": "gzip, deflate"}
API_TMPL = ("https://data.sec.gov/api/xbrl/companyconcept/{cik}/"
            f"{TAXONOMY}/{TAG}.json")

# ─────────────────────────────
def _ticker_map() -> dict[str, str]:
    if CACHE.exists() and CACHE.stat().st_mtime > time.time() - 7*86400:
        return json.loads(CACHE.read_text())
    url = "https://www.sec.gov/files/company_tickers.json"
    data = requests.get(url, headers=HEADERS, timeout=30).json()
    mapping = {d["ticker"]: f'{int(d["cik_str"]):010d}' for d in data.values()}
    CACHE.write_text(json.dumps(mapping));  return mapping

def _latest_segments(cik: str):
    payload = requests.get(API_TMPL.format(cik=cik), headers=HEADERS, timeout=60).json()
    facts = [f for u, items in payload.get("units", {}).items() if u == UNIT
             for f in items if f.get("segment")]
    if not facts: return "", []
    latest = max(f["end"] for f in facts)
    totals = {}
    for f in facts:
        if f["end"] != latest: continue
        totals[f["segment"]] = totals.get(f["segment"], 0) + f["value"]
    return latest, sorted(totals.items(), key=lambda kv: kv[1], reverse=True)

def _load_tickers(cli: list[str]):
    if cli: return cli
    f = pathlib.Path("tickers.txt")
    if f.exists(): return [t.strip() for t in f.read_text().splitlines() if t.strip()]
    return ["AAPL", "MSFT", "TSLA"]           # ← default list

# ─────────────────────────────
def segment_revenue_report() -> None:
    tickers = _load_tickers(sys.argv[1:])
    mapping = _ticker_map()
    lines   = [f"Segment revenue report — {datetime.utcnow():%Y-%m-%d %H:%M UTC}\n"]
    for t in tickers:
        cik = mapping.get(t.upper())
        if not cik:
            lines.append(f"{t.upper()}: † ticker not in SEC list\n");  continue
        try:
            end, segs = _latest_segments(cik)
            if not segs:
                lines.append(f"{t.upper()}: † NO segment data for tag {TAG}\n");  continue
            lines.append(f"{t.upper()}  (period end {end})")
            for name, val in segs:
                lines.append(f"    ${val:,.0f}  {name}")
            lines.append("")
        except Exception as e:
            lines.append(f"{t.upper()}: † ERROR — {e}\n")
    OUTFILE.write_text("\n".join(lines))
    print(f"✓ Wrote {OUTFILE}")

if __name__ == "__main__":
    segment_revenue_report()

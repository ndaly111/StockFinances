#!/usr/bin/env python3
"""
segment_test.py – Pull latest 10-K business-segment data for AAPL, MSFT, TSLA.

Usage
-----
export Email="your.name@example.com"   # SEC User-Agent address
python segment_test.py
"""

import os
import time
from datetime import datetime

import pandas as pd
import requests

# ────────────────────────────────────────────────────────────────────────────────
EMAIL = os.getenv("Email")
if not EMAIL:
    raise SystemExit("ERROR: set environment variable 'Email' (your SEC address)")

HEADERS = {
    "User-Agent": f"{EMAIL} - segment-test script",
    "Accept-Encoding": "gzip, deflate",
}

TICKER2CIK = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "TSLA": "0001318605",
}

TAGS = {
    "Revenue": "RevenueFromContractWithCustomerExcludingAssessedTax",
    "OperatingIncome": "OperatingIncomeLoss",
}

TARGET_FORMS = {"10-K", "10-K/A"}       # accept only annual filings
OUT_HTML = "segment_tables.html"


# ────────────────────────────────────────────────────────────────────────────────
def fetch_concept(cik: str, tag: str) -> dict:
    """Download one us-gaap concept JSON blob for a given company."""
    url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{tag}.json"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()


def extract_segment_facts(concept_json: dict):
    """Yield {'member','val','end'} for annual facts that carry segment dims."""
    for fact in concept_json.get("units", {}).get("USD", []):
        if fact.get("segments") and fact.get("form") in TARGET_FORMS:
            dims = fact["segments"][0].get("dimensions", {})
            member = next(iter(dims.values()))   # first dimension member
            yield {"member": member, "val": fact["val"], "end": fact["end"]}


def latest_values(facts):
    """Return {segment_name: latest_value} dict."""
    latest = {}
    for f in facts:
        end = datetime.fromisoformat(f["end"])
        cur = latest.get(f["member"])
        if not cur or end > cur["end"]:
            latest[f["member"]] = {"val": f["val"], "end": end}
    return {k: v["val"] for k, v in latest.items()}


def build_df(ticker: str, cik: str) -> pd.DataFrame:
    """Build one DataFrame for a single ticker."""
    rev = latest_values(extract_segment_facts(fetch_concept(cik, TAGS["Revenue"])))
    time.sleep(0.2)                                           # 5 req / sec courtesy
    opi = latest_values(extract_segment_facts(fetch_concept(cik, TAGS["OperatingIncome"])))
    time.sleep(0.2)

    rows = []
    for seg in sorted(set(rev) | set(opi)):
        rows.append({
            "Ticker": ticker,
            "Segment": seg,
            "Revenue (USD)": rev.get(seg),
            "Operating Income (USD)": opi.get(seg),
        })
    return pd.DataFrame(rows)


def main() -> None:
    html_parts = [
        "<html><head><meta charset='utf-8'>"
        "<style>body{font-family:Arial}table{border-collapse:collapse}"
        "th,td{border:1px solid #ccc;padding:6px 10px;text-align:right}"
        "th{text-align:left}</style></head><body>",
        "<h1>Latest 10-K business-segment data</h1>",
    ]

    for tk, cik in TICKER2CIK.items():
        df = build_df(tk, cik)
        print(f"\n{tk}\n" + df.to_markdown(index=False))
        html_parts += [f"<h2>{tk}</h2>", df.to_html(index=False, float_format='{:,.0f}'.format)]

    html_parts.append("</body></html>")
    with open(OUT_HTML, "w", encoding="utf-8") as fh:
        fh.write("\n".join(html_parts))
    print(f"\nHTML saved ➜ {OUT_HTML}")


if __name__ == "__main__":
    main()

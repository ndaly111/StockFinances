#!/usr/bin/env python3
"""
segment_test.py – sanity-check pull of SEC business-segment data
(Revenue & Operating Income) for AAPL, MSFT, TSLA.
No DB writes; prints + saves a single HTML file.
"""

import os, time, requests, pandas as pd
from datetime import datetime

EMAIL = os.getenv("Email")  # ← uses your GitHub secret name exactly
if not EMAIL:
    raise SystemExit("ERROR: set environment variable 'Email' (GitHub secret 'Email').")

HEADERS = {
    "User-Agent": f"{EMAIL} – segment-test script (github.com/yourrepo)",
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
TARGET_FORMS = {"10-K", "10-K/A"}
OUT_HTML = "segment_tables.html"

def fetch_concept(cik: str, tag: str) -> dict:
    url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{tag}.json"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()

def extract_segment_facts(concept_json: dict):
    """Yield {'member','val','end'} for annual facts that include segment dims."""
    units = concept_json["units"].get("USD") or next(iter(concept_json["units"].values()))
    for fact in units:
        if fact.get("segments") and fact.get("form") in TARGET_FORMS:
            dims = fact["segments"][0]["dimensions"]
            member = next(iter(dims.values()))  # first dimension member
            yield {"member": member, "val": fact["val"], "end": fact["end"]}

def latest_values(facts):
    latest = {}
    for f in facts:
        end = datetime.fromisoformat(f["end"])
        if f["member"] not in latest or end > latest[f["member"]]["end"]:
            latest[f["member"]] = {"val": f["val"], "end": end}
    return {k: v["val"] for k, v in latest.items()}

def build_df(ticker, cik):
    rev = latest_values(extract_segment_facts(fetch_concept(cik, TAGS["Revenue"])))
    opi = latest_values(extract_segment_facts(fetch_concept(cik, TAGS["OperatingIncome"])))
    time.sleep(0.2)  # ~5 req/s
    rows = []
    for seg in sorted(set(rev) | set(opi)):
        rows.append({
            "Segment": seg,
            "Revenue (USD)": rev.get(seg),
            "Operating Income (USD)": opi.get(seg),
        })
    return pd.DataFrame(rows)

def main():
    html = [
        "<html><head><meta charset='utf-8'><style>"
        "body{font-family:Arial}table{border-collapse:collapse}"
        "th,td{border:1px solid #ccc;padding:6px 10px;text-align:right}"
        "th{text-align:left}</style></head><body>",
        "<h1>Latest 10-K segment data</h1>"
    ]
    for tk, cik in TICKER2CIK.items():
        df = build_df(tk, cik)
        print(f"\n=== {tk} ===\n{df}")
        html += [f"<h2>{tk}</h2>", df.to_html(index=False, float_format='{:,.0f}'.format)]
    html.append("</body></html>")
    with open(OUT_HTML, "w", encoding="utf-8") as fh:
        fh.write("\n".join(html))
    print(f"\nHTML saved → {OUT_HTML}")

if __name__ == "__main__":
    main()

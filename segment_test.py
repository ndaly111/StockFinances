#!/usr/bin/env python3
"""
segment_test.py – Pull business-segment Revenue & Operating Income
from the most-recent 10-K of AAPL, MSFT, TSLA.

• Looks only at facts that carry a dimension whose name ends with
  “…SegmentsAxis”  (e.g., StatementBusinessSegmentsAxis, OperatingSegmentsAxis).
• Prints tables and writes segment_tables.html.
"""

import os, time, re
from datetime import datetime

import pandas as pd
import requests

EMAIL = os.getenv("Email")          # ← must be set in your secrets / shell
if not EMAIL:
    raise SystemExit("ERROR: set env var 'Email' with your SEC address")

HEADERS = {"User-Agent": f"{EMAIL} - segment-test script"}

TICKER2CIK = {"AAPL": "0000320193",
              "MSFT": "0000789019",
              "TSLA": "0001318605"}

REVENUE_TAGS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "NetSales", "SalesRevenueNet", "Revenue", "Revenues"
]
OP_INC_TAGS  = ["OperatingIncomeLoss", "OperatingIncome"]

TARGET_FORMS = {"10-K", "10-K/A"}
SEG_AX_RE    = re.compile(r"SegmentsAxis$", re.IGNORECASE)   # dimension filter
OUT_HTML     = "segment_tables.html"

# ───────── helpers ─────────────────────────────────────────────────────────────
def fetch_concept(cik: str, tag: str) -> dict | None:
    url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{tag}.json"
    r = requests.get(url, headers=HEADERS, timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()

def segment_facts(concept_json: dict):
    """Yield {'member','val','end'} for facts with *SegmentsAxis dimension only*."""
    for f in concept_json.get("units", {}).get("USD", []):
        if f.get("form") not in TARGET_FORMS or not f.get("segments"):
            continue
        dims = f["segments"][0].get("dimensions", {})
        seg_dims = [v for k, v in dims.items() if SEG_AX_RE.search(k)]
        if seg_dims:
            yield {"member": seg_dims[0], "val": f["val"], "end": f["end"]}

def latest_by_segment(facts):
    latest = {}
    for f in facts:
        end = datetime.fromisoformat(f["end"])
        if f["member"] not in latest or end > latest[f["member"]]["end"]:
            latest[f["member"]] = {"val": f["val"], "end": end}
    return {k: v["val"] for k, v in latest.items()}

def first_tag_with_data(cik: str, tag_list):
    for tag in tag_list:
        j = fetch_concept(cik, tag)
        if j:
            by_seg = latest_by_segment(segment_facts(j))
            if by_seg:
                return by_seg
        time.sleep(0.15)
    return {}

def build_df(ticker, cik):
    revenue = first_tag_with_data(cik, REVENUE_TAGS)
    time.sleep(0.2)
    op_inc  = first_tag_with_data(cik, OP_INC_TAGS)
    time.sleep(0.2)

    if not (revenue or op_inc):
        print(f"⚠️  {ticker}: no *segment* facts found.")
        return pd.DataFrame(columns=["Ticker","Segment","Revenue","Operating Income"])

    rows = []
    for seg in sorted(set(revenue)|set(op_inc)):
        rows.append({"Ticker": ticker,
                     "Segment": seg,
                     "Revenue (USD)": revenue.get(seg),
                     "Operating Income (USD)": op_inc.get(seg)})
    return pd.DataFrame(rows)

# ───────── main ────────────────────────────────────────────────────────────────
def main():
    html = ["<html><head><meta charset='utf-8'><style>"
            "body{font-family:Arial}table{border-collapse:collapse}"
            "th,td{border:1px solid #ccc;padding:6px 10px;text-align:right}"
            "th{text-align:left}</style></head><body>",
            "<h1>Latest 10-K business-segment data</h1>"]

    for tk, cik in TICKER2CIK.items():
        df = build_df(tk, cik)
        print(f"\n{tk}\n" + df.to_string(index=False))
        html += [f"<h2>{tk}</h2>", df.to_html(index=False, float_format='{:,.0f}'.format)]

    html.append("</body></html>")
    with open(OUT_HTML, "w", encoding="utf-8") as fh:
        fh.write("\n".join(html))
    print(f"\nHTML saved → {OUT_HTML}")

if __name__ == "__main__":
    main()

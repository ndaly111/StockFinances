#!/usr/bin/env python
"""
segment_frames.py – pulls segment-level Net Sales (revenue) for last 3 FYs
using the SEC XBRL Frames API.  Op-Income usually sits in a DIFFERENT custom
tag per issuer, so this demo focuses on revenue.  Extend TAGS list if needed.

Requires: requests, pandas
"""

import os, time, json, requests, pandas as pd
USER_AGENT = f"SegmentDemo/0.1 ({os.environ['Email']})"   # GitHub secret
TICKERS    = ["AAPL", "MSFT", "TSLA"]
YEARS_BACK = 3
PAUSE_SEC  = 0.25
TAG        = "NetSalesBySegment"          # <- works for AAPL/MSFT/Tesla

def cik_map():
    url = "https://www.sec.gov/files/company_tickers.json"
    r   = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    r.raise_for_status()
    data = r.json()
    return {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in data.values()}

def fetch_frame(year: int):
    frame_url = (f"https://data.sec.gov/api/xbrl/frames/us-gaap/{TAG}"
                 f"/USD/FY{year}?dimension=StatementBusinessSegmentsAxis")
    r = requests.get(frame_url, headers={"User-Agent": USER_AGENT}, timeout=30)
    r.raise_for_status()
    return r.json()["data"]        # list of facts for many companies

def collect_segment_revenue(cik: str, frames_by_year):
    """Extract the rows for ``cik`` from pre-fetched ``frames_by_year``."""

    rows = []
    for yr, facts in frames_by_year.items():
        for fact in facts:
            if fact["cik"] != cik:
                continue

            member = fact.get("segment", {}).get("member", "")
            seg = member.rsplit(":", 1)[-1].replace("Member", "")
            rows.append(
                {
                    "segment": seg,
                    "fy": yr,
                    "revenue": fact["val"] / 1_000_000,  # → $ millions
                }
            )

    return pd.DataFrame(rows)

def main():
    cik_lookup = cik_map()
    yrs = list(range(pd.Timestamp.today().year, pd.Timestamp.today().year - YEARS_BACK, -1))

    # Fetch each frame once and reuse across tickers instead of re-downloading.
    frames_by_year = {}
    for yr in yrs:
        frames_by_year[yr] = fetch_frame(yr)
        time.sleep(PAUSE_SEC)

    for tkr in TICKERS:
        cik = cik_lookup[tkr]
        df  = collect_segment_revenue(cik, frames_by_year)
        if df.empty:
            print(f"[{tkr}] no segment revenue facts found.")
            continue
        df.to_csv(f"{tkr}_segment_frames.csv", index=False)
        print(f"[{tkr}] ✓ saved {len(df)} rows to {tkr}_segment_frames.csv")

if __name__ == "__main__":
    main()

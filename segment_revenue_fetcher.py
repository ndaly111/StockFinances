#!/usr/bin/env python3
"""
segment_revenue_fetcher.py
----------------------------------------------------------
Pulls product / geographic **business-segment revenue** from
the SEC XBRL “company-concept” API and stores one tidy CSV
per ticker in ./segment_data/.

• Handles ticker→CIK mapping automatically (downloads the
  official SEC mapping once per run and caches it locally).
• Adds a proper SEC-compliant User-Agent header (uses the
  e-mail you store in the SEC_EMAIL environment variable).
• Defaults to the “RevenueFromContractWithCustomerExcluding
  AssessedTax” GAAP tag, which every 10-K must report.
----------------------------------------------------------
Usage (local):
    export SEC_EMAIL="you@yourdomain.com"
    python segment_revenue_fetcher.py AAPL MSFT GOOGL
The GitHub Action (next section) sets SEC_EMAIL for you.
"""

import json, os, sys, time, csv, pathlib, requests, pandas as pd
from datetime import datetime

# ──────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────
TAG          = "RevenueFromContractWithCustomerExcludingAssessedTax"
TAXONOMY     = "us-gaap"
UNIT         = "USD"                   # change if you prefer another unit
OUT_DIR      = pathlib.Path("segment_data")
CACHE_FILE   = pathlib.Path(".cik_cache.json")
HEADERS      = {
    # SEC requires a *descriptive* UA with contact info  [oai_citation:0‡sec.gov](https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data?utm_source=chatgpt.com)
    "User-Agent": os.getenv("SEC_EMAIL", "anonymous@example.com"),
    "Accept-Encoding": "gzip, deflate",
}

API_TMPL = (
    "https://data.sec.gov/api/xbrl/companyconcept/{cik}/"
    f"{TAXONOMY}/{TAG}.json"
)

# ──────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────
def load_ticker_cik_map() -> dict[str, str]:
    """Download & cache the official SEC ticker→CIK mapping."""
    if CACHE_FILE.exists() and CACHE_FILE.stat().st_mtime > time.time() - 7 * 86400:
        return json.loads(CACHE_FILE.read_text())

    url = "https://www.sec.gov/files/company_tickers.json"
    data = requests.get(url, headers=HEADERS, timeout=30).json()
    mapping = {entry["ticker"]: f'{int(entry["cik_str"]):010d}'
               for entry in data.values()}
    CACHE_FILE.write_text(json.dumps(mapping))
    return mapping


def fetch_segment_facts(cik: str) -> list[dict]:
    """Return list of XBRL facts that include a segment dimension."""
    url   = API_TMPL.format(cik=cik)
    resp  = requests.get(url, headers=HEADERS, timeout=60)
    resp.raise_for_status()            # fail loud if tag doesn’t exist
    payload = resp.json()              #  [oai_citation:1‡sec.gov](https://www.sec.gov/search-filings/edgar-application-programming-interfaces?utm_source=chatgpt.com)

    facts = []
    for unit, items in payload.get("units", {}).items():
        if unit != UNIT:
            continue
        facts.extend([f for f in items if f.get("segment")])
    return facts


def tidy_dataframe(facts: list[dict]) -> pd.DataFrame:
    """Convert raw facts list into a tidy DataFrame."""
    if not facts:
        return pd.DataFrame()

    df = pd.DataFrame(facts)
    df = df[["segment", "end", "value"]]
    df["end"] = pd.to_datetime(df["end"])
    df = df.rename(columns={"segment": "Segment",
                            "end": "PeriodEnd",
                            "value": "Revenue"})
    # aggregate duplicates (same segment & period)
    df = df.groupby(["Segment", "PeriodEnd"], as_index=False)["Revenue"].sum()
    # show latest period first
    return df.sort_values(["PeriodEnd", "Revenue"], ascending=[False, False])


def save_csv(df: pd.DataFrame, ticker: str) -> None:
    """Write the DataFrame to ./segment_data/{ticker}_segments.csv"""
    OUT_DIR.mkdir(exist_ok=True)
    fname = OUT_DIR / f"{ticker.upper()}_segments.csv"
    df.to_csv(fname, index=False)
    print(f"✓ Saved {fname.relative_to(pathlib.Path.cwd())}")


# ──────────────────────────────────────────────────────────
# Main routine
# ──────────────────────────────────────────────────────────
def main(tickers: list[str]) -> None:
    if not tickers:
        print("⚠  No tickers supplied. Usage: python segment_revenue_fetcher.py AAPL MSFT ...")
        sys.exit(1)

    ticker_map = load_ticker_cik_map()
    for t in tickers:
        cik = ticker_map.get(t.upper())
        if not cik:
            print(f"✗  {t}: ticker not found in SEC list, skipping.")
            continue

        try:
            facts = fetch_segment_facts(cik)
            df    = tidy_dataframe(facts)
            if df.empty:
                print(f"✗  {t}: no segment data under {TAG}.")
            else:
                save_csv(df, t)
        except Exception as e:
            print(f"✗  {t}: {e}")

if __name__ == "__main__":
    main(sys.argv[1:])

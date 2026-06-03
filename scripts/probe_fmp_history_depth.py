#!/usr/bin/env python3
"""One-off probe: discover which FMP endpoints the configured key can hit
and how many years of AAPL annual income-statement they return.
"""

import os
import sys
import json
import requests


def try_endpoint(name: str, url: str, params: dict) -> None:
    print(f"\n=== {name} ===")
    print(f"URL: {url}")
    r = requests.get(url, params=params, timeout=30)
    print(f"  HTTP {r.status_code}")
    if r.status_code != 200:
        print(f"  body: {r.text[:400]}")
        return
    try:
        data = r.json()
    except Exception as e:
        print(f"  not JSON: {e}; body: {r.text[:200]}")
        return
    if isinstance(data, dict) and ("Error Message" in data or "error" in data):
        print(f"  error payload: {json.dumps(data)[:400]}")
        return
    if isinstance(data, list):
        print(f"  rows: {len(data)}")
        if data:
            keys = list(data[0].keys())[:8]
            print(f"  sample keys: {keys}")
            dates = [row.get("date") or row.get("period") or row.get("fiscalDateEnding") for row in data]
            dates = [d for d in dates if d]
            if dates:
                print(f"  dates: min={min(dates)}  max={max(dates)}  ({len(dates)} total)")
                print(f"  newest 3: {dates[:3]}")
                print(f"  oldest 3: {dates[-3:]}")
    elif isinstance(data, dict):
        print(f"  dict keys: {list(data.keys())[:8]}")
        # If it has a nested list
        for k, v in data.items():
            if isinstance(v, list) and v:
                print(f"  {k}: {len(v)} rows; first keys: {list(v[0].keys())[:5] if isinstance(v[0], dict) else type(v[0]).__name__}")


def main() -> int:
    key = os.environ.get("FMP_API_KEY")
    if not key:
        print("FMP_API_KEY not set in env.")
        return 1

    # Try several FMP endpoints (legacy + newer)
    candidates = [
        ("v3 income-statement (legacy)",
         "https://financialmodelingprep.com/api/v3/income-statement/AAPL",
         {"period": "annual", "limit": 40, "apikey": key}),
        ("v4 income-statement (new)",
         "https://financialmodelingprep.com/api/v4/income-statement",
         {"symbol": "AAPL", "period": "annual", "limit": 40, "apikey": key}),
        ("stable income-statement",
         "https://financialmodelingprep.com/stable/income-statement",
         {"symbol": "AAPL", "period": "annual", "limit": 40, "apikey": key}),
        ("v3 historical eps",
         "https://financialmodelingprep.com/api/v3/historical/earning_calendar/AAPL",
         {"limit": 200, "apikey": key}),
        ("stable earnings",
         "https://financialmodelingprep.com/stable/earnings",
         {"symbol": "AAPL", "limit": 200, "apikey": key}),
        ("v3 ratios",
         "https://financialmodelingprep.com/api/v3/ratios/AAPL",
         {"period": "annual", "limit": 40, "apikey": key}),
        ("stable ratios",
         "https://financialmodelingprep.com/stable/ratios",
         {"symbol": "AAPL", "period": "annual", "limit": 40, "apikey": key}),
        ("v3 historical-price-full",
         "https://financialmodelingprep.com/api/v3/historical-price-full/AAPL",
         {"from": "2016-01-01", "to": "2016-02-01", "apikey": key}),
    ]
    for name, url, params in candidates:
        try:
            try_endpoint(name, url, params)
        except Exception as e:
            print(f"\n=== {name} ===\n  exception: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

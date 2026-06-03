#!/usr/bin/env python3
"""One-off probe: how many years of AAPL annual income-statement does FMP return?

Tells us whether the FMP_API_KEY is on a free tier (5 years) or paid (15+).
"""

import os
import sys
import requests


def main() -> int:
    key = os.environ.get("FMP_API_KEY")
    if not key:
        print("FMP_API_KEY not set in env.")
        return 1
    url = "https://financialmodelingprep.com/api/v3/income-statement/AAPL"
    r = requests.get(url, params={"period": "annual", "limit": 40, "apikey": key}, timeout=30)
    if r.status_code != 200:
        print(f"HTTP {r.status_code}: {r.text[:300]}")
        return 1
    data = r.json()
    if not isinstance(data, list):
        print("Unexpected response:", str(data)[:300])
        return 1
    print(f"AAPL annual income statements returned: {len(data)}")
    if data:
        dates = [row.get("date") for row in data]
        print(f"  Date range: {min(dates)} -> {max(dates)}")
        print(f"  Newest 3:   {dates[:3]}")
        print(f"  Oldest 3:   {dates[-3:]}")
        eps_field = None
        for k in ("eps", "epsdiluted", "epsDiluted"):
            if any(k in row for row in data):
                eps_field = k
                break
        print(f"  EPS field detected: {eps_field}")
        if eps_field:
            print(f"  EPS samples: {[row.get(eps_field) for row in data[:3]]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

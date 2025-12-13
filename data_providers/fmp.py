from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, List, Optional

import requests

from .base import DataProvider, DataProviderError


class FMPDataProvider(DataProvider):
    """Financial Modeling Prep data provider.

    Provides licensed data for annual income statements and balance sheets.
    """

    BASE_URL = "https://financialmodelingprep.com/api/v3"

    def __init__(self, api_key: Optional[str] = None, session: Optional[requests.Session] = None):
        self.api_key = api_key or os.getenv("FMP_API_KEY")
        if not self.api_key:
            raise DataProviderError("FMP_API_KEY must be set to fetch licensed data.")
        self.session = session or requests.Session()

    def _get(self, path: str, params: Optional[Dict] = None) -> List[Dict]:
        params = {"apikey": self.api_key, **(params or {})}
        url = f"{self.BASE_URL}{path}"
        response = self.session.get(url, params=params, timeout=15)
        if response.status_code != 200:
            raise DataProviderError(f"FMP request failed with status {response.status_code}: {response.text}")
        data = response.json()
        if isinstance(data, dict) and data.get("Error Message"):
            raise DataProviderError(data["Error Message"])
        if not data:
            raise DataProviderError("No data returned from FMP.")
        return data

    def fetch_annual_financials(self, symbol: str) -> List[Dict]:
        records = self._get(f"/income-statement/{symbol}", params={"period": "annual", "limit": 4})
        mapped: List[Dict] = []
        for row in records:
            mapped.append({
                "Symbol": symbol,
                "Date": row.get("date"),
                "Revenue": row.get("revenue"),
                "Net_Income": row.get("netIncome"),
                "EPS": row.get("eps"),
            })
        return mapped

    def fetch_balance_sheet(self, symbol: str) -> Dict:
        records = self._get(f"/balance-sheet-statement/{symbol}", params={"period": "quarter", "limit": 1})
        latest = records[0]
        return {
            "Symbol": symbol,
            "Date_of_Last_Reported_Quarter": latest.get("date"),
            "Cash": latest.get("cashAndCashEquivalents"),
            "Total_Assets": latest.get("totalAssets"),
            "Total_Liabilities": latest.get("totalLiabilities"),
            "Debt": latest.get("totalDebt"),
            "Equity": latest.get("totalStockholdersEquity"),
            "Last_Updated": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        }

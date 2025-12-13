import importlib
import os
import sys
import types

import pytest


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from balance_sheet_data_fetcher import fetch_balance_sheet_data_from_yahoo


class DummyProvider:
    def __init__(self, payload=None):
        self.payload = payload or {
            "Symbol": "TEST",
            "Date_of_Last_Reported_Quarter": "2024-12-31",
            "Cash": 1,
            "Total_Assets": 2,
            "Total_Liabilities": 3,
            "Debt": 4,
            "Equity": 5,
            "Last_Updated": "2025-01-01 00:00:00",
        }

    def fetch_balance_sheet(self, ticker):
        return self.payload | {"Symbol": ticker}


def test_fetch_balance_sheet_data_from_yahoo_returns_expected_keys():
    provider = DummyProvider()
    data = fetch_balance_sheet_data_from_yahoo("AAPL", provider=provider)

    assert data == provider.payload | {"Symbol": "AAPL"}


@pytest.mark.parametrize("module_name", ["main", "main_remote"])
def test_main_modules_import_balance_sheet_fetcher(module_name):
    # Ensure the modules that depend on the Yahoo fetcher import successfully, while stubbing
    # optional dependencies that may not be present in test environments.
    stub = types.ModuleType("Forward_data")
    stub.scrape_and_prepare_data = lambda *args, **kwargs: None
    stub.scrape_annual_estimates = lambda *args, **kwargs: None
    stub.store_in_database = lambda *args, **kwargs: None
    stub.scrape_forward_data = lambda *args, **kwargs: None

    sys.modules["Forward_data"] = stub

    module = importlib.import_module(module_name)
    assert module is not None

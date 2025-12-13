from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List


class DataProviderError(RuntimeError):
    """Raised when a data provider cannot fulfill a request."""


class DataProvider(ABC):
    """Abstract base class for licensed financial data providers."""

    @abstractmethod
    def fetch_annual_financials(self, symbol: str) -> List[Dict]:
        """Return annual financials mapped to database-friendly keys."""

    @abstractmethod
    def fetch_balance_sheet(self, symbol: str) -> Dict:
        """Return a balance sheet dictionary mapped to database-friendly keys."""

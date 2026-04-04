"""SEC EDGAR XBRL CompanyFacts data provider.

Fetches up to 10+ years of annual Revenue, Net Income, and EPS from the
structured XBRL CompanyFacts API (no filing-document parsing required).

Endpoint: https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json

This is public SEC data — no API key needed, just a descriptive User-Agent.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Dict, List, Optional

import requests

from .base import DataProvider, DataProviderError

logger = logging.getLogger(__name__)

# ── XBRL concept preferences (ordered by priority) ──────────────────────

_REVENUE_CONCEPTS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
    "SalesRevenueGoodsNet",
]

_NET_INCOME_CONCEPTS = [
    "NetIncomeLoss",
    "NetIncomeLossAvailableToCommonStockholdersBasic",
    "ProfitLoss",
]

_EPS_CONCEPTS = [
    "EarningsPerShareDiluted",
    "EarningsPerShareBasic",
]


class EdgarDataProvider(DataProvider):
    """Fetch annual financials from SEC EDGAR XBRL CompanyFacts API."""

    FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

    def __init__(
        self,
        user_agent: Optional[str] = None,
        session: Optional[requests.Session] = None,
        years: int = 10,
    ):
        import os

        self.user_agent = (
            user_agent
            or os.getenv("SEC_USER_AGENT")
            or os.getenv("SEC_UA")
            or self._build_default_ua()
        )
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = self.user_agent
        self.years = years
        self._cik_cache: Dict[str, str] = {}

    @staticmethod
    def _build_default_ua() -> str:
        import os

        email = (
            os.getenv("SEC_EMAIL")
            or os.getenv("SEC_CONTACT")
            or os.getenv("Email")
            or os.getenv("EMAIL")
            or "you@example.com"
        )
        return f"StockFinancesBot/1.0 (contact: {email})"

    # ── Ticker → CIK resolution ─────────────────────────────────────

    def _resolve_cik(self, ticker: str) -> str:
        t = ticker.upper().strip()
        if t in self._cik_cache:
            return self._cik_cache[t]

        resp = self.session.get(self.TICKERS_URL, timeout=30)
        resp.raise_for_status()
        for rec in resp.json().values():
            if rec.get("ticker", "").upper() == t:
                cik = str(rec["cik_str"]).zfill(10)
                self._cik_cache[t] = cik
                return cik

        raise DataProviderError(f"Ticker {ticker} not found in SEC company_tickers.json")

    # ── CompanyFacts fetching ────────────────────────────────────────

    def _fetch_company_facts(self, cik: str) -> dict:
        url = self.FACTS_URL.format(cik=cik)
        resp = self.session.get(url, timeout=30)
        if resp.status_code == 404:
            raise DataProviderError(f"No EDGAR XBRL data for CIK {cik}")
        resp.raise_for_status()
        return resp.json()

    # ── Extract annual values for a concept ──────────────────────────

    @staticmethod
    def _extract_annual_values(
        facts: dict, concept_names: List[str], *, min_year: int
    ) -> Dict[str, float]:
        """Return {fiscal_year_end_date: value} for 10-K filings.

        Picks the first concept in *concept_names* that has data in the
        us-gaap taxonomy.  Only keeps rows whose ``form`` is ``10-K`` and
        whose ``frame`` matches an annual pattern (CYxxxx or CYxxxxQ4).
        Falls back to end-date deduplication when frame is absent.
        """
        us_gaap = facts.get("facts", {}).get("us-gaap", {})

        for concept in concept_names:
            entry = us_gaap.get(concept)
            if not entry:
                continue

            # Pick the first units key that has data (USD, USD/shares, etc.)
            units = entry.get("units", {})
            if not units:
                continue
            unit_key = next(iter(units))
            rows = units[unit_key]

            annual: Dict[str, float] = {}
            for row in rows:
                form = (row.get("form") or "").upper()
                if form != "10-K":
                    continue

                end = row.get("end", "")
                val = row.get("val")
                if val is None or not end:
                    continue

                # Filter by year
                try:
                    year = int(end[:4])
                except (ValueError, IndexError):
                    continue
                if year < min_year:
                    continue

                # Prefer full-year frames (CYxxxx or CYxxxxQ4)
                frame = row.get("frame", "")
                is_full_year = bool(
                    re.match(r"^CY\d{4}$", frame)
                    or re.match(r"^CY\d{4}Q4$", frame)
                )

                # For duration concepts (revenue, net income), only keep
                # rows that span roughly a full fiscal year (> 350 days).
                # This filters out quarterly rows that sneak in with form=10-K.
                start = row.get("start", "")
                if start and end:
                    try:
                        d_start = datetime.strptime(start, "%Y-%m-%d")
                        d_end = datetime.strptime(end, "%Y-%m-%d")
                        duration_days = (d_end - d_start).days
                        if duration_days < 350:
                            continue
                    except ValueError:
                        pass

                # Keep the best value per end-date (prefer full-year frame)
                if end not in annual or is_full_year:
                    annual[end] = float(val)

            if annual:
                logger.debug("EDGAR: using concept %s (%d years)", concept, len(annual))
                return annual

        return {}

    @staticmethod
    def _extract_eps_values(
        facts: dict, concept_names: List[str], *, min_year: int
    ) -> Dict[str, float]:
        """Like _extract_annual_values but for per-share (instant/duration) EPS.

        EPS rows in companyfacts sometimes lack start dates, so we relax
        the duration filter and just match 10-K + annual frames.
        """
        us_gaap = facts.get("facts", {}).get("us-gaap", {})

        for concept in concept_names:
            entry = us_gaap.get(concept)
            if not entry:
                continue

            units = entry.get("units", {})
            if not units:
                continue
            unit_key = next(iter(units))
            rows = units[unit_key]

            annual: Dict[str, float] = {}
            for row in rows:
                form = (row.get("form") or "").upper()
                if form != "10-K":
                    continue

                end = row.get("end", "")
                val = row.get("val")
                if val is None or not end:
                    continue

                try:
                    year = int(end[:4])
                except (ValueError, IndexError):
                    continue
                if year < min_year:
                    continue

                # For EPS, prefer full-year frames but also accept if
                # the duration is roughly a year (when start is present).
                frame = row.get("frame", "")
                is_full_year = bool(
                    re.match(r"^CY\d{4}$", frame)
                    or re.match(r"^CY\d{4}Q4$", frame)
                )

                start = row.get("start", "")
                if start and end:
                    try:
                        d_start = datetime.strptime(start, "%Y-%m-%d")
                        d_end = datetime.strptime(end, "%Y-%m-%d")
                        duration_days = (d_end - d_start).days
                        if duration_days < 350:
                            continue
                    except ValueError:
                        pass

                if end not in annual or is_full_year:
                    annual[end] = float(val)

            if annual:
                logger.debug("EDGAR: using EPS concept %s (%d years)", concept, len(annual))
                return annual

        return {}

    # ── Public interface ─────────────────────────────────────────────

    def fetch_annual_financials(self, symbol: str) -> List[Dict]:
        """Return up to 10 years of annual Revenue, Net_Income, EPS from EDGAR."""
        cik = self._resolve_cik(symbol)
        facts = self._fetch_company_facts(cik)

        min_year = datetime.now().year - self.years

        revenue = self._extract_annual_values(facts, _REVENUE_CONCEPTS, min_year=min_year)
        net_income = self._extract_annual_values(facts, _NET_INCOME_CONCEPTS, min_year=min_year)
        eps = self._extract_eps_values(facts, _EPS_CONCEPTS, min_year=min_year)

        # Merge all dates
        all_dates = sorted(set(list(revenue.keys()) + list(net_income.keys()) + list(eps.keys())))

        records: List[Dict] = []
        for date in all_dates:
            rev = revenue.get(date)
            ni = net_income.get(date)
            ep = eps.get(date)

            # Only include rows that have at least one value
            if rev is None and ni is None and ep is None:
                continue

            records.append({
                "Symbol": symbol.upper(),
                "Date": date,
                "Revenue": rev,
                "Net_Income": ni,
                "EPS": ep,
            })

        logger.info("[%s] EDGAR returned %d annual records (CIK %s)", symbol, len(records), cik)
        return records

    def fetch_balance_sheet(self, symbol: str) -> Dict:
        """Not implemented — use FMP or Yahoo for balance sheet data."""
        raise DataProviderError("EdgarDataProvider does not support balance sheet fetching.")

"""NYSE + NASDAQ universe acquisition for the microcap screener.

Pulls the canonical exchange ticker lists from NASDAQ Trader:
  https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt
  https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt

These are pipe-delimited text files updated nightly. Together they cover
every common-stock ticker listed on NASDAQ, NYSE, NYSE American, and a few
smaller exchanges. We filter out:
  - test issues (Test Issue = Y)
  - ETFs (ETF = Y)
  - units, warrants, rights (Security Name patterns)
  - non-listed exchanges (BATS, ARCA — we keep only NASDAQ + NYSE + NYSE American)

Returns a deduplicated list of ticker symbols.
"""

from __future__ import annotations

import io
import logging
import re
from typing import Iterable

import pandas as pd
import requests

log = logging.getLogger("microcap_universe")

NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

# Other-listed Exchange codes worth keeping. NASDAQ Trader documents these.
# We skip: P (NYSE Arca — mostly ETFs), Z (BATS), V (IEX — sparse), etc.
ALLOWED_OTHER_EXCHANGES = {"N", "A"}  # N = NYSE, A = NYSE American

# Security-name patterns that indicate not-a-common-stock instruments we
# don't want to screen.
SKIP_NAME_PATTERNS = re.compile(
    r"(warrant|right(s)?\b|unit\b|preferred|depositary|notes? due|ETN |ETF |"
    r"acquisition right|trust certificate|when[- ]?issued)",
    re.IGNORECASE,
)

USER_AGENT = (
    "Mozilla/5.0 (compatible; StockFinancesScreener/1.0; +https://nicksstockfinancials.com)"
)


def _fetch(url: str) -> pd.DataFrame:
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    r.raise_for_status()
    # NASDAQ Trader files have a "File Creation Time" trailer line we drop.
    text = "\n".join(
        ln for ln in r.text.splitlines() if not ln.startswith("File Creation Time")
    )
    return pd.read_csv(io.StringIO(text), sep="|", dtype=str).fillna("")


def _passes_common_stock_filter(name: str) -> bool:
    return not SKIP_NAME_PATTERNS.search(name or "")


def fetch_universe() -> list[str]:
    """Return a deduplicated, filtered list of NYSE + NASDAQ common-stock tickers."""
    log.info("Pulling NASDAQ Trader ticker directories...")
    nasdaq = _fetch(NASDAQ_LISTED_URL)
    other = _fetch(OTHER_LISTED_URL)

    # nasdaqlisted columns: Symbol, Security Name, Market Category, Test Issue,
    #   Financial Status, Round Lot Size, ETF, NextShares
    nasdaq = nasdaq[
        (nasdaq.get("Test Issue", "N") == "N")
        & (nasdaq.get("ETF", "N") == "N")
        # Financial Status 'N' = Normal; anything else (D=deficient, E=delinquent,
        # Q=bankrupt, etc.) is risky territory we skip.
        & (nasdaq.get("Financial Status", "N") == "N")
    ]
    nasdaq = nasdaq[nasdaq["Security Name"].apply(_passes_common_stock_filter)]
    nasdaq_tickers = nasdaq["Symbol"].astype(str).str.strip().str.upper().tolist()

    # otherlisted columns: ACT Symbol, Security Name, Exchange, CQS Symbol,
    #   ETF, Round Lot Size, Test Issue, NASDAQ Symbol
    other = other[
        (other.get("Test Issue", "N") == "N")
        & (other.get("ETF", "N") == "N")
        & (other.get("Exchange", "").isin(ALLOWED_OTHER_EXCHANGES))
    ]
    other = other[other["Security Name"].apply(_passes_common_stock_filter)]
    other_tickers = other["ACT Symbol"].astype(str).str.strip().str.upper().tolist()

    universe = sorted(set(nasdaq_tickers + other_tickers))
    # Drop empties + obvious garbage tickers (those with '$' or '.' in them
    # are usually preferred shares or special classes we already filtered by
    # name, but belt-and-suspenders).
    universe = [
        t for t in universe
        if t and not any(ch in t for ch in ("$", "^"))
    ]
    log.info(
        f"Universe: {len(universe)} tickers "
        f"({len(nasdaq_tickers)} NASDAQ + {len(other_tickers)} NYSE/NYSE-AMEX)"
    )
    return universe


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    tickers = fetch_universe()
    print(f"\nFirst 20: {tickers[:20]}")
    print(f"Last 20:  {tickers[-20:]}")

#!/usr/bin/env python
"""
test.py – harvest, deduplicate, and extract expense categories
--------------------------------------------------------------
• Reads tickers from tickers.csv
• Pulls annual income statement data
• Extracts all raw row categories → all_income_statement_categories.csv
• Deduplicates → unique_income_statement_categories.csv
• Filters for expense categories → expense_categories.csv
"""

import os
import pandas as pd
import yfinance as yf

TICKER_FILE       = "tickers.csv"
RAW_OUTPUT_FILE   = "all_income_statement_categories.csv"
UNIQ_OUTPUT_FILE  = "unique_income_statement_categories.csv"
EXPENSE_OUTPUT_FILE = "expense_categories.csv"

EXPENSE_KEYWORDS = [
    "cost", "expense", "selling", "marketing", "administrative",
    "sg&a", "r&d", "research", "development"
]

def read_tickers(path: str) -> list[str]:
    """Return the first column of ``path`` as a list of ticker symbols.

    The previous implementation read the CSV twice – once to load an empty
    header and again to coerce the entire file into a headerless DataFrame.
    That extra I/O shows up when the ticker list grows into the hundreds.

    Pandas already understands the header row, so we just ask for the first
    column directly.  ``squeeze"`` keeps the code resilient to files with a
    single column, and ``dropna``/``astype`` mirror the original behaviour
    without the redundant read.
    """

    series = pd.read_csv(path, usecols=[0]).squeeze("columns")
    return (
        series.dropna()
              .astype(str)
              .str.strip()
              .loc[lambda s: s != ""]
              .tolist()
    )

def fetch_all_categories(tickers: list[str]) -> pd.DataFrame:
    """Collect raw category labels for each ticker.

    Building up a list of dicts inside the nested loop forced Python to append
    row-by-row, which becomes noticeably slow when ``annual.index`` contains
    hundreds of entries per ticker.  Instead we append lightweight DataFrames
    and let ``pandas.concat`` perform the heavy lifting in vectorised C code.
    ``unique`` trims duplicates within the same ticker so we don't emit the
    same category dozens of times only to drop them later.
    """

    frames = []
    for tkr in tickers:
        print(f"🔍 Fetching {tkr}")
        annual = yf.Ticker(tkr).income_stmt
        if isinstance(annual, pd.DataFrame) and not annual.empty:
            frames.append(pd.DataFrame({"category": annual.index.unique()}))

    if not frames:
        return pd.DataFrame(columns=["category"])

    return pd.concat(frames, ignore_index=True)

def is_expense(category: str) -> bool:
    category_lower = category.lower()
    return any(keyword in category_lower for keyword in EXPENSE_KEYWORDS)

def main():
    if not os.path.exists(TICKER_FILE):
        raise FileNotFoundError(f"{TICKER_FILE} not found")
    tickers = read_tickers(TICKER_FILE)
    if not tickers:
        raise ValueError("Ticker file is empty")

    # Step 1: Raw data dump
    df_raw = fetch_all_categories(tickers)
    df_raw.to_csv(RAW_OUTPUT_FILE, index=False)
    print(f"✅ Saved raw categories → {RAW_OUTPUT_FILE}")

    # Step 2: Deduplicate + sort
    unique = df_raw['category'].dropna().drop_duplicates().sort_values()
    unique.to_frame().to_csv(UNIQ_OUTPUT_FILE, index=False)
    print(f"✅ Saved unique categories → {UNIQ_OUTPUT_FILE}")

    # Step 3: Filter expenses
    expense_only = [cat for cat in unique if is_expense(cat)]
    pd.DataFrame({"expense_category": expense_only}).to_csv(EXPENSE_OUTPUT_FILE, index=False)
    print(f"✅ Saved expense categories → {EXPENSE_OUTPUT_FILE}")

if __name__ == "__main__":
    main()

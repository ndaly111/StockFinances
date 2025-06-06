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
    df = pd.read_csv(path, nrows=0)
    df = pd.read_csv(path, header=None)
    return df.iloc[:, 0].dropna().astype(str).tolist()

def fetch_all_categories(tickers: list[str]) -> pd.DataFrame:
    records = []
    for tkr in tickers:
        print(f"🔍 Fetching {tkr}")
        yf_tkr = yf.Ticker(tkr)
        annual = yf_tkr.income_stmt
        if isinstance(annual, pd.DataFrame) and not annual.empty:
            for cat in annual.index:
                records.append({"category": cat})
    return pd.DataFrame(records)

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

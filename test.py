#!/usr/bin/env python
"""
test.py – harvest all annual income-statement categories from Yahoo Finance
--------------------------------------------------------------------------
• Reads tickers from tickers.csv (first column)
• For each ticker:
    – pulls annual income_stmt
    – extracts every row-label (“category”) exactly as received
• Appends everything into one column (duplicates allowed)

Output: all_income_statement_categories.csv  with column:
        category
"""

import os
import pandas as pd
import yfinance as yf

TICKER_FILE = "tickers.csv"
OUTPUT_FILE = "all_income_statement_categories.csv"


def read_tickers(path: str) -> list[str]:
    df = pd.read_csv(path, nrows=0)  # check exists
    df = pd.read_csv(path, header=None)
    return df.iloc[:, 0].dropna().astype(str).tolist()


def collect_annual_categories(tickers: list[str]) -> pd.DataFrame:
    records = []

    for tkr in tickers:
        print(f"🔍 Fetching {tkr}")
        yf_tkr = yf.Ticker(tkr)

        annual = yf_tkr.income_stmt
        if isinstance(annual, pd.DataFrame) and not annual.empty:
            for cat in annual.index:
                records.append({"category": cat})

    return pd.DataFrame(records)


def main():
    if not os.path.exists(TICKER_FILE):
        raise FileNotFoundError(f"{TICKER_FILE} not found")
    tickers = read_tickers(TICKER_FILE)
    if not tickers:
        raise ValueError("Ticker file is empty")

    df_all = collect_annual_categories(tickers)
    df_all.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"\n✅ Saved category list → {OUTPUT_FILE}")
    print(f"Rows written: {len(df_all):,}")


if __name__ == "__main__":
    main()

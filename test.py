#!/usr/bin/env python
"""
test.py – harvest every raw Yahoo-Finance income-statement category
------------------------------------------------------------------
• Reads tickers from tickers.csv (first column)
• For each ticker:
    – pulls annual  income_stmt
    – pulls quarterly_income_stmt
    – writes every row-label (“category”) exactly as received
• Appends everything into one CSV (duplicates allowed)

Output: all_income_statement_categories.csv  with columns
        ticker | statement_type | period_end | category
"""

import os
import pandas as pd
import yfinance as yf

TICKER_FILE   = "tickers.csv"                         # input list
OUTPUT_FILE   = "all_income_statement_categories.csv" # master output


def read_tickers(path: str) -> list[str]:
    """Return clean list of tickers from first column of CSV."""
    df = pd.read_csv(path, nrows=0)  # just to validate file exists
    df = pd.read_csv(path, header=None)  # read raw
    return df.iloc[:, 0].dropna().astype(str).tolist()


def collect_raw_categories(tickers: list[str]) -> pd.DataFrame:
    """Loop through tickers and return long-form DataFrame of categories."""
    records: list[dict] = []

    for tkr in tickers:
        print(f"🔍 Fetching {tkr}")
        yf_tkr = yf.Ticker(tkr)

        # ---- annual -------------------------------------------------------
        annual = yf_tkr.income_stmt
        if isinstance(annual, pd.DataFrame) and not annual.empty:
            for period_end in annual.columns:         # each fiscal year
                for cat in annual.index:
                    records.append({
                        "ticker": tkr,
                        "statement_type": "annual",
                        "period_end": str(period_end),
                        "category": cat
                    })

        # ---- quarterly ----------------------------------------------------
        qtr = yf_tkr.quarterly_income_stmt
        if isinstance(qtr, pd.DataFrame) and not qtr.empty:
            for period_end in qtr.columns:            # each fiscal quarter
                for cat in qtr.index:
                    records.append({
                        "ticker": tkr,
                        "statement_type": "quarterly",
                        "period_end": str(period_end),
                        "category": cat
                    })

    return pd.DataFrame(records)


def main():
    if not os.path.exists(TICKER_FILE):
        raise FileNotFoundError(f"{TICKER_FILE} not found")
    tickers = read_tickers(TICKER_FILE)
    if not tickers:
        raise ValueError("Ticker file is empty")

    df_all = collect_raw_categories(tickers)
    df_all.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"\n✅ Saved full category list → {OUTPUT_FILE}")
    print(f"Rows written: {len(df_all):,}")


if __name__ == "__main__":
    main()

import os
import yfinance as yf
import pandas as pd

TICKER_FILE = "tickers.csv"
OUTPUT_FILE = "income_statement_categories.txt"

def read_tickers(file_path):
    df = pd.read_csv(file_path)
    tickers = df.iloc[:, 0].dropna().astype(str).tolist()
    return tickers

def collect_income_statement_fields(tickers):
    fields = set()

    for ticker in tickers:
        print(f"🔍 Fetching financials for {ticker}")
        try:
            tkr = yf.Ticker(ticker)

            # Annual income statement
            annual = tkr.financials
            if not annual.empty:
                fields.update(annual.index.tolist())

            # Quarterly income statement
            quarterly = tkr.quarterly_financials
            if not quarterly.empty:
                fields.update(quarterly.index.tolist())

        except Exception as e:
            print(f"⚠️ Error with {ticker}: {e}")

    return sorted(fields)

def save_fields_to_file(fields, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for field in fields:
            f.write(field + "\n")
    print(f"✅ Saved income statement categories to → {output_path}")

def main():
    tickers = read_tickers(TICKER_FILE)
    fields = collect_income_statement_fields(tickers)
    save_fields_to_file(fields, OUTPUT_FILE)

if __name__ == "__main__":
    main()

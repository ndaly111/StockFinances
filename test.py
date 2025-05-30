import os
import yfinance as yf
from ticker_manager import read_tickers, modify_tickers

OUTPUT_FILE = "income_statement_categories.txt"

def collect_income_statement_fields(tickers):
    fields = set()

    for ticker in tickers:
        print(f"🔍 Fetching financials for {ticker}")
        try:
            tkr = yf.Ticker(ticker)

            # Annual financials
            annual = tkr.financials
            if not annual.empty:
                fields.update(annual.index.tolist())

            # Quarterly financials
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
    tickers = modify_tickers(read_tickers("tickers.csv"))
    fields = collect_income_statement_fields(tickers)
    save_fields_to_file(fields, OUTPUT_FILE)

if __name__ == "__main__":
    main()

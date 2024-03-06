#start of main
import os
import datetime
import sqlite3
from data_fetcher import (fetch_ticker_data, get_latest_annual_data_date, determine_if_annual_data_missing,calculate_next_annual_check_date_from_data, check_null_fields_annual, fetch_annual_data_from_yahoo,store_annual_data,fetch_ttm_data,check_null_fields_ttm, is_ttm_data_outdated,is_ttm_data_blank,fetch_ttm_data_from_yahoo,store_ttm_data,prompt_and_update_partial_entries)
import ticker_manager
from chart_generator import (prepare_data_for_charts, generate_financial_charts)
import yfinance as yf
from html_generator import (create_html_for_tickers)
from html_to_pdf_converter import html_to_pdf





# Constants
TICKERS_FILE_PATH = 'tickers.csv'
DB_PATH = 'Stock Data.db'
CHARTS_OUTPUT_DIR = 'charts/'
HTML_OUTPUT_FILE = 'financial_charts.html'
PDF_OUTPUT_FILE = '/Users/nicholasdaly/Library/Mobile Documents/com~apple~CloudDocs/Stock Data/stock_charts.pdf'
is_remote = True
debug_this = False


def check_database_exists(db_path):
    full_path = os.path.abspath(db_path)
    if not os.path.exists(full_path):
        print(f"Database file not found: {full_path}")
        return False
    return True


def main_remote():
    print("main start")
    # Initialize the dictionary to store financial data for each ticker
    financial_data = {}
    print("main 1 financial data", financial_data)

    current_tickers = ticker_manager.read_tickers(TICKERS_FILE_PATH)
    current_tickers = ticker_manager.modify_tickers(current_tickers, is_remote=True)
    sorted_tickers = sorted(current_tickers)  # Ensure tickers are sorted
    ticker_manager.write_tickers(sorted_tickers, TICKERS_FILE_PATH)
    print("main 2 current tickers, sorted tickers, ticker manager")

    db_full_path = os.path.abspath(DB_PATH)
    if not os.path.exists(db_full_path):
        print(f"Database file not found: {db_full_path}")
        return
    print("main 3 check if database exists")

    with sqlite3.connect(db_full_path) as conn:
        cursor = conn.cursor()
        for ticker in sorted_tickers:
            print(f"Processing ticker: {ticker}")
            print("main 4 starting main ticker loop")

            ticker_data = fetch_ticker_data(ticker, cursor)
            print("ticker data", ticker_data)
            missing_data = determine_if_annual_data_missing(ticker_data)

            print("missing data", missing_data)
            check_annual_data = calculate_next_annual_check_date_from_data(ticker_data)
            print("check annual data?", check_annual_data)
            null_data_present = check_null_fields_annual(ticker,ticker_data)
            print("null data?", null_data_present)

            if missing_data or check_annual_data or null_data_present or debug_this:
                print("One or more conditions met: Missing data, time to check for new data, or null fields present.")

                # Fetch new annual data from Yahoo Finance
                annual_data = fetch_annual_data_from_yahoo(ticker)

                # Check if annual_data is not empty before proceeding
                if not annual_data.empty:
                    annual_update = True
                    print("---m updated annual data")

                    # The 'Date' column is assumed to be in the correct format
                    # If it's in datetime format, convert to string as needed
                    if isinstance(annual_data.index, pd.DatetimeIndex):
                        annual_data['Date'] = annual_data.index.date.astype(str)
                        # Store the data
                        store_annual_data(ticker, annual_data, cursor)
                    else:
                        print("---m Checked but, no annual data found in yahoo")


            else:
                print("Data is up-to-date and complete. Continuing with the script...")


            print("---m starting TTM data update process")
            ttm_data = fetch_ttm_data(ticker,cursor)
            print("---M fetching ttm data")

            ttm_null = check_null_fields_ttm(ttm_data)
            print("---M checking if ttm data has null fields",ttm_null)

            ttm_blank = is_ttm_data_blank(ttm_data)
            print("---M checking if ttm data is blank",ttm_blank)

            ttm_outdated = is_ttm_data_outdated(ttm_data)
            print("---M checking if ttm data is outdated",ttm_outdated)

            print("---M checking if we should fetch ttm data")
            if ttm_blank or ttm_null or ttm_outdated or debug_this:
                print("One or more conditions met: Missing data, null fields present, or data is outdated.")

                # Fetch new TTM data from Yahoo Finance
                ttm_data_from_yahoo = fetch_ttm_data_from_yahoo(ticker)

                # Check if ttm_data_from_yahoo is not None before proceeding
                if ttm_data_from_yahoo:
                    ttm_update = True
                    # Store the data
                    ttm_data = store_ttm_data(ticker, ttm_data_from_yahoo, cursor)
                else:
                    print(f"No TTM data found for {ticker} from Yahoo Finance.")
                    ttm_update = False
            else:
                print("TTM data is up-to-date and complete. Continuing with the script...")

            ticker_data_df = pd.DataFrame(ticker_data)
            ttm_data_df = pd.DataFrame([ttm_data])  # Assuming ttm_data is a single dict

            combined_data = pd.concat([ticker_data_df, ttm_data_df], ignore_index=True)

            # These lines should be inside the loop as they are part of the processing for each ticker
            prompt_and_update_partial_entries(ticker, cursor, is_remote=False)
            ticker_financial_data = prepare_data_for_charts(ticker, cursor)
            if not ticker_financial_data.empty:
                financial_data[ticker] = ticker_financial_data
                generate_financial_charts(ticker, charts_output_dir, financial_data[ticker])

        # This line should be outside the loop, to execute after all tickers have been processed
        if financial_data:
            create_html_for_tickers(sorted_tickers, financial_data, charts_output_dir)
            html_full_path = os.path.join(HTML_OUTPUT_FILE)
            if os.path.exists(html_full_path):
                html_to_pdf(html_full_path, PDF_OUTPUT_FILE)
            else:
                print(f"HTML file not found: {html_full_path}")
        else:
            print("No financial data collected, skipping HTML and PDF generation.")


if __name__ == "__main__":
    main_remote()

    #end of main.py
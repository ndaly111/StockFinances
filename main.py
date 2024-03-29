#start of main
import os
import sqlite3
import ticker_manager
from data_fetcher import (fetch_ticker_data, determine_if_annual_data_missing,calculate_next_annual_check_date_from_data, check_null_fields_annual, fetch_annual_data_from_yahoo,store_annual_data,fetch_ttm_data,check_null_fields_ttm, is_ttm_data_outdated,is_ttm_data_blank,fetch_ttm_data_from_yahoo,store_ttm_data,prompt_and_update_partial_entries)
from chart_generator import (prepare_data_for_charts, generate_financial_charts)
from html_generator import (create_html_for_tickers)
from html_to_pdf_converter import html_to_pdf
from balance_sheet_data_fetcher import (
    fetch_balance_sheet_data,
    check_missing_balance_sheet_data,
    is_balance_sheet_data_outdated,
    fetch_balance_sheet_data_from_yahoo,
    store_fetched_balance_sheet_data
)
from balance_sheet_data_fetcher import balance_sheet_data_fetcher
from balancesheet_chart import (fetch_balance_sheet_data,plot_chart,format_value, create_and_save_table)
import pandas as pd





# Constants
TICKERS_FILE_PATH = 'tickers.csv'
DB_PATH = 'Stock Data.db'
charts_output_dir = 'charts/'
HTML_OUTPUT_FILE = 'financial_charts.html'
PDF_OUTPUT_FILE = '/Users/nicholasdaly/Library/Mobile Documents/com~apple~CloudDocs/Stock Data/stock_charts.pdf'
is_remote = False
print("constants")

debug_this = False


def manage_tickers(TICKERS_FILE_PATH, is_remote=False):
    print("main 1 manage tickers")
    current_tickers = ticker_manager.read_tickers(TICKERS_FILE_PATH)
    print("Ticker Manager: Read current tickers.")

    # Modify the tickers as needed (this may involve user input or other logic)
    current_tickers = ticker_manager.modify_tickers(current_tickers, is_remote)
    print("Ticker Manager: Modified tickers.")

    # Sort the tickers alphabetically
    sorted_tickers = sorted(current_tickers)
    print("Ticker Manager: Sorted tickers.")

    # Write the potentially modified tickers back to the file
    ticker_manager.write_tickers(sorted_tickers, TICKERS_FILE_PATH)
    print("Ticker Manager: Wrote tickers to file.")

    return sorted_tickers


def establish_database_connection(db_path):
    print("main 2 establish database connection")
    db_full_path = os.path.abspath(db_path)

    if not os.path.exists(db_full_path):
        print(f"Database file not found: {db_full_path}")
        return None

    print("Database exists. Establishing connection...")
    return sqlite3.connect(db_full_path)


def fetch_financial_data(ticker, cursor):
    print("main 3 fetch financial data")
    print(f"Fetching financial data for ticker: {ticker}")

    # Fetch existing data
    ticker_data = fetch_ticker_data(ticker, cursor)
    print("Ticker data:", ticker_data)

    # Determine if annual data is missing
    missing_data = determine_if_annual_data_missing(ticker_data)
    print("Missing annual data:", missing_data)

    # Calculate next annual check date based on the latest data
    check_annual_data = calculate_next_annual_check_date_from_data(ticker_data)
    print("Check annual data:", check_annual_data)

    # Check for null fields in annual data
    null_data_present = check_null_fields_annual(ticker, ticker_data)
    print("Null data present:", null_data_present)

    # If data is missing or needs checking, fetch new data from Yahoo Finance
    if missing_data or check_annual_data or null_data_present:
        annual_data = fetch_annual_data_from_yahoo(ticker)
        if annual_data is not None:
            store_annual_data(ticker, annual_data, cursor)
            print("Stored new annual data for ticker:", ticker)

    # Fetch TTM data and check for completeness and freshness
    ttm_data = fetch_ttm_data(ticker, cursor)
    ttm_null = check_null_fields_ttm(ttm_data)
    ttm_blank = is_ttm_data_blank(ttm_data)
    ttm_outdated = is_ttm_data_outdated(ttm_data)

    if ttm_blank or ttm_null or ttm_outdated:
        ttm_data_from_yahoo = fetch_ttm_data_from_yahoo(ticker)
        if ttm_data_from_yahoo:
            store_ttm_data(ticker, ttm_data_from_yahoo, cursor)
            print("Stored new TTM data for ticker:", ticker)

    # Combine the annual and TTM data into one DataFrame
    ticker_data_df = pd.DataFrame(ticker_data)
    ttm_data_df = pd.DataFrame([ttm_data])  # Assuming ttm_data is a single dictionary
    combined_data = pd.concat([ticker_data_df, ttm_data_df], ignore_index=True)
    prompt_and_update_partial_entries(ticker,cursor, is_remote=False)

    return combined_data



def balancesheet_chart(ticker, charts_output_dir):
    print("main 4 balancesheet chart")
    balance_sheet_data = fetch_balance_sheet_data(ticker)
    if balance_sheet_data:
        plot_chart(balance_sheet_data, charts_output_dir, ticker)
        # Calculate debt to equity ratio
        balance_sheet_data['Debt_to_Equity_Ratio'] = balance_sheet_data['Total_Debt'] / balance_sheet_data[
            'Total_Equity']

        # Call function to create and save the table
        create_and_save_table(balance_sheet_data, charts_output_dir,
                              ticker)  # Assuming this function also needs 'ticker'
    else:
        print(f"No balance sheet data found for {ticker}.")

def fetch_and_update_balance_sheet_data(ticker, cursor):
    print("main 5 fetch and update balance sheet data")
    print(f"Fetching balance sheet data for ticker: {ticker}")

    # Step 1: Fetch current balance sheet data
    current_data = fetch_balance_sheet_data(ticker)
    print("---Current balance sheet data:", current_data)

    # Step 2: Check for missing or null data
    data_missing_or_null = check_missing_balance_sheet_data(ticker, cursor)
    print("---Missing or null data?", data_missing_or_null)

    # Step 3: Check if data is outdated
    data_outdated = is_balance_sheet_data_outdated(current_data)
    print("---Outdated balance sheet?", data_outdated)

    # If any data is missing, null, or outdated, fetch from Yahoo and update the database
    if data_missing_or_null or data_outdated:
        balance_sheet_data = fetch_balance_sheet_data_from_yahoo(ticker)
        print("---Missing or null balance sheet data:", balance_sheet_data)

        if balance_sheet_data:
            store_fetched_balance_sheet_data(cursor, balance_sheet_data)
            print("---Stored fetched balance sheet data for", ticker)
        else:
            print(f"---No balance sheet data fetched from Yahoo for {ticker}.")



def generate_charts(ticker, cursor, output_dir, financial_data):
    print("main 6 generate charts")
    """
    Generates charts for the given ticker using its financial data.

    :param ticker: The ticker symbol for which to generate charts.
    :param cursor: The database cursor to fetch data (not used in this modified version).
    :param output_dir: The directory where charts will be saved.
    :param financial_data: The financial data for the given ticker.
    """
    if not financial_data.empty:
        print(f"Generating charts for {ticker}...")
        generate_financial_charts(ticker, output_dir, financial_data)
        print(f"Charts generated for {ticker} and saved to {output_dir}")
    else:
        print(f"No data available to generate charts for {ticker}.")


def generate_html_report(sorted_tickers, financial_data, output_dir, output_file):
    print("main 7 generate html")
    """
    Generates an HTML report for the financial data of tickers.

    :param sorted_tickers: The list of ticker symbols processed.
    :param financial_data: A dictionary containing financial data for each ticker.
    :param output_dir: The directory where the HTML report will be saved.
    :param output_file: The name of the HTML file to be generated.
    """
    # Assume `create_html_for_tickers` is implemented to generate the HTML content
    html_content = create_html_for_tickers(sorted_tickers, financial_data, output_dir)

    # Ensure html_content is not None
    if html_content is None:
        raise ValueError("No HTML content to write. The create_html_for_tickers function returned None.")

    # Write the HTML content to a file
    html_full_path = os.path.join(output_dir, output_file)
    with open(html_full_path, 'w') as file:
        file.write(html_content)

    print(f"HTML report generated at {html_full_path}")
    return html_full_path




def main():
    print("main start")
    financial_data = {}

    # Manage tickers and establish database connection
    sorted_tickers = manage_tickers(TICKERS_FILE_PATH, is_remote=False)
    print("---main loop 1 sorted tickers")

    conn = establish_database_connection(DB_PATH)
    if conn is None:
        return

    try:
        cursor = conn.cursor()
        for ticker in sorted_tickers:
            print("main loop start")
            print(f"Processing ticker: {ticker}")

            # Existing data fetching and processing
            combined_data = fetch_financial_data(ticker, cursor)
            print("---m combined data")

            # Fetch and update balance sheet data
            fetch_and_update_balance_sheet_data(ticker, cursor)
            print("---m fetch and update balance sheet data")

            # Generate balance sheet chart and table
            balancesheet_chart(ticker, charts_output_dir)
            print("---m generate balance sheet chart and table")

            print("---m define ticker financial data")
            ticker_financial_data = prepare_data_for_charts(ticker, cursor)
            print(type(ticker_financial_data))  # Should print <class 'pandas.core.frame.DataFrame'>
            print(ticker_financial_data.empty)  # Should print False if there is data

            if not ticker_financial_data.empty:
                print("Financial data has data, generating charts")
                financial_data[ticker] = ticker_financial_data
                generate_financial_charts(ticker, charts_output_dir, financial_data[ticker])
            else:
                print(f"No data available to generate charts for {ticker}.")
            # Generate HTML report after all tickers have been processed
            # Your existing chart generation code
            print("---m chart generation code")
            generate_charts(ticker, cursor, charts_output_dir, combined_data)



        # Generate HTML report after processing all tickers
        html_full_path = generate_html_report(sorted_tickers, financial_data, '.', HTML_OUTPUT_FILE)
        print("---m generating html report")

        # Convert HTML to PDF outside the ticker loop
        if os.path.exists(html_full_path):
            html_to_pdf(html_full_path, PDF_OUTPUT_FILE)
            print(f"PDF report generated at {PDF_OUTPUT_FILE}")

    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    main()

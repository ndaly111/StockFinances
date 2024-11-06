#start of main
import os
import sqlite3
import ticker_manager
from datetime import datetime
from annual_and_ttm_update import annual_and_ttm_update
from html_generator import (create_html_for_tickers)
from balance_sheet_data_fetcher import (
    fetch_balance_sheet_data,
    check_missing_balance_sheet_data,
    is_balance_sheet_data_outdated,
    fetch_balance_sheet_data_from_yahoo,
    store_fetched_balance_sheet_data
)
from balancesheet_chart import (fetch_balance_sheet_data,plot_chart,format_value, create_and_save_table)
import pandas as pd
from Forward_data import (scrape_forward_data)
from forecasted_earnings_chart import generate_forecast_charts_and_tables
from bs4 import BeautifulSoup
from ticker_info import (prepare_data_for_display,generate_html_table)
import requests
from html_generator2 import html_generator2, generate_dashboard_table
from valuation_update import (valuation_update, process_update_growth_csv)


# Constants
TICKERS_FILE_PATH = 'tickers.csv'
db_path = 'Stock Data.db'
DB_PATH = 'Stock Data.db'
file_path = "update_growth.csv"
charts_output_dir = 'charts/'
HTML_OUTPUT_FILE = 'index.html'
PDF_OUTPUT_FILE = '/Users/nicholasdaly/Library/Mobile Documents/com~apple~CloudDocs/Stock Data/stock_charts.pdf'
is_remote = True
historical_table_name = 'Annual_Data'
forecast_table_name = 'ForwardFinancialData'
print("constants")

debug_this = False
table_name = 'ForwardFinancialData'


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




def log_average_valuations(avg_values, TICKERS_FILE_PATH):
    if TICKERS_FILE_PATH != 'tickers.csv':
        print("Skipping average valuation update, as TICKERS_FILE_PATH is not 'tickers.csv'.")
        return

    avg_ttm_valuation = avg_values['Nicks_TTM_Value_Average']
    avg_forward_valuation = avg_values['Nicks_Forward_Value_Average']
    avg_finviz_valuation = avg_values['Finviz_TTM_Value_Average']

    current_date = datetime.now().strftime('%Y-%m-%d')

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # Create the table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS AverageValuations (
                date DATE PRIMARY KEY,
                avg_ttm_valuation REAL,
                avg_forward_valuation REAL,
                avg_finviz_valuation REAL
            );
        ''')

        # Check if a record already exists for the current date
        cursor.execute('''
            SELECT 1 FROM AverageValuations WHERE date = ?;
        ''', (current_date,))
        if cursor.fetchone():
            print(f"Average valuations for {current_date} already recorded. Skipping.")
        else:
            # Insert the new average values
            cursor.execute('''
                INSERT INTO AverageValuations (date, avg_ttm_valuation, avg_forward_valuation, avg_finviz_valuation)
                VALUES (?, ?, ?, ?);
            ''', (current_date, avg_ttm_valuation, avg_forward_valuation, avg_finviz_valuation))
            conn.commit()
            print(f"Inserted average valuations for {current_date} into AverageValuations.")


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







def fetch_10_year_treasury_yield():
    """
    Fetches the latest 10-year Treasury note yield from the FRED website.

    Returns:
        str: The latest 10-year Treasury note yield as a string, or 'N/A' if not found.
    """
    url = "https://fred.stlouisfed.org/series/GS10"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        soup = BeautifulSoup(response.content, 'html.parser')
        yield_value = soup.find("span", class_="series-meta-observation-value")
        if yield_value:
            return yield_value.text.strip()
        else:
            return "N/A"
    except requests.RequestException as e:
        print(f"Error fetching the 10-year Treasury note yield: {e}")
        return "N/A"

def main():
    print("main start")
    financial_data = {}
    dashboard_data = []
    treasury_yield = fetch_10_year_treasury_yield()

    # Manage tickers and establish database connection
    sorted_tickers = manage_tickers(TICKERS_FILE_PATH, is_remote=True)
    print("---main loop 1 sorted tickers")

    conn = establish_database_connection(DB_PATH)
    if conn is None:
        return

    try:
        cursor = conn.cursor()
        print("cursor", cursor)
        process_update_growth_csv(file_path, db_path)
        for ticker in sorted_tickers:
            print("main loop start")
            print(f"Processing ticker: {ticker}")

            # Existing data fetching and processing
            annual_and_ttm_update(ticker, db_path)

            # Fetch and update balance sheet data
            fetch_and_update_balance_sheet_data(ticker, cursor)
            print("---m fetch and update balance sheet data")

            # Generate balance sheet chart and table
            balancesheet_chart(ticker, charts_output_dir)
            print("---m generate balance sheet chart and table")

            scrape_forward_data(ticker, db_path, table_name)

            # Generate HTML report after all tickers have been processed
            generate_forecast_charts_and_tables(ticker, db_path, charts_output_dir)

            prepared_data, marketcap = prepare_data_for_display(ticker, treasury_yield)

            generate_html_table(prepared_data, ticker)

            valuation_update(ticker, cursor, treasury_yield, marketcap, dashboard_data)

        # Generate the dashboard table HTML
        full_dashboard_html, avg_values = generate_dashboard_table(dashboard_data)

        # Log average valuations to the database
        log_average_valuations(avg_values, TICKERS_FILE_PATH)

        print("generating HTML2")
        # Call html_generator2 function after all tickers have been processed
        html_generator2(sorted_tickers, financial_data, full_dashboard_html, avg_values)

    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    main()

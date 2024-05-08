import sqlite3
import requests
from lxml import html
import os

# Constants
DB_PATH = 'Stock Data.db'  # Path to your SQLite database
TABLE_NAME = 'Tickers_Info'  # Table name in your database
FINVIZ_URL_TEMPLATE = "https://finviz.com/quote.ashx?t={ticker}"

# Establish database connection
def establish_database_connection(db_path):
    db_full_path = os.path.abspath(db_path)
    return sqlite3.connect(db_full_path)

# Function to fetch the Finviz 5-year estimates
def fetch_finviz_estimates(ticker):
    try:
        # Construct the Finviz URL for the given ticker
        url = FINVIZ_URL_TEMPLATE.format(ticker=ticker)

        # Fetch the page content
        response = requests.get(url)
        response.raise_for_status()  # Ensure the request was successful

        # Parse the HTML content using lxml
        tree = html.fromstring(response.content)

        # Define the XPath to the 5-year estimate data
        xpath = '/html/body/div[3]/div[3]/div[4]/table/tbody/tr/td/div/table[1]/tbody/tr/td/div[2]/table/tbody/tr[6]/td[6]/b'

        # Extract the 5-year estimate using the XPath
        five_year_estimate = tree.xpath(xpath)
        if five_year_estimate:
            estimate_value = five_year_estimate[0].text_content().strip()
            print(f"5-Year Estimate for {ticker}: {estimate_value}")
            return estimate_value
        else:
            print(f"No 5-year estimate data found for {ticker} on Finviz.")
            return None

    except requests.RequestException as e:
        print(f"Error fetching Finviz data for {ticker}: {e}")
        return None

# Function to update the 5-year estimates in the database
def valuation_update(tickers, cursor, table_name='Tickers_Info'):
    """
    Fetches the Finviz 5-year estimates for the provided tickers and stores them in the database.

    :param tickers: List of ticker symbols to fetch and update valuations for.
    :param cursor: Database cursor to execute SQL commands.
    :param table_name: The name of the table in the database where valuations will be stored.
    """
    for ticker in tickers:
        try:
            # Fetch the 5-year estimate from Finviz
            estimate = fetch_finviz_estimates(ticker)

            # Convert the estimate to a numeric value and update the database
            if estimate is not None:
                try:
                    # Remove the percentage sign and convert to a float (as a decimal value)
                    estimate_value = float(estimate.strip('%')) / 100

                    # Update the database with the fetched estimate
                    cursor.execute(f'''
                    UPDATE {table_name} 
                    SET FINVIZ_5yr_gwth = ? 
                    WHERE ticker = ?;
                    ''', (estimate_value, ticker))

                    print(f"Stored Finviz 5-Year Growth for {ticker}: {estimate_value}")
                except ValueError as e:
                    print(f"Unable to convert the estimate for {ticker} to a numeric value: {e}")

        except Exception as e:
            print(f"Error fetching Finviz data for {ticker}: {e}")

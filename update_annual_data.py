# update_annual_data.py
import sqlite3
import pandas as pd
import yfinance as yf

DB_PATH = 'Stock Data.db'  # Update this with the path to your database
TICKERS_FILE_PATH = 'tickers.csv'

def read_tickers_from_csv(file_path):
    """Reads ticker symbols from a CSV file."""
    return pd.read_csv(file_path, header=None).iloc[:, 0].tolist()


def fetch_annual_data_from_yahoo(ticker):
    """Fetches annual financial data from Yahoo Finance for a given ticker."""
    stock = yf.Ticker(ticker)
    financials = stock.financials.T
    if financials.empty:
        print(f"No financials data for {ticker}")
        return pd.DataFrame()

    financials['Date'] = financials.index.strftime('%Y-%m-%d')
    financials['Symbol'] = ticker

    # Display the columns to help us debug if there's an error
    print(f"Columns available for {ticker}: {list(financials.columns)}")

    # Return only the columns we need, handling the case where columns might not exist
    columns_needed = []
    columns_to_rename = {}

    # Check each required column and only add if it exists
    if 'Total Revenue' in financials.columns:
        columns_needed.append('Total Revenue')
        columns_to_rename['Total Revenue'] = 'Revenue'
    else:
        print(f"'Total Revenue' not found for ticker: {ticker}")

    if 'Net Income' in financials.columns:
        columns_needed.append('Net Income')
        columns_to_rename['Net Income'] = 'Net_Income'
    else:
        print(f"'Net Income' not found for ticker: {ticker}")

    if 'Basic EPS' in financials.columns:
        columns_needed.append('Basic EPS')
        columns_to_rename['Basic EPS'] = 'EPS'
    else:
        print(f"'Basic EPS' not found for ticker: {ticker}")

    # If no financial columns found, return empty DataFrame
    if not columns_needed:
        print(f"No required financial columns found for {ticker}")
        return pd.DataFrame()

    try:
        return financials[['Symbol', 'Date'] + columns_needed].rename(columns=columns_to_rename)
    except KeyError as e:
        print(f"KeyError: {e}")
        return pd.DataFrame()


def update_database_with_fetched_data(cursor, financial_data):
    """Updates the database with the fetched financial data if there are differences."""
    for _, row in financial_data.iterrows():
        # Query the current data from the database for this ticker and date
        cursor.execute("""
            SELECT Revenue, Net_Income, EPS FROM Annual_Data
            WHERE Symbol = ? AND Date = ?
        """, (row['Symbol'], row['Date']))
        current_data = cursor.fetchone()

        # If 'EPS' is not present in the DataFrame, it defaults to None
        eps_value = row['EPS'] if 'EPS' in financial_data.columns else None

        # Determine if an update is needed
        update_needed = (
            not current_data or  # No existing data for this date
            current_data[0] != row['Revenue'] or
            current_data[1] != row['Net_Income'] or
            (current_data[2] or 0) != (eps_value or 0)  # Handle None and 0 equivalency
        )

        if update_needed:
            # Update the data in the database for this ticker and date
            cursor.execute("""
                INSERT INTO Annual_Data (Symbol, Date, Revenue, Net_Income, EPS, Last_Updated)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(Symbol, Date) DO UPDATE SET
                Revenue = excluded.Revenue,
                Net_Income = excluded.Net_Income,
                EPS = excluded.EPS,
                Last_Updated = CURRENT_TIMESTAMP;
            """, (row['Symbol'], row['Date'], row['Revenue'], row['Net_Income'], eps_value))
            print(f"Updated data for {row['Symbol']} on {row['Date']}.")
        else:
            print(f"No update needed for {row['Symbol']} on {row['Date']}.")
    cursor.connection.commit()

def update_short_name_if_needed(cursor, ticker):
    """Updates the short_name field in Tickers_Info table if it is null or blank."""
    cursor.execute("""
        SELECT short_name FROM Tickers_Info WHERE ticker = ?
    """, (ticker,))
    result = cursor.fetchone()

    if result and (result[0] is None or result[0].strip() == ''):
        stock = yf.Ticker(ticker)
        short_name = stock.info.get('shortName')
        if short_name:
            cursor.execute("""
                UPDATE Tickers_Info
                SET short_name = ?
                WHERE ticker = ?;
            """, (short_name, ticker))
            cursor.connection.commit()
            print(f"Updated short name for {ticker}: {short_name}")
        else:
            print(f"No short name found for {ticker} in Yahoo Finance.")


def main():
    # Establish database connection
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Read tickers from CSV
    tickers = read_tickers_from_csv(TICKERS_FILE_PATH)

    # Loop through each ticker, fetch data, and update the database
    for ticker in tickers:
        print(f"Processing {ticker}...")

        # Update short name if needed
        update_short_name_if_needed(cursor, ticker)

        # Fetch and update annual financial data
        financial_data = fetch_annual_data_from_yahoo(ticker)
        if not financial_data.empty:
            update_database_with_fetched_data(cursor, financial_data)
            print(f"Updated database with annual data for {ticker}.")
        else:
            print(f"No financial data found for {ticker}.")

    # Close the database connection
    conn.close()
    print("All tickers processed.")


if __name__ == "__main__":
    main()
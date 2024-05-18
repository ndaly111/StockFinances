#start of blaancesheet_data_fetcher


import yfinance as yf
import pandas as pd
import sqlite3
from datetime import datetime, timedelta


def fetch_balance_sheet_data(ticker, cursor):
    print("balance sheet data fetcher 1 fetch balance sheet data from db")
    print(f"Fetching balance sheet data for {ticker}")
    try:
        # Specify columns explicitly to avoid order-based mapping issues
        query = """
        SELECT Symbol, Date, Cash_and_Cash_Equivalents, Total_Assets, Total_Liabilities,
               Total_Debt, Total_Shareholder_Equity, Last_Updated
        FROM BalanceSheetData
        WHERE Symbol = ?
        ORDER BY Date ASC
        """
        cursor.execute(query, (ticker,))
        results = cursor.fetchall()

        if not results:
            print(f"No balance sheet data found for {ticker}.")
            return None

        # Map query results to dictionary using column names explicitly
        balance_sheet_data = [{
            'Symbol': row[0],
            'Date': row[1],
            'Cash_and_Cash_Equivalents': row[2],
            'Total_Assets': row[3],
            'Total_Liabilities': row[4],
            'Total_Debt': row[5],
            'Total_Shareholder_Equity': row[6],
            'Last_Updated': row[7]
        } for row in results]

        print("---balance sheet data in database fetched")
        return balance_sheet_data

    except sqlite3.Error as e:
        print(f"Database error while fetching balance sheet data for {ticker}: {e}")
        return None



def check_missing_balance_sheet_data(ticker, cursor):
    print(f"balance sheet data fetcher 2 Checking for missing balance sheet data for ticker: {ticker}")
    try:
        # Define the column names based on your database structure
        columns = [
            'Date', 'Cash_and_Cash_Equivalents', 'Total_Assets', 'Total_Liabilities',
            'Total_Debt', 'Total_Shareholder_Equity', 'Last_Updated'
        ]

        # Fetch all balance sheet data for the ticker
        cursor.execute("SELECT * FROM BalanceSheetData WHERE Symbol = ? ORDER BY Date", (ticker,))
        results = cursor.fetchall()
        print("---defining the names for balance sheet data")

        # Check for missing data
        missing_data = False
        for row in results:
            print("---checking for missing data")
            row_data = dict(zip(columns, row))
            for key, value in row_data.items():
                print("---checking row",row_data)
                # Check if the value is None or empty string
                if value is None or (isinstance(value, str) and not value.strip()):
                    print(f"Missing data for {key} in row: {row_data}")
                    missing_data = True
                    break  # No need to check further fields if one is missing

        return missing_data

    except sqlite3.Error as e:
        print(f"Database error while checking balance sheet data for {ticker}: {e}")
        return True  # Assume missing data if there's a database error


def is_balance_sheet_data_outdated(balance_sheet_data):
    print("balance sheet data fetcher 3 is balance sheet data outdated?")

    print(type(balance_sheet_data))  # Add this line to check the data type

    if not balance_sheet_data:
        print("No balance sheet data available.")
        return True  # Assume data is outdated if there's no data

    # Ensure balance_sheet_data is a list before accessing it
    if isinstance(balance_sheet_data, list) and balance_sheet_data:
        last_update_value = balance_sheet_data[-1].get('Last_Updated')
        print("---last updated", last_update_value)
    else:
        print("balance_sheet_data is not a list or is empty.")
        return True  # Treat as outdated if the structure is not as expected

    try:
        # Attempt to handle Last_Updated as a string in 'YYYY-MM-DD' format
        try:
            latest_update = datetime.strptime(last_update_value, '%Y-%m-%d %H:%M:%S')
        except TypeError:
            # If Last_Updated is a Unix timestamp (integer), convert accordingly
            latest_update = datetime.utcfromtimestamp(int(last_update_value))
    except (ValueError, TypeError) as e:
        print(f"Error handling Last_Updated value: {e}")
        return True  # Assume data is outdated if handling fails

    print(f"---Latest balance sheet data update date: {latest_update}")

    # Calculate the threshold date for updating data
    threshold_date = latest_update + timedelta(days=90 + 21)  # Example: 3 months + 3 weeks
    print(f"---Threshold date for updating balance sheet data: {threshold_date}")

    if datetime.utcnow() > threshold_date:
        print("---Balance sheet data is outdated")
        return True
    else:
        print("---Balance sheet data is up-to-date")
        return False



def fetch_balance_sheet_data_from_yahoo(ticker):
    print("balance sheet data fetcher 5 | fetch balance sheet data from yahoo")
    stock = yf.Ticker(ticker)
    balance_sheet_df = stock.quarterly_balance_sheet
    print("---checking if balance sheet data is empty")
    if balance_sheet_df.empty:
        print("---balance sheet data is empty for ticker:", ticker)
        return None

    print("---getting most recent quarter's data")
    latest_bs_data = balance_sheet_df.iloc[:, 0]
    date_of_last_reported_quarter = balance_sheet_df.columns[0].strftime('%Y-%m-%d')
    print(f"---date of last reported quarter: {date_of_last_reported_quarter}")

    print("---extracting required fields")
    balance_sheet_data = {
        'Symbol': ticker,  # Add the ticker symbol manually
        'Date_of_Last_Reported_Quarter': date_of_last_reported_quarter,
        'Cash': latest_bs_data.get('Cash And Cash Equivalents', None),
        'Total_Assets': latest_bs_data.get('Total Assets', None),
        'Total_Liabilities': latest_bs_data.get('Total Liabilities Net Minority Interest', None),
        'Debt': latest_bs_data.get('Total Debt', None),
        'Equity': latest_bs_data.get('Stockholders Equity', None),
        'Last_Updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    print(f"---extracted balance sheet data: {balance_sheet_data}")

    return balance_sheet_data


def store_fetched_balance_sheet_data(cursor, balance_sheet_data):
    print("balance sheet data fetcher 6 Storing Fetched balance sheet data")
    """
    Store the balance sheet data into the SQLite database.

    :param cursor: sqlite3.Cursor - The cursor to perform database operations.
    :param balance_sheet_data: dict - The balance sheet data to store.
    """
    # Prepare the SQL statement for inserting or updating data
    sql_statement = """
    INSERT INTO BalanceSheetData (
        Symbol, 
        Date, 
        Cash_and_Cash_Equivalents, 
        Total_Assets, 
        Total_Liabilities, 
        Total_Debt, 
        Total_Shareholder_Equity,  
        Last_Updated
    ) VALUES (
        ?, ?, ?, ?, ?, ?, ?, ?
    )
    ON CONFLICT(Symbol) DO UPDATE SET
        Date = excluded.Date,
        Cash_and_Cash_Equivalents = excluded.Cash_and_Cash_Equivalents,
        Total_Assets = excluded.Total_Assets,
        Total_Liabilities = excluded.Total_Liabilities,
        Total_Debt = excluded.Total_Debt,
        Total_Shareholder_Equity = excluded.Total_Shareholder_Equity,
        Last_Updated = excluded.Last_Updated;
    """
    # Parameters for the SQL statement
    params = (
        balance_sheet_data['Symbol'],
        balance_sheet_data['Date_of_Last_Reported_Quarter'],  # Make sure the key matches the dictionary
        balance_sheet_data['Cash'],  # This key must match your dictionary's key for cash
        balance_sheet_data['Total_Assets'],
        balance_sheet_data['Total_Liabilities'],
        balance_sheet_data['Debt'],  # This key must match your dictionary's key for debt
        balance_sheet_data['Equity'],  # This key must match your dictionary's key for equity
        balance_sheet_data['Last_Updated']
    )

    try:
        cursor.execute(sql_statement, params)
        cursor.connection.commit()
        print(f"Balance sheet data for {balance_sheet_data['Symbol']} on {balance_sheet_data['Date_of_Last_Reported_Quarter']} has been stored.")
    except sqlite3.Error as e:
        print(f"An error occurred while storing balance sheet data: {e}")


DB_PATH = 'stock data.db'
TICKER = 'AAPL'  # Example ticker


def balance_sheet_data_fetcher():

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Fetch existing balance sheet data from the database
    balance_sheet_data = fetch_balance_sheet_data(TICKER, cursor)

    # Check if data is missing or outdated
    if check_missing_balance_sheet_data(TICKER, cursor) or is_balance_sheet_data_outdated(balance_sheet_data):
        # Fetch new data from Yahoo Finance
        new_balance_sheet_data = fetch_balance_sheet_data_from_yahoo(TICKER)
        if new_balance_sheet_data:
            # Store the new data
            store_fetched_balance_sheet_data(cursor, new_balance_sheet_data)
            print(f"New balance sheet data stored for {TICKER}.")
        else:
            print(f"No new balance sheet data available for {TICKER}.")
    else:
        print(f"Balance sheet data for {TICKER} is up to date.")

    # Close the database connection
    conn.close()

if __name__ == "__main__":
    main()

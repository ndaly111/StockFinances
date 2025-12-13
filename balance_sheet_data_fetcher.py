#start of balance_sheet_data_fetcher.py

import sqlite3
from datetime import datetime, timedelta

import pandas as pd

from config import get_fmp_api_key
from data_providers import FMPDataProvider, DataProviderError

DB_PATH = 'stock data.db'
TICKER = 'AAPL'  # Example ticker


def fetch_balance_sheet_data(ticker, cursor):
    print("balance sheet data fetcher 1 fetch balance sheet data from db")
    print(f"Fetching balance sheet data for {ticker}")
    try:
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
        columns = [
            'Date', 'Cash_and_Cash_Equivalents', 'Total_Assets', 'Total_Liabilities',
            'Total_Debt', 'Total_Shareholder_Equity', 'Last_Updated'
        ]

        cursor.execute("SELECT * FROM BalanceSheetData WHERE Symbol = ? ORDER BY Date", (ticker,))
        results = cursor.fetchall()
        print("---defining the names for balance sheet data")

        missing_data = False
        for row in results:
            print("---checking for missing data")
            row_data = dict(zip(columns, row))
            for key, value in row_data.items():
                print("---checking row", row_data)
                if value is None or (isinstance(value, str) and not value.strip()):
                    print(f"Missing data for {key} in row: {row_data}")
                    missing_data = True
                    break
        return missing_data

    except sqlite3.Error as e:
        print(f"Database error while checking balance sheet data for {ticker}: {e}")
        return True


def is_balance_sheet_data_outdated(balance_sheet_data):
    print("balance sheet data fetcher 3 is balance sheet data outdated?")
    print(type(balance_sheet_data))

    if not balance_sheet_data:
        print("No balance sheet data available.")
        return True

    if isinstance(balance_sheet_data, list) and balance_sheet_data:
        last_update_value = balance_sheet_data[-1].get('Last_Updated')
        print("---last updated", last_update_value)
    else:
        print("balance_sheet_data is not a list or is empty.")
        return True

    try:
        try:
            latest_update = datetime.strptime(last_update_value, '%Y-%m-%d %H:%M:%S')
        except TypeError:
            latest_update = datetime.utcfromtimestamp(int(last_update_value))
    except (ValueError, TypeError) as e:
        print(f"Error handling Last_Updated value: {e}")
        return True

    print(f"---Latest balance sheet data update date: {latest_update}")
    threshold_date = latest_update + timedelta(days=111)
    print(f"---Threshold date for updating balance sheet data: {threshold_date}")

    if datetime.utcnow() > threshold_date:
        print("---Balance sheet data is outdated")
        return True
    else:
        print("---Balance sheet data is up-to-date")
        return False


def fetch_balance_sheet_data_from_provider(ticker, provider=None):
    print("balance sheet data fetcher 5 | fetch balance sheet data from licensed provider")
    provider = provider or FMPDataProvider(api_key=get_fmp_api_key())

    try:
        balance_sheet_data = provider.fetch_balance_sheet(ticker)
    except DataProviderError as exc:
        print(f"Provider error for {ticker}: {exc}")
        return None

    print(f"---extracted balance sheet data: {balance_sheet_data}")
    return balance_sheet_data


def store_fetched_balance_sheet_data(cursor, balance_sheet_data):
    print("balance sheet data fetcher 6 Storing Fetched balance sheet data")
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
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(Symbol) DO UPDATE SET
        Date = excluded.Date,
        Cash_and_Cash_Equivalents = excluded.Cash_and_Cash_Equivalents,
        Total_Assets = excluded.Total_Assets,
        Total_Liabilities = excluded.Total_Liabilities,
        Total_Debt = excluded.Total_Debt,
        Total_Shareholder_Equity = excluded.Total_Shareholder_Equity,
        Last_Updated = excluded.Last_Updated;
    """
    params = (
        balance_sheet_data['Symbol'],
        balance_sheet_data['Date_of_Last_Reported_Quarter'],
        balance_sheet_data['Cash'],
        balance_sheet_data['Total_Assets'],
        balance_sheet_data['Total_Liabilities'],
        balance_sheet_data['Debt'],
        balance_sheet_data['Equity'],
        balance_sheet_data['Last_Updated']
    )

    try:
        cursor.execute(sql_statement, params)
        cursor.connection.commit()
        print(f"Balance sheet data for {balance_sheet_data['Symbol']} on {balance_sheet_data['Date_of_Last_Reported_Quarter']} has been stored.")
    except sqlite3.Error as e:
        print(f"An error occurred while storing balance sheet data: {e}")


def delete_invalid_records(cursor):
    print("Cleaning up bad balance sheet records...")
    try:
        cursor.execute("DELETE FROM BalanceSheetData WHERE Last_Updated LIKE '-%'")
        bad = cursor.rowcount
        cursor.execute("""
            DELETE FROM BalanceSheetData
            WHERE 
                Cash_and_Cash_Equivalents IS NULL OR
                Total_Assets IS NULL OR
                Total_Liabilities IS NULL OR
                Total_Debt IS NULL OR
                Total_Shareholder_Equity IS NULL
        """)
        more_bad = cursor.rowcount
        cursor.connection.commit()
        print(f"Deleted {bad + more_bad} invalid records.")
    except sqlite3.Error as e:
        print(f"Error during cleanup: {e}")


def balance_sheet_data_fetcher():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Clean up previously bad data
    delete_invalid_records(cursor)

    balance_sheet_data = fetch_balance_sheet_data(TICKER, cursor)

    if check_missing_balance_sheet_data(TICKER, cursor) or is_balance_sheet_data_outdated(balance_sheet_data):
        new_balance_sheet_data = fetch_balance_sheet_data_from_provider(TICKER)

        required_keys = ['Cash', 'Total_Assets', 'Total_Liabilities', 'Debt', 'Equity']
        if new_balance_sheet_data and all(
            new_balance_sheet_data.get(k) is not None and not pd.isna(new_balance_sheet_data.get(k))
            for k in required_keys
        ):
            store_fetched_balance_sheet_data(cursor, new_balance_sheet_data)
            print(f"New balance sheet data stored for {TICKER}.")
        else:
            print(f"Skipping storing invalid balance sheet data for {TICKER}: {new_balance_sheet_data}")
    else:
        print(f"Balance sheet data for {TICKER} is up to date.")

    conn.close()

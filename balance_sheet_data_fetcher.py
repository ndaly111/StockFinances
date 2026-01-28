#start of balance_sheet_data_fetcher.py

import logging
import sqlite3
from datetime import datetime, timedelta

import pandas as pd

from config import get_fmp_api_key
from data_providers import FMPDataProvider, DataProviderError

DB_PATH = 'Stock Data.db'  # Fixed: consistent casing with other files
TICKER = 'AAPL'  # Example ticker

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def fetch_balance_sheet_data(ticker, cursor):
    logger.debug(f"[{ticker}] Fetching balance sheet data from DB")
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
            logger.debug(f"[{ticker}] No balance sheet data found")
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

        logger.debug(f"[{ticker}] Balance sheet data fetched: {len(balance_sheet_data)} records")
        return balance_sheet_data

    except sqlite3.Error as e:
        logger.error(f"[{ticker}] Database error fetching balance sheet data: {e}")
        return None


def check_missing_balance_sheet_data(ticker, cursor):
    logger.debug(f"[{ticker}] Checking for missing balance sheet data")
    try:
        columns = [
            'Date', 'Cash_and_Cash_Equivalents', 'Total_Assets', 'Total_Liabilities',
            'Total_Debt', 'Total_Shareholder_Equity', 'Last_Updated'
        ]

        cursor.execute("SELECT * FROM BalanceSheetData WHERE Symbol = ? ORDER BY Date", (ticker,))
        results = cursor.fetchall()

        missing_data = False
        for row in results:
            row_data = dict(zip(columns, row))
            for key, value in row_data.items():
                if value is None or (isinstance(value, str) and not value.strip()):
                    logger.debug(f"[{ticker}] Missing data for {key}")
                    missing_data = True
                    break
        return missing_data

    except sqlite3.Error as e:
        logger.error(f"[{ticker}] Database error checking balance sheet data: {e}")
        return True


def is_balance_sheet_data_outdated(balance_sheet_data):
    if not balance_sheet_data:
        logger.debug("No balance sheet data available")
        return True

    if isinstance(balance_sheet_data, list) and balance_sheet_data:
        last_update_value = balance_sheet_data[-1].get('Last_Updated')
    else:
        logger.debug("balance_sheet_data is not a list or is empty")
        return True

    try:
        try:
            latest_update = datetime.strptime(last_update_value, '%Y-%m-%d %H:%M:%S')
        except TypeError:
            latest_update = datetime.utcfromtimestamp(int(last_update_value))
    except (ValueError, TypeError) as e:
        logger.warning(f"Error parsing Last_Updated value: {e}")
        return True

    threshold_date = latest_update + timedelta(days=111)

    if datetime.utcnow() > threshold_date:
        logger.debug("Balance sheet data is outdated")
        return True
    else:
        logger.debug("Balance sheet data is up-to-date")
        return False


def fetch_balance_sheet_data_from_provider(ticker, provider=None):
    logger.debug(f"[{ticker}] Fetching balance sheet from licensed provider")
    provider = provider or FMPDataProvider(api_key=get_fmp_api_key())

    try:
        balance_sheet_data = provider.fetch_balance_sheet(ticker)
    except DataProviderError as exc:
        logger.warning(f"[{ticker}] Provider error: {exc}")
        return None

    logger.debug(f"[{ticker}] Balance sheet data extracted")
    return balance_sheet_data


def fetch_balance_sheet_data_from_yahoo(ticker, provider=None):
    logger.debug(f"[{ticker}] Fetching balance sheet via provider")
    # Reuse the provider-based fetch logic to maintain a single mapping/validation path.
    balance_sheet_data = fetch_balance_sheet_data_from_provider(ticker, provider)

    if balance_sheet_data is None:
        return None

    required_keys = {
        'Symbol',
        'Date_of_Last_Reported_Quarter',
        'Cash',
        'Total_Assets',
        'Total_Liabilities',
        'Debt',
        'Equity',
        'Last_Updated',
    }

    if not required_keys.issubset(balance_sheet_data.keys()):
        missing = required_keys.difference(balance_sheet_data.keys())
        logger.warning(f"[{ticker}] Missing balance sheet fields: {missing}")
        return None

    return balance_sheet_data


def store_fetched_balance_sheet_data(cursor, balance_sheet_data):
    logger.debug(f"[{balance_sheet_data.get('Symbol')}] Storing balance sheet data")
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
        logger.info(f"[{balance_sheet_data['Symbol']}] Balance sheet data stored")
    except sqlite3.Error as e:
        logger.error(f"Error storing balance sheet data: {e}")


def delete_invalid_records(cursor):
    logger.debug("Cleaning up bad balance sheet records")
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
        if bad + more_bad > 0:
            logger.info(f"Deleted {bad + more_bad} invalid balance sheet records")
    except sqlite3.Error as e:
        logger.error(f"Error during cleanup: {e}")


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
            logger.info(f"[{TICKER}] New balance sheet data stored")
        else:
            logger.warning(f"[{TICKER}] Skipping invalid balance sheet data")
    else:
        logger.debug(f"[{TICKER}] Balance sheet data is up to date")

    conn.close()

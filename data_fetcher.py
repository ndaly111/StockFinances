#start of data_fetcher.py

import logging
from datetime import datetime, timedelta

import pandas as pd
import sqlite3

from config import ALLOW_YAHOO_STORAGE, get_fmp_api_key
from data_providers import FMPDataProvider, DataProviderError
from split_utils import apply_split_adjustments, ensure_splits_table

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _ensure_split_alignment(ticker: str, cursor: sqlite3.Cursor) -> None:
    """Ensure split events are recorded and prior rows are split-adjusted."""
    try:
        ensure_splits_table(cursor)
        adjusted = apply_split_adjustments(ticker, cursor)
        if adjusted:
            logger.info(f"[{ticker}] Applied split adjustments before data fetch")
    except Exception as exc:
        logger.warning(f"[{ticker}] Split check failed: {exc}")


def fetch_ticker_data(ticker, cursor):
    logger.debug(f"[{ticker}] Fetching ticker data")
    try:
        _ensure_split_alignment(ticker, cursor)

        cursor.execute("PRAGMA table_info(Annual_Data)")
        columns = [col[1] for col in cursor.fetchall()]
        cursor.execute("SELECT * FROM Annual_Data WHERE Symbol = ? ORDER BY Date ASC", (ticker,))
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]

        if results:
            logger.debug(f"[{ticker}] Fetched {len(results)} rows")
            return results
        else:
            logger.debug(f"[{ticker}] No data found in database")
            return None
    except sqlite3.Error as e:
        logger.error(f"[{ticker}] Database error: {e}")
        return None


def get_latest_annual_data_date(ticker, ticker_data):
    logger.debug(f"[{ticker}] Getting latest annual data date")
    try:
        dates = [row['Date'] for row in ticker_data if row['Symbol'] == ticker]

        if dates:
            latest_date = max(dates)
            logger.debug(f"[{ticker}] Latest date: {latest_date}")
            return latest_date
        else:
            logger.debug(f"[{ticker}] No dates found")
            return None

    except Exception as e:
        logger.error(f"[{ticker}] Error processing data: {e}")
        return None


def determine_if_annual_data_missing(ticker_data):
    logger.debug("Determining if annual data is missing")

    if not ticker_data:
        logger.debug("No ticker data found - data is missing")
        return True

    try:
        years_listed = [datetime.strptime(row['Date'], '%Y-%m-%d').year for row in ticker_data]
        years_listed_sorted = sorted(years_listed)

        end_range = datetime.now().year - 1
        start_range = end_range - 3
        years_expected = list(range(start_range, end_range + 1))

        if years_expected == years_listed_sorted:
            logger.debug(f"Years match expected: {years_listed_sorted}")
            return False
        else:
            logger.debug(f"Years mismatch - expected {years_expected}, got {years_listed_sorted}")
            return True
    except Exception as e:
        logger.error(f"Error processing annual data: {e}")
        return True


def calculate_next_annual_check_date_from_data(ticker_data):
    logger.debug("Calculating next annual check date")

    if not ticker_data:
        logger.debug("No data found - needs fetch")
        return True

    dates = [row['Date'] for row in ticker_data]
    latest_date_str = max(dates) if dates else None

    if latest_date_str is None:
        logger.debug("No latest date found - needs fetch")
        return True

    latest_date = datetime.strptime(latest_date_str, '%Y-%m-%d')
    next_check_date = latest_date + timedelta(days=365 + 21)

    if datetime.now().date() > next_check_date.date():
        logger.debug("Time to check for new annual data")
        return True
    else:
        logger.debug(f"No need to check until {next_check_date.date()}")
        return False


def check_null_fields_annual(ticker, ticker_data):
    logger.debug(f"[{ticker}] Checking for null fields in annual data")
    if not ticker_data:
        logger.debug(f"[{ticker}] No ticker data - assuming incomplete")
        return True

    for entry in ticker_data:
        if entry.get('Revenue') in [None, ''] or entry.get('Net_Income') in [None, ''] or entry.get('EPS') in [None, '']:
            logger.debug(f"[{ticker}] Null fields found for date {entry.get('Date')}")
            return True
    return False


def fetch_annual_data_from_provider(ticker, provider=None):
    logger.debug(f"[{ticker}] Fetching annual data from provider")
    provider = provider or FMPDataProvider(api_key=get_fmp_api_key())
    try:
        raw_records = provider.fetch_annual_financials(ticker)
        if not raw_records:
            logger.warning(f"[{ticker}] No financial data from provider")
            return pd.DataFrame()

        financials = pd.DataFrame(raw_records)
        column_mapping = {
            'Revenue': 'Revenue',
            'Net_Income': 'Net_Income',
            'EPS': 'EPS',
            'Date': 'Date'
        }

        missing_columns = [db_col for db_col in ['Revenue', 'Net_Income', 'EPS', 'Date'] if db_col not in financials.columns]
        if missing_columns:
            logger.warning(f"[{ticker}] Missing required columns: {missing_columns}")
            return pd.DataFrame()

        financials = financials[list(column_mapping.keys())]
        financials.rename(columns=column_mapping, inplace=True)
        logger.debug(f"[{ticker}] Financials fetched and aligned")
        return financials
    except DataProviderError as e:
        logger.warning(f"[{ticker}] Provider error: {e}")
        return pd.DataFrame()


def store_annual_data(ticker, annual_data, cursor):
    logger.debug(f"[{ticker}] Storing annual data")

    rows_to_upsert = []

    for index, row in annual_data.iterrows():
        date_str = row['Date'].strftime('%Y-%m-%d') if isinstance(row['Date'], pd.Timestamp) else row['Date']

        cursor.execute("""
            SELECT * FROM Annual_Data
            WHERE Symbol = ? AND Date = ? AND Revenue IS NOT NULL AND Net_Income IS NOT NULL AND EPS IS NOT NULL;
        """, (ticker, date_str))
        existing_row = cursor.fetchone()

        if existing_row:
            logger.debug(f"[{ticker}] Complete data exists for {date_str} - skipping")
            continue

        rows_to_upsert.append((ticker, date_str, row['Revenue'], row['Net_Income'], row['EPS']))

    if rows_to_upsert:
        try:
            insert_sql = """
                INSERT INTO Annual_Data (Symbol, Date, Revenue, Net_Income, EPS, Last_Updated)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(Symbol, Date) DO UPDATE SET
                Revenue = EXCLUDED.Revenue,
                Net_Income = EXCLUDED.Net_Income,
                EPS = EXCLUDED.EPS,
                Last_Updated = CURRENT_TIMESTAMP
                WHERE Revenue IS NULL OR Net_Income IS NULL OR EPS IS NULL;
            """
            cursor.executemany(insert_sql, rows_to_upsert)
            cursor.connection.commit()
            logger.info(f"[{ticker}] Stored {len(rows_to_upsert)} annual data rows")
        except sqlite3.Error as e:
            logger.error(f"[{ticker}] Database error storing annual data: {e}")


def handle_ttm_duplicates(ticker, cursor):
    logger.debug(f"[{ticker}] Checking for duplicate TTM entries")
    try:
        cursor.execute("SELECT * FROM TTM_Data WHERE Symbol = ?", (ticker,))
        results = cursor.fetchall()
        if len(results) > 1:
            logger.warning(f"[{ticker}] Multiple TTM entries detected: {len(results)}")
            cursor.execute("DELETE FROM TTM_Data WHERE Symbol = ?", (ticker,))
            cursor.connection.commit()
            logger.info(f"[{ticker}] Cleared duplicate TTM entries")
            return True
        return False
    except sqlite3.Error as e:
        logger.error(f"[{ticker}] Database error during duplicate check: {e}")
        return False


def fetch_ttm_data(ticker, cursor):
    logger.debug(f"[{ticker}] Fetching TTM data")
    try:
        _ensure_split_alignment(ticker, cursor)

        cursor.execute("PRAGMA table_info(TTM_Data)")
        columns = [col[1] for col in cursor.fetchall()]

        cursor.execute("SELECT * FROM TTM_Data WHERE Symbol = ? ORDER BY Quarter DESC", (ticker,))
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]

        if len(results) > 1:
            logger.warning(f"[{ticker}] Multiple TTM entries found - clearing")
            cursor.execute("DELETE FROM TTM_Data WHERE Symbol = ?", (ticker,))
            cursor.connection.commit()
            return None
        elif results:
            logger.debug(f"[{ticker}] Fetched TTM data")
            return results[0]
        else:
            logger.debug(f"[{ticker}] No TTM data found")
            return None
    except sqlite3.Error as e:
        logger.error(f"[{ticker}] Database error: {e}")
        return None


def check_null_fields_ttm(ttm_data):
    logger.debug("Checking for null fields in TTM data")
    if not ttm_data or not isinstance(ttm_data, list):
        logger.debug("TTM data is missing or invalid format")
        return True

    for row in ttm_data:
        if not isinstance(row, dict):
            logger.debug("Row in TTM data is not a dictionary")
            return True
        if row.get('TTM_Revenue') is None or row.get('TTM_Net_Income') is None or row.get('TTM_EPS') is None:
            logger.debug("Null fields found in TTM data")
            return True

    logger.debug("TTM data is complete")
    return False


def is_ttm_data_outdated(ttm_data):
    logger.debug("Checking if TTM data is outdated")
    try:
        latest_date = max(datetime.strptime(row['Quarter'], '%Y-%m-%d') for row in ttm_data)
        threshold_date = latest_date + timedelta(days=90)

        if datetime.now() > threshold_date:
            logger.debug("TTM data is outdated")
            return True
        else:
            logger.debug("TTM data is up-to-date")
            return False
    except Exception as e:
        logger.error(f"Error checking TTM data: {e}")
        return True


def is_ttm_data_blank(ttm_data):
    logger.debug("Checking if TTM data is blank")
    if ttm_data is None or not ttm_data:
        logger.debug("TTM data is blank")
        return True
    else:
        logger.debug("TTM data has content")
        return False


def fetch_ttm_data_from_yahoo(ticker):
    if not ALLOW_YAHOO_STORAGE:
        raise RuntimeError(
            "Yahoo-derived data ingestion is disabled. Set ALLOW_YAHOO_STORAGE=true to explicitly allow it."
        )

    import yfinance as yf

    stock = yf.Ticker(ticker)
    ttm_financials = stock.quarterly_financials
    if ttm_financials is None or ttm_financials.empty:
        return None

    ttm_data = {}
    try:
        ttm_data['TTM_Revenue'] = ttm_financials.loc['Total Revenue', :].iloc[:4].sum()
        ttm_data['TTM_Net_Income'] = ttm_financials.loc['Net Income', :].iloc[:4].sum()
    except KeyError:
        ttm_data['TTM_Revenue'] = None
        ttm_data['TTM_Net_Income'] = None

    ttm_data['TTM_EPS'] = stock.info.get('trailingEps', None)
    ttm_data['Shares_Outstanding'] = stock.info.get('sharesOutstanding', None)
    ttm_data['Quarter'] = stock.quarterly_financials.columns[0].strftime('%Y-%m-%d')
    return ttm_data


def store_ttm_data(ticker, ttm_data, cursor):
    logger.debug(f"[{ticker}] Storing TTM data")

    if not ALLOW_YAHOO_STORAGE:
        raise RuntimeError("Yahoo-derived data storage is disabled. Set ALLOW_YAHOO_STORAGE=true to override.")

    ttm_values = (
        ticker,
        ttm_data['TTM_Revenue'],
        ttm_data['TTM_Net_Income'],
        ttm_data['TTM_EPS'],
        ttm_data.get('Shares_Outstanding'),
        ttm_data['Quarter']
    )

    try:
        cursor.execute("""
            INSERT OR REPLACE INTO TTM_Data (Symbol, TTM_Revenue, TTM_Net_Income, TTM_EPS, Shares_Outstanding, Quarter, Last_Updated)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP);
        """, ttm_values)
        cursor.connection.commit()
        logger.info(f"[{ticker}] TTM data stored")
    except sqlite3.Error as e:
        logger.error(f"[{ticker}] Database error storing TTM data: {e}")


def prompt_and_update_partial_entries(ticker, cursor, is_remote=False):
    logger.debug(f"[{ticker}] Checking for partial entries")

    if is_remote:
        logger.debug("Running in remote mode - skipping user prompts")
        return

    cursor.execute("""
        SELECT Symbol, Date, Revenue, Net_Income, EPS
        FROM Annual_Data
        WHERE (Revenue IS NOT NULL OR Net_Income IS NOT NULL OR EPS IS NOT NULL)
        AND (Revenue IS NULL OR Net_Income IS NULL OR EPS IS NULL)
        AND Symbol = ?
    """, (ticker,))
    partial_rows = cursor.fetchall()

    for row in partial_rows:
        symbol, date, revenue, net_income, eps = row
        updates = []
        params = []

        logger.info(f"[{symbol}] Partial data found for {date}")

        if revenue is None:
            response = input(f"Please enter the Revenue for {symbol} in {date} (or type 'skip' to leave unchanged): ")
            if response.lower() != 'skip':
                updates.append("Revenue = ?")
                params.append(response)

        if net_income is None:
            response = input(f"Please enter the Net Income for {symbol} in {date} (or type 'skip' to leave unchanged): ")
            if response.lower() != 'skip':
                updates.append("Net_Income = ?")
                params.append(response)

        if eps is None:
            response = input(
                f"Please enter the EPS for {symbol} in {date} (or type 'skip' to leave unchanged): ")
            if response.lower() != 'skip':
                updates.append("EPS = ?")
                params.append(response)

        if updates:
            update_query = f"UPDATE Annual_Data SET {', '.join(updates)} WHERE Symbol = ? AND Date = ?"
            params.extend([symbol, date])
            cursor.execute(update_query, params)
            cursor.connection.commit()
            logger.info(f"[{symbol}] Data updated for {date}")


#end of data_fetcher.py

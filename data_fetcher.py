#start of data_fetcher.py

import logging
from functools import lru_cache

import yfinance as yf
import pandas as pd
import sqlite3
from datetime import datetime, timedelta


logger = logging.getLogger(__name__)


_TABLE_COLUMN_CACHE = {}


def _get_table_columns(cursor, table_name):
    """Return cached column names for ``table_name`` using ``cursor``."""

    key = (id(cursor.connection), table_name)
    if key not in _TABLE_COLUMN_CACHE:
        cursor.execute(f"PRAGMA table_info({table_name})")
        _TABLE_COLUMN_CACHE[key] = [col[1] for col in cursor.fetchall()]
    return _TABLE_COLUMN_CACHE[key]


@lru_cache(maxsize=None)
def _parse_date(date_str):
    return datetime.strptime(date_str, '%Y-%m-%d')



def fetch_ticker_data(ticker, cursor):
    logger.debug("Fetching annual data for %s", ticker)
    try:
        columns = _get_table_columns(cursor, 'Annual_Data')
        cursor.execute("SELECT * FROM Annual_Data WHERE Symbol = ? ORDER BY Date ASC", (ticker,))
        rows = cursor.fetchall()
        results = [dict(zip(columns, row)) for row in rows]
        if results:
            logger.debug("Fetched %d annual rows for %s", len(results), ticker)
            return results
        logger.info("No annual data found for %s", ticker)
        return None
    except sqlite3.Error as e:
        logger.error("Database error while fetching annual data for %s: %s", ticker, e)
        return None



def get_latest_annual_data_date(ticker, ticker_data):
    try:
        if not ticker_data:
            logger.debug("No ticker data available when checking latest date for %s", ticker)
            return None

        dates = [row['Date'] for row in ticker_data if row.get('Symbol') == ticker]
        if not dates:
            logger.info("No annual rows matched %s when determining latest date", ticker)
            return None

        latest_date = max(dates)
        logger.debug("Latest annual date for %s is %s", ticker, latest_date)
        return latest_date
    except Exception as e:
        logger.error("Error processing annual data for %s: %s", ticker, e)
        return None


def determine_if_annual_data_missing(ticker_data):
    if not ticker_data:
        logger.debug("No ticker data found. Treating annual data as missing.")
        return True

    try:
        years_listed = sorted({_parse_date(row['Date']).year for row in ticker_data})
        end_range = datetime.now().year - 1
        start_range = end_range - 3
        years_expected = list(range(start_range, end_range + 1))

        if years_expected == years_listed:
            logger.debug("Annual data covers expected years %s", years_expected)
            return False

        logger.info("Annual data missing years. Expected %s but found %s", years_expected, years_listed)
        return True
    except Exception as e:
        logger.error("Error while checking annual data completeness: %s", e)
        return True


def calculate_next_annual_check_date_from_data(ticker_data):
    if not ticker_data:
        logger.debug("No ticker data provided; annual data needs refresh.")
        return True

    dates = [row['Date'] for row in ticker_data]
    if not dates:
        logger.debug("Ticker data contains no dates; annual data needs refresh.")
        return True

    latest_date_str = max(dates)
    latest_date = _parse_date(latest_date_str)
    next_check_date = latest_date + timedelta(days=365 + 21)

    if datetime.now().date() > next_check_date.date():
        logger.debug("Annual data last updated on %s; refresh required.", latest_date_str)
        return True

    logger.debug("Annual data last updated on %s; refresh not yet required.", latest_date_str)
    return False



def check_null_fields_annual(ticker, ticker_data):
    if not ticker_data:
        logger.debug("No annual data found for %s when checking null fields.", ticker)
        return True

    for entry in ticker_data:
        if any(entry.get(field) in (None, '') for field in ('Revenue', 'Net_Income', 'EPS')):
            logger.info("Null or empty annual fields detected for %s: %s", ticker, entry)
            return True
    return False



def fetch_annual_data_from_yahoo(ticker):
    logger.debug("Fetching annual data from Yahoo Finance for %s", ticker)
    try:
        stock = yf.Ticker(ticker)
        financials = stock.financials

        if financials.empty:
            logger.info("Yahoo Finance returned no financials for %s", ticker)
            return pd.DataFrame()

        financials = financials.T
        financials['Date'] = financials.index

        column_mapping = {
            'Total Revenue': 'Revenue',
            'Net Income': 'Net_Income',
            'Basic EPS': 'EPS'
        }

        renamed_columns = {yahoo: db for yahoo, db in column_mapping.items() if yahoo in financials.columns}
        if len(renamed_columns) < len(column_mapping):
            missing_columns = set(column_mapping.values()) - set(renamed_columns.values())
            logger.warning("Missing required columns for %s: %s", ticker, missing_columns)
            return pd.DataFrame()

        financials = financials.rename(columns=renamed_columns)
        return financials
    except Exception as e:
        logger.error("Error fetching data from Yahoo Finance for %s: %s", ticker, e)
        return pd.DataFrame()


def store_annual_data(ticker, annual_data, cursor):
    logger.debug("Storing annual data for %s", ticker)

    rows_to_upsert = []

    for _, row in annual_data.iterrows():
        date_str = row['Date'].strftime('%Y-%m-%d') if isinstance(row['Date'], pd.Timestamp) else row['Date']

        cursor.execute("""
            SELECT 1 FROM Annual_Data
            WHERE Symbol = ? AND Date = ? AND Revenue IS NOT NULL AND Net_Income IS NOT NULL AND EPS IS NOT NULL;
        """, (ticker, date_str))
        if cursor.fetchone():
            logger.debug("Skipping annual row for %s on %s; already complete.", ticker, date_str)
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
            logger.info("Upserted %d annual rows for %s", len(rows_to_upsert), ticker)
        except sqlite3.Error as e:
            logger.error("Database error while storing annual data for %s: %s", ticker, e)


def handle_ttm_duplicates(ticker, cursor):
    logger.debug("Checking for duplicate TTM entries for %s", ticker)
    try:
        cursor.execute("SELECT rowid FROM TTM_Data WHERE Symbol = ?", (ticker,))
        rows = cursor.fetchall()
        if len(rows) > 1:
            logger.info("Removing %d duplicate TTM rows for %s", len(rows) - 1, ticker)
            cursor.execute("DELETE FROM TTM_Data WHERE Symbol = ?", (ticker,))
            cursor.connection.commit()
            return True
        return False
    except sqlite3.Error as e:
        logger.error("Database error during TTM duplicate check for %s: %s", ticker, e)
        return False


def fetch_ttm_data(ticker, cursor):
    logger.debug("Fetching TTM data for %s", ticker)
    try:
        columns = _get_table_columns(cursor, 'TTM_Data')
        cursor.execute("SELECT * FROM TTM_Data WHERE Symbol = ? ORDER BY Quarter DESC", (ticker,))
        rows = cursor.fetchall()
        results = [dict(zip(columns, row)) for row in rows]

        if len(results) > 1:
            logger.info("Multiple TTM rows found for %s; clearing stale entries.", ticker)
            cursor.execute("DELETE FROM TTM_Data WHERE Symbol = ?", (ticker,))
            cursor.connection.commit()
            return None
        if results:
            logger.debug("Fetched TTM data for %s", ticker)
            return results[0]

        logger.info("No TTM data found for %s", ticker)
        return None
    except sqlite3.Error as e:
        logger.error("Database error while fetching TTM data for %s: %s", ticker, e)
        return None






def check_null_fields_ttm(ttm_data):
    if not ttm_data or not isinstance(ttm_data, list):
        logger.debug("TTM data missing or not list; treating as incomplete.")
        return True

    for row in ttm_data:
        if not isinstance(row, dict):
            logger.warning("Unexpected TTM row format: %s", row)
            return True
        if any(row.get(field) is None for field in ('TTM_Revenue', 'TTM_Net_Income', 'TTM_EPS')):
            logger.info("Null or empty TTM fields detected: %s", row)
            return True

    logger.debug("TTM dataset contains no null fields.")
    return False





def is_ttm_data_outdated(ttm_data):
    try:
        latest_date = max(_parse_date(row['Quarter']) for row in ttm_data)
        threshold_date = latest_date + timedelta(days=90)
        if datetime.now() > threshold_date:
            logger.debug("TTM data last updated on %s; refresh required.", latest_date.date())
            return True
        logger.debug("TTM data last updated on %s; refresh not required.", latest_date.date())
        return False
    except Exception as e:
        logger.error("Unable to determine if TTM data is outdated: %s", e)
        return True

def is_ttm_data_blank(ttm_data):
    if not ttm_data:
        logger.debug("TTM dataset is blank.")
        return True
    logger.debug("TTM dataset contains data.")
    return False


def fetch_ttm_data_from_yahoo(ticker):
    stock = yf.Ticker(ticker)
    ttm_financials = stock.quarterly_financials
    if ttm_financials is None or ttm_financials.empty:
        return None

    ttm_data = {}
    # Attempt to calculate TTM Revenue and Net Income
    try:
        ttm_data['TTM_Revenue'] = ttm_financials.loc['Total Revenue', :].iloc[:4].sum()
        ttm_data['TTM_Net_Income'] = ttm_financials.loc['Net Income', :].iloc[:4].sum()
    except KeyError:
        # If the key doesn't exist, or an error occurs, default these values to None
        ttm_data['TTM_Revenue'] = None
        ttm_data['TTM_Net_Income'] = None

    # Directly use trailingEps for TTM_EPS without conditionally checking for NaNs in quarterly EPS data
    ttm_data['TTM_EPS'] = stock.info.get('trailingEps', None)

    # Get shares outstanding
    ttm_data['Shares_Outstanding'] = stock.info.get('sharesOutstanding', None)

    # Get the quarter information
    ttm_data['Quarter'] = stock.quarterly_financials.columns[0].strftime('%Y-%m-%d')
    return ttm_data



def store_ttm_data(ticker, ttm_data, cursor):
    logger.debug("Storing TTM data for %s", ticker)

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
        logger.info("Stored TTM data for %s", ticker)
    except sqlite3.Error as e:
        logger.error("Database error while storing TTM data for %s: %s", ticker, e)



def prompt_and_update_partial_entries(ticker, cursor, is_remote=False):
    logger.debug("Prompting to fill partial annual data for %s", ticker)

    if is_remote:
        logger.debug("Remote mode active; skipping interactive prompts.")
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
        logger.info("Partial annual data found for %s on %s", symbol, date)

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
            stock = yf.Ticker(symbol)
            trailing_eps = stock.info.get('trailingEps')

            if trailing_eps is not None:
                logger.info("Fetched trailing EPS for %s: %s", symbol, trailing_eps)
                eps = trailing_eps
                updates.append("EPS = ?")
                params.append(eps)
            else:
                response = input(
                    f"Please enter the EPS for {symbol} in {date} (or type 'skip' to leave unchanged): "
                )
                if response.lower() != 'skip':
                    updates.append("EPS = ?")
                    params.append(response)

        if updates:
            update_query = f"UPDATE Annual_Data SET {', '.join(updates)} WHERE Symbol = ? AND Date = ?"
            params.extend([symbol, date])
            cursor.execute(update_query, params)
            cursor.connection.commit()
            logger.info("Updated partial annual data for %s on %s", symbol, date)


# end of data_fetcher.py
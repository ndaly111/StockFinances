#start of data_fetcher.py

import yfinance as yf
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import os



def fetch_ticker_data(ticker, cursor):
    print("data_fetcher 1(new) fetching ticker data")
    try:
        cursor.execute("PRAGMA table_info(Annual_Data)")
        columns = [col[1] for col in cursor.fetchall()]  # Get column names
        print("---getting column names",columns)
        cursor.execute("SELECT * FROM Annual_Data WHERE Symbol = ? ORDER BY Date ASC", (ticker,))
        print("---fetching all data for ticker")
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]  # Convert each tuple to a dictionary
        print("---converting tuple to a dictionary",results)

        if results:

            print("---fetched {len(results)} rows for {ticker}")
            print("---storing into the data frame variable", results)
            return results
        else:
            print(f"No data found for {ticker} in the database.")
            return None
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return None



def get_latest_annual_data_date(ticker, ticker_data):
    print("data_fetcher 2(new) getting latest annual data date from ticker data")
    try:
        # Assuming ticker_data is a list of dictionaries, each with a 'Date' key
        dates = [row['Date'] for row in ticker_data if row['Symbol'] == ticker]

        if dates:
            latest_date = max(dates)
            print(f"---collecting the most recent date from ticker data: {latest_date}")
            print(f"Latest date in data for {ticker}: {latest_date}")
            return latest_date
        else:
            print(f"No data found for {ticker} in the data.")
            return None

    except Exception as e:  # Catch a more general exception if not interacting with a database
        print(f"Error processing data: {e}")
        return None


def determine_if_annual_data_missing(ticker_data):
    print("data fetcher 3(new) determine if annual data missing")

    # Check if ticker_data is None or empty
    if not ticker_data:
        print("---No ticker data found. Data is missing.")
        return True  # Indicate that data is missing

    # Proceed if ticker_data is not None
    try:
        years_listed = [datetime.strptime(row['Date'], '%Y-%m-%d').year for row in ticker_data]
        years_listed_sorted = sorted(years_listed)
        print("---years listed sorted", years_listed_sorted)

        end_range = datetime.now().year - 1
        print("---calculating end range", end_range)
        start_range = end_range - 3
        print("---calculating start range", start_range)

        years_expected = list(range(start_range, end_range + 1))
        print("---determining the years expected", years_expected)

        print("---comparing years expected to the current years")
        if years_expected == years_listed_sorted:
            print("---years expected is equal to years sorted", years_listed_sorted)
            return False  # No need to fetch years
        else:
            print("---years expected is not equal to years sorted", years_listed_sorted)
            return True  # Missing data detected
    except Exception as e:
        print(f"Error processing annual data: {e}")
        return True  # Assume data is missing in case of any error

def calculate_next_annual_check_date_from_data(ticker_data):
    print("data fetcher 4(new) calculate next annual check date based on the latest date from data")

    if not ticker_data:
        print("---No data found, data needs to be fetched")
        return True

    # Correctly extracting all dates using dictionary access
    dates = [row['Date'] for row in ticker_data]
    print("extracting all dates", dates)

    # Find the latest date
    latest_date_str = max(dates) if dates else None
    print("---find the latest date", latest_date_str)

    if latest_date_str is None:
        print("---No latest date found after extraction, data needs to be fetched")
        return True

    # Convert the latest date string from the data to a datetime object
    latest_date = datetime.strptime(latest_date_str, '%Y-%m-%d')
    print(f"---Latest date from data: {latest_date_str}")

    # Calculate the next check date by adding one year and three weeks to the latest date
    next_check_date = latest_date + timedelta(days=365 + 21)  # Equivalent to one year and three weeks
    print(f"---Next annual check date should be: {next_check_date.date()}")

    # Compare the next check date to today's date
    if datetime.now().date() > next_check_date.date():
        print("---It's time to check for new annual data.")
        return True
    else:
        print("---No need to check for new annual data yet.")
        return False



def check_null_fields_annual(ticker, ticker_data):
    print("data_fetcher 5(new) checking for null fields in annual data")
    if not ticker_data:
        print("---No ticker data found. Assuming data is missing or incomplete.")
        return True  # Return True to indicate that data is missing or might have null fields

    for entry in ticker_data:
        print("---checking ticker data", entry)
        # Corrected line: use 'entry' instead of 'row'
        print([type(x) for x in
               (ticker, entry['Date'], entry['Revenue'], entry['Net_Income'], entry['EPS'], entry['Last_Updated'])])

        # Access dictionary values by keys correctly
        if entry.get('Revenue') in [None, ''] or entry.get('Net_Income') in [None, ''] or entry.get('EPS') in [None, '']:
            print(f"Null or empty fields found in Annual Data: {entry}")
            return True
    return False



def fetch_annual_data_from_yahoo(ticker):
    print("data_fetcher 6(new) fetch annual data from yahoo")
    try:
        print("---trying to fetch annual data")
        stock = yf.Ticker(ticker)
        financials = stock.financials
        print("---collected stock variable and financials",stock,financials)

        if financials.empty:
            print(f"No financial data available for {ticker}.")
            return pd.DataFrame()

        financials = financials.T  # Transpose to get dates as rows
        print("---transpose financials",financials)
        financials['Date'] = financials.index  # Keep the full date
        print("---financials date")

        # Define a mapping of Yahoo Finance column names to your database column names
        column_mapping = {
            'Total Revenue': 'Revenue',
            'Net Income': 'Net_Income',
            'Basic EPS': 'EPS'
        }
        print("---mapping columns")

        # Check and rename the columns as per the mapping
        renamed_columns = {yahoo: db for yahoo, db in column_mapping.items() if yahoo in financials.columns}
        print("---renaming columns from yahoo to the db")
        if len(renamed_columns) < len(column_mapping):
            missing_columns = set(column_mapping.values()) - set(renamed_columns.values())
            print(f"Missing required columns for {ticker}: {missing_columns}")
            return pd.DataFrame()

        financials.rename(columns=renamed_columns, inplace=True)
        print("---financials renamed")


        return financials
    except Exception as e:
        print(f"Error fetching data from Yahoo Finance for {ticker}: {e}")
        return pd.DataFrame()


def store_annual_data(ticker, annual_data, cursor):
    print("data_fetcher (new)7 storing annual data")
    print(f"Storing updated annual data for {ticker}")

    for index, row in annual_data.iterrows():
        # Convert the Date from Timestamp to string format if necessary
        date_str = row['Date'].strftime('%Y-%m-%d') if isinstance(row['Date'], pd.Timestamp) else row['Date']

        # Check for the existence and completeness of the row in the database
        cursor.execute("""
            SELECT * FROM Annual_Data
            WHERE Symbol = ? AND Date = ? AND Revenue IS NOT NULL AND Net_Income IS NOT NULL AND EPS IS NOT NULL;
        """, (ticker, date_str))  # Use date_str instead of row['Date']
        existing_row = cursor.fetchone()

        # If the row exists and is complete, skip the update for this row
        if existing_row:
            print(f"Complete data already exists for {ticker} on {date_str}. Skipping update.")  # Use date_str in log
            continue

        # Otherwise, insert or update the row
        try:
            cursor.execute("""
                INSERT INTO Annual_Data (Symbol, Date, Revenue, Net_Income, EPS, Last_Updated)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(Symbol, Date) DO UPDATE SET
                Revenue = EXCLUDED.Revenue,
                Net_Income = EXCLUDED.Net_Income,
                EPS = EXCLUDED.EPS,
                Last_Updated = CURRENT_TIMESTAMP
                WHERE Revenue IS NULL OR Net_Income IS NULL OR EPS IS NULL;
            """, (ticker, date_str, row['Revenue'], row['Net_Income'], row['EPS']))  # Use date_str here as well
            cursor.connection.commit()
        except sqlite3.Error as e:
            print(f"Database error while storing/updating annual data for {ticker}: {e}")


def handle_ttm_duplicates(ticker, cursor):
    print("Checking for duplicate TTM entries for ticker:", ticker)
    try:
        cursor.execute("SELECT * FROM TTM_Data WHERE Symbol = ?", (ticker,))
        results = cursor.fetchall()
        if len(results) > 1:
            print("---Multiple TTM entries detected for", ticker, ":", len(results))
            cursor.execute("DELETE FROM TTM_Data WHERE Symbol = ?", (ticker,))
            cursor.connection.commit()
            print("---Cleared duplicate TTM entries for", ticker)
            return True  # Indicates duplicates were found and cleared
        return False  # Indicates no duplicates were found
    except sqlite3.Error as e:
        print("Database error during duplicate check:", e)
        return False


def fetch_ttm_data(ticker, cursor):
    print("data fetcher (new)0 Fetching TTM data for ticker")
    try:
        # Fetch the column names for TTM_Data table
        cursor.execute("PRAGMA table_info(TTM_Data)")
        columns = [col[1] for col in cursor.fetchall()]  # Get column names
        print("---getting column names", columns)

        # Fetch all TTM data entries for the given ticker, ordered by Quarter
        cursor.execute("SELECT * FROM TTM_Data WHERE Symbol = ? ORDER BY Quarter DESC", (ticker,))
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]  # Convert each tuple to a dictionary
        print("---converting tuple to a dictionary", results)

        if len(results) > 1:
            print("---Multiple TTM entries found for", ticker, "clearing existing data...")
            cursor.execute("DELETE FROM TTM_Data WHERE Symbol = ?", (ticker,))
            cursor.connection.commit()
            print("---All existing TTM data for", ticker, "has been cleared.")
            return None  # Returning None to indicate that new data needs to be fetched
        elif results:
            print(f"---Fetched {len(results)} rows of TTM data for {ticker}")
            return results[0]  # Return the most recent entry if only one exists
        else:
            print(f"No TTM data found for {ticker} in the database.")
            return None
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return None






def check_null_fields_ttm(ttm_data):
    print("data_fetcher 9(new) checking for null fields in TTM data")
    # Check if ttm_data is a list and not empty
    if not ttm_data or not isinstance(ttm_data, list):
        print("---TTM data is missing or not in the expected format (list).")
        return True  # Assume true to indicate missing or incomplete data

    # Check each dictionary in the list for null or empty fields
    for row in ttm_data:
        # Ensure that row is a dictionary
        if not isinstance(row, dict):
            print("---Row in TTM data is not a dictionary:", row)
            return True  # Invalid data format detected
        # Check for null or empty fields using get method on the dictionary
        if row.get('TTM_Revenue') is None or row.get('TTM_Net_Income') is None or row.get('TTM_EPS') is None:
            print("---Null or empty fields found in TTM Data:", row)
            return True  # Null data detected

    print("---TTM data is complete with no null or empty fields.")
    return False  # No null fields found





def is_ttm_data_outdated(ttm_data):
    print("data_fetcher 10(new) checking if TTM data is outdated")
    print("---pre-processed ttm_data", ttm_data)
    try:
        # Since ttm_data is a list of dictionaries, access the 'Quarter' field directly
        latest_date = max(datetime.strptime(row['Quarter'], '%Y-%m-%d') for row in ttm_data)
        print(f"---Latest TTM data date: {latest_date}")

        # Calculate the date 3 months and 3 weeks ago
        threshold_date = latest_date + timedelta(days=90)  # approximately three months
        print(f"---Threshold date for updating TTM data: {threshold_date}")

        # If the current date is past the threshold date, TTM data needs updating
        if datetime.now() > threshold_date:
            print("---TTM data is outdated")
            return True
        else:
            print("---TTM data is up-to-date")
            return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return True

def is_ttm_data_blank(ttm_data):
    print("data fetcher 11(new) is ttm data blank?")
    if ttm_data is None or not ttm_data:
        print("---ttm is blank")
        return True
    else:
        print("---ttm has data")
        return False


import yfinance as yf

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
    print("Storing TTM data")
    print(f"Storing updated TTM data for {ticker}")

    # Prepare the data tuple for the SQL query
    ttm_values = (
        ticker,
        ttm_data['TTM_Revenue'],
        ttm_data['TTM_Net_Income'],
        ttm_data['TTM_EPS'],
        ttm_data.get('Shares_Outstanding'),  # Get shares outstanding, using .get to avoid KeyError if not present
        ttm_data['Quarter']
    )

    # Use INSERT OR REPLACE to overwrite existing data if the same ticker and quarter are found
    try:
        cursor.execute("""
            INSERT OR REPLACE INTO TTM_Data (Symbol, TTM_Revenue, TTM_Net_Income, TTM_EPS, Shares_Outstanding, Quarter, Last_Updated)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP);
        """, ttm_values)
        cursor.connection.commit()
        print(f"TTM data stored/updated for {ticker}")
    except sqlite3.Error as e:
        print(f"Database error while storing/updating TTM data for {ticker}: {e}")



def prompt_and_update_partial_entries(ticker, cursor, is_remote=False):
    print("data_fetcher 9 prompt and update partial entries")
    print("prompt and updating partial rows")

    if is_remote:
        print("Running in remote mode, skipping user prompts.")
        return

    else:
        # Select rows with at least one non-NULL and at least one NULL among Revenue, Net_Income, EPS
        cursor.execute("""
            SELECT Symbol, Date, Revenue, Net_Income, EPS 
            FROM Annual_Data 
            WHERE (Revenue IS NOT NULL OR Net_Income IS NOT NULL OR EPS IS NOT NULL)
            AND (Revenue IS NULL OR Net_Income IS NULL OR EPS IS NULL)
            AND Symbol = ?
        """, (ticker,))
        partial_rows = cursor.fetchall()

        for row in partial_rows:
            print("rows", row)
            symbol, date, revenue, net_income, eps = row
            updates = []
            params = []
            print("partial rows ",partial_rows)

            print(f"Partial data found for {symbol} in {date}.")

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
                info = stock.info
                trailing_eps = info.get('trailingEps')

                if trailing_eps is not None:
                    print(f"Fetched trailing EPS for {symbol}: {trailing_eps}")
                    eps = trailing_eps  # Use the fetched trailing EPS
                    updates.append("EPS = ?")
                    params.append(eps)
                else:
                    # Prompt for manual input if trailing EPS is not available
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
                print(f"Data updated for {symbol} in {date}.")



#end of data_fetcher.py
import sqlite3
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Database path - use the absolute path to ensure accuracy
DB_PATH = '/Users/nicholasdaly/PycharmProjects/Stock Finances/Stock Data.db'


def establish_database_connection(db_path):
    """
    Establishes a connection to the SQLite database.
    """
    if not os.path.exists(db_path):
        logging.error(f"Database file not found: {db_path}")
        return None

    try:
        conn = sqlite3.connect(db_path)
        logging.info("Database connection established.")
        return conn
    except sqlite3.Error as e:
        logging.error(f"Failed to connect to database: {e}")
        return None


def verify_table_exists(cursor, table_name):
    """
    Verifies that the specified table exists in the database.
    """
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table_name,))
    return cursor.fetchone() is not None


def apply_stock_split(ticker, split_ratio, db_path):
    """
    Apply a stock split to EPS data in the database.

    :param ticker: The stock ticker symbol to update.
    :param split_ratio: The ratio of the split (e.g., 2 for a 2-for-1 split, 0.5 for a 1-for-2 reverse split).
    :param db_path: The path to the SQLite database.
    """
    conn = establish_database_connection(db_path)
    if conn is None:
        logging.error("Could not establish a database connection.")
        return

    try:
        cursor = conn.cursor()

        # Verify that the 'Annual_Data' table exists
        if not verify_table_exists(cursor, 'Annual_Data'):
            logging.error("Table 'Annual_Data' does not exist in the database.")
            return

        # Fetch current EPS values for the specified ticker
        cursor.execute("SELECT Date, EPS FROM Annual_Data WHERE Symbol = ?", (ticker,))
        records = cursor.fetchall()

        # Check if any records were found
        if not records:
            logging.warning(f"No records found for ticker '{ticker}' in the database.")
            return

        # Apply split to each EPS value
        for date, eps in records:
            if eps is not None:
                new_eps = eps / split_ratio  # Adjust EPS by the split ratio
                cursor.execute("UPDATE Annual_Data SET EPS = ? WHERE Symbol = ? AND Date = ?", (new_eps, ticker, date))
                logging.info(f"Updated EPS for {ticker} on {date}: {eps} -> {new_eps}")

        # Commit changes to the database
        conn.commit()
        logging.info(f"Stock split applied successfully for ticker '{ticker}' with a split ratio of {split_ratio}.")

    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
    finally:
        # Close the database connection
        conn.close()
        logging.info("Database connection closed.")


# Enter the ticker and split ratio here
ticker = "NVDA"  # Replace with the actual ticker
split_ratio = 10  # Replace with the actual split ratio (e.g., 2 for a 2-for-1 split)

# Run the function to apply the stock split
apply_stock_split(ticker, split_ratio, DB_PATH)

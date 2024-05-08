import sqlite3
import yfinance as yf
import os

# Constants (adjust these paths according to your setup)
TICKERS_FILE_PATH = 'tickers.csv'  # File containing a list of tickers
DB_PATH = 'Stock Data.db'
TABLE_NAME = 'Tickers_Info'  # Adjust this name to your desired table name

# Function to establish a database connection
def establish_database_connection(db_path):
    db_full_path = os.path.abspath(db_path)
    return sqlite3.connect(db_full_path)

# Function to read tickers from a file
def read_tickers(file_path):
    with open(file_path, 'r') as file:
        tickers = [line.strip() for line in file.readlines()]
    return tickers

# Function to create a table if it doesn't exist
def create_tickers_table(cursor, table_name):
    cursor.execute(f'''
    CREATE TABLE IF NOT EXISTS {table_name} (
        ticker TEXT UNIQUE PRIMARY KEY,
        short_name TEXT,
        nicks_growth_rate REAL,
        FINVIZ_5yr_gwth REAL
    )
    ''')

# Function to fetch and store short names
def fetch_and_store_short_names(tickers, cursor, table_name):
    for ticker in tickers:
        try:
            # Fetch the short name from Yahoo Finance
            stock = yf.Ticker(ticker)
            short_name = stock.info.get('shortName', 'N/A')

            # Insert or update the record
            cursor.execute(f'''
            INSERT OR REPLACE INTO {table_name} (ticker, short_name)
            VALUES (?, ?)
            ''', (ticker, short_name))

            print(f"Stored short name for {ticker}: {short_name}")
        except Exception as e:
            print(f"Error fetching or storing data for {ticker}: {e}")

# Main function to execute the standalone script
def main():
    # Establish a database connection
    conn = establish_database_connection(DB_PATH)
    cursor = conn.cursor()

    # Create the tickers table if it doesn't already exist
    create_tickers_table(cursor, TABLE_NAME)

    # Read tickers from the file
    tickers = read_tickers(TICKERS_FILE_PATH)

    # Fetch and store the short names for all tickers
    fetch_and_store_short_names(tickers, cursor, TABLE_NAME)

    # Commit changes and close the connection
    conn.commit()
    conn.close()

if __name__ == "__main__":
    main()

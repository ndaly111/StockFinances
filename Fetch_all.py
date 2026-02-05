import pandas as pd
import sqlite3
import yfinance


def fetch_and_store_all_data(ticker, cursor):
    print(f"Fetching data for {ticker}")
    stock = yfinance.Ticker(ticker)

    # Fetch annual financial data
    try:
        financials = stock.financials.T
        financials['Date'] = financials.index.strftime('%Y-%m-%d')  # Use full date instead of just year

        # Prepare annual financial data rows for bulk insert
        rows_to_insert = [
            (
                ticker,
                row['Date'],
                row.get('Total Revenue'),
                row.get('Net Income'),
                row.get('Basic EPS')
            )
            for _, row in financials.iterrows()
        ]

        insert_sql = """
                INSERT INTO Annual_Data (Symbol, Date, Revenue, Net_Income, EPS, Last_Updated)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(Symbol, Date) DO UPDATE SET
                Revenue = excluded.Revenue,
                Net_Income = excluded.Net_Income,
                EPS = excluded.EPS,
                Last_Updated = CURRENT_TIMESTAMP;
            """

        cursor.executemany(insert_sql, rows_to_insert)
    except Exception as e:
        print(f"Error fetching/storing annual data for {ticker}: {e}")

    # Fetch and store TTM data
    try:
        quarterly_fin = stock.quarterly_financials
        if quarterly_fin is None or quarterly_fin.empty:
            print(f"No quarterly financials data for {ticker}")
        else:
            # Safely sum available quarters (up to 4)
            num_quarters = min(4, len(quarterly_fin.columns))
            ttm_data = quarterly_fin.iloc[:, :num_quarters].sum(axis=1)

            # Safely extract quarter end date
            try:
                quarter_end_date = quarterly_fin.columns[0].strftime('%Y-%m-%d')
            except (IndexError, AttributeError):
                quarter_end_date = None

            if quarter_end_date:
                cursor.execute("""
                    INSERT INTO TTM_Data (Symbol, TTM_Revenue, TTM_Net_Income, TTM_EPS, Quarter, Last_Updated)
                    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(Symbol, Quarter) DO UPDATE SET
                    TTM_Revenue = excluded.TTM_Revenue,
                    TTM_Net_Income = excluded.TTM_Net_Income,
                    TTM_EPS = excluded.TTM_EPS,
                    Last_Updated = CURRENT_TIMESTAMP;
                """, (ticker, ttm_data.get('Total Revenue'), ttm_data.get('Net Income'), ttm_data.get('Basic EPS'), quarter_end_date))
                cursor.connection.commit()
    except Exception as e:
        print(f"Error fetching/storing TTM data for {ticker}: {e}")

db_path = "Stock Data.db"
tickers_file = "tickers.csv"

# Connect to the database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Read tickers from CSV
tickers = pd.read_csv(tickers_file)['Ticker'].tolist()

for ticker in tickers:
    fetch_and_store_all_data(ticker, cursor)

# Commit changes and close the connection
conn.commit()
conn.close()
print("All data fetched and stored.")



#main_remote()
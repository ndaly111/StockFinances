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

        # Store annual financial data
        for index, row in financials.iterrows():
            cursor.execute("""
                INSERT INTO Annual_Data (Symbol, Date, Revenue, Net_Income, EPS, Last_Updated)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(Symbol, Date) DO UPDATE SET
                Revenue = excluded.Revenue,
                Net_Income = excluded.Net_Income,
                EPS = excluded.EPS,
                Last_Updated = CURRENT_TIMESTAMP;
            """, (ticker, row['Date'], row.get('Total Revenue'), row.get('Net Income'), row.get('Basic EPS')))
    except Exception as e:
        print(f"Error fetching/storing annual data for {ticker}: {e}")

    # Fetch and store TTM data
    try:
        ttm_data = stock.quarterly_financials.iloc[:, :4].sum(axis=1)
        quarter_end_date = stock.quarterly_financials.columns[0].strftime('%Y-%m-%d')

        cursor.execute("""
            INSERT INTO TTM_Data (Symbol, TTM_Revenue, TTM_Net_Income, TTM_EPS, Quarter, Last_Updated)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(Symbol, Quarter) DO UPDATE SET
            TTM_Revenue = excluded.TTM_Revenue,
            TTM_Net_Income = excluded.TTM_Net_Income,
            TTM_EPS = excluded.TTM_EPS,
            Last_Updated = CURRENT_TIMESTAMP;
        """, (ticker, ttm_data.get('Total Revenue'), ttm_data.get('Net Income'), ttm_data.get('Basic EPS'), quarter_end_date))
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
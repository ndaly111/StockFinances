import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3
from datetime import datetime






def scrape_annual_estimates(ticker,table_id):
    print("forward_data 1 scrape annual estimates")
    ticker = ticker.replace("-", ".")
    print("---ticker", ticker)
    url = f'https://fintel.io/sfo/us/{ticker}'  # URL formatted with the ticker variable
    print("---url",url)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.select_one(f'#{table_id}')
        print("---fetching table")

        if not table:
            print(f"Table with ID #{table_id} not found.")
            return pd.DataFrame()  # Return an empty DataFrame

        # Extracting headers and rows
        headers = [th.get_text(strip=True) for th in table.find_all('th')]
        rows = []
        for tr in table.find_all('tr'):
            cols = [td.get_text(strip=True) for td in tr.find_all('td')]
            if cols:  # This ensures that header rows are not included
                rows.append(cols)

        # Convert to DataFrame
        df = pd.DataFrame(rows, columns=headers)


        # Display the headers to check if they match the expected column names
        print("Headers found in table:")


        # Assuming 'Date' is the correct column name for the years
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  # Handle any parsing errors
        df = df.dropna(subset=['Date'])  # Drop rows where 'Date' could not be parsed

        # Filter to keep only rows where 'Date' is within the next three years
        current_year = pd.Timestamp.now().year
        future_years = [current_year + i for i in range(0, 4)]
        df = df[df['Date'].dt.year.isin(future_years)]

        # Assuming that 'Annual' is part of the column name for annual revenue
        annual_revenue_cols = [col for col in df.columns if 'Annual' in col]
        print("Annual revenue columns found:")
        print(annual_revenue_cols)

        # Filter out the rows where annual revenue is not available ('--' or empty)
        for col in annual_revenue_cols:
            df = df[df[col] != '--']
            df = df[df[col].astype(str).str.strip().astype(bool)]
            # Convert Revenue from string in millions to float, then multiply by 1 million

        # Convert revenue in millions if processing revenue table
        if table_id == 'revenue':
            df['Revenue Average Annually (MM)'] = df['Revenue Average Annually (MM)'].replace(
                {'\$': '', ',': '', '--': None}, regex=True).astype(float) * 1e6

        # Return the DataFrame with the first three consecutive years' data
        print("---final dataframe",df)
        return df.iloc[:3] if len(df) >= 3 else df
    else:
        print(f"Failed to retrieve data, status code: {response.status_code}")
        return pd.DataFrame()  # Return an empty DataFrame


def scrape_and_prepare_data(ticker):
    eps_df = scrape_annual_estimates(ticker, 'eps')
    revenue_df = scrape_annual_estimates(ticker, 'revenue')

    # Check if both DataFrames are empty, and if so, return an empty DataFrame immediately
    if eps_df.empty and revenue_df.empty:
        print(f"No forecast data found for {ticker}.")
        return pd.DataFrame()

    try:
        # Preparing DataFrames for merging
        if not eps_df.empty:
            eps_df.set_index('Date', inplace=True)
        if not revenue_df.empty:
            revenue_df.set_index('Date', inplace=True)

        # Merging DataFrames, considering that one or both might be empty
        if not eps_df.empty and not revenue_df.empty:
            combined_df = eps_df.merge(revenue_df, left_index=True, right_index=True, how='outer')
        elif not eps_df.empty:
            combined_df = eps_df
        else:
            combined_df = revenue_df

        print(f"Combined DataFrame for {ticker}:")
        print(combined_df)
    except Exception as e:
        print(f"Error preparing data for {ticker}: {e}")
        return pd.DataFrame()

    return combined_df.reset_index()  # Resetting index to bring 'Date' back as a column


def store_in_database(df, ticker, db_path, table_name):
    print(f"Storing combined data in database for {ticker}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Special rule for BRK.B or BRK-B tickers
    eps_scale_factor = 1500 if ticker in ['BRK.B', 'BRK-B'] else 1
    print("---scale factor", eps_scale_factor)

    for _, row in df.iterrows():
        date_str = row['Date'].strftime('%Y-%m-%d')
        revenue = row.get('Revenue Average Annually (MM)', 0)

        # Convert EPS to a float, handling any conversion errors
        try:
            eps_raw = row.get('EPS Average (Annual)', '0')  # Default to '0' if key not found
            eps_raw = eps_raw.replace(',', '')  # Remove commas
            eps = float(eps_raw) / eps_scale_factor
        except ValueError as e:
            print(f"Error converting EPS: {e}")
            print(f"Warning: Unable to convert EPS value '{eps_raw}' to float for {ticker} on {date_str}. Defaulting to 0.")
            eps = 0.0

        eps_analysts = row.get('Number of Analysts (Annually)_x', 0)
        revenue_analysts = row.get('Number of Analysts (Annually)_y', 0)
        last_updated = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        insert_query = f'''
        INSERT INTO {table_name} (Ticker, Date, ForwardEPS, ForwardRevenue, LastUpdated, ForwardEPSAnalysts, ForwardRevenueAnalysts)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(Ticker, Date) DO UPDATE SET
            ForwardRevenue = EXCLUDED.ForwardRevenue,
            ForwardEPS = EXCLUDED.ForwardEPS,
            ForwardEPSAnalysts = EXCLUDED.ForwardEPSAnalysts,
            ForwardRevenueAnalysts = EXCLUDED.ForwardRevenueAnalysts,
            LastUpdated = EXCLUDED.LastUpdated;
        '''

        cursor.execute(insert_query, (
            ticker,
            date_str,
            eps,
            revenue,
            last_updated,
            eps_analysts,
            revenue_analysts
        ))

    conn.commit()
    conn.close()
    print("Data stored successfully.")



# Example usage
#ticker = 'AAPL'  # Replace with any ticker
#db_path = 'Stock Data.db'  # Ensure the path is correct
#table_name = 'ForwardFinancialData'

#combined_df = scrape_and_prepare_data(ticker)

#if not combined_df.empty:
    #store_in_database(combined_df, ticker, db_path, table_name)



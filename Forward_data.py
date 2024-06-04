import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3
from datetime import datetime

def update_database_schema(db_path, table_name):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check if 'ForwardEPSAnalysts' and 'ForwardRevenueAnalysts' columns exist and add them if not
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [column[1] for column in cursor.fetchall()]
    if 'ForwardEPSAnalysts' not in columns:
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN ForwardEPSAnalysts INTEGER")
    if 'ForwardRevenueAnalysts' not in columns:
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN ForwardRevenueAnalysts INTEGER")
    conn.commit()
    conn.close()

def scrape_annual_estimates(ticker):
    print("Scraping annual estimates from Zacks")
    ticker = ticker.replace('-', '.')
    url = f'https://www.zacks.com/stock/quote/{ticker}/detailed-earning-estimates'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        return pd.DataFrame()
    except Exception as err:
        print(f"Other error occurred: {err}")
        return pd.DataFrame()

    soup = BeautifulSoup(response.content, 'html.parser')

    sales_table = soup.select_one('#detailed_estimate_full_body #detailed_earnings_estimates:nth-of-type(1) table')
    earnings_table = soup.select_one('#detailed_estimate_full_body #detailed_earnings_estimates:nth-of-type(2) table')

    if not sales_table or not earnings_table:
        print("Data tables not found.")
        return pd.DataFrame()

    def extract_data(table):
        headers = [th.get_text(strip=True) for th in table.find_all('th')]
        rows = table.find_all('tr')[1:]
        data = []
        for row in rows:
            cells = row.find_all('td')
            if len(cells) != len(headers):
                continue
            data.append([cell.get_text(strip=True) for cell in cells])
        return pd.DataFrame(data, columns=headers)

    sales_df = extract_data(sales_table)
    earnings_df = extract_data(earnings_table)

    combined_df = pd.concat([sales_df, earnings_df], axis=1)
    combined_df.columns = ['Year', 'Revenue', 'RevenueAnalysts', 'EPS', 'EPSAnalysts']

    # Replace NaN values with a default value
    combined_df.fillna(0, inplace=True)

    combined_df['Revenue'] = combined_df['Revenue'].apply(lambda x: float(str(x).replace('B', '').replace('M', '')) * (10**9 if 'B' in x else 10**6))
    combined_df['EPS'] = combined_df['EPS'].astype(float, errors='ignore')
    combined_df['ForwardRevenueAnalysts'] = combined_df['RevenueAnalysts'].astype(int, errors='ignore')
    combined_df['ForwardEPSAnalysts'] = combined_df['EPSAnalysts'].astype(int, errors='ignore')

    print("Scraped DataFrame:", combined_df)
    return combined_df

def scrape_and_prepare_data(ticker):
    data_df = scrape_annual_estimates(ticker)
    if data_df.empty:
        print(f"No forecast data found for {ticker}.")
        return pd.DataFrame()
    print(f"Prepared DataFrame for {ticker}:")
    print(data_df)
    return data_df.reset_index(drop=True)

def store_in_database(df, ticker, db_path, table_name):
    print(f"Storing data in database for {ticker}")
    update_database_schema(db_path, table_name)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    for _, row in df.iterrows():
        date_str = row['Year']
        if pd.isna(date_str):
            continue  # Skip rows with no valid date
        revenue = row['Revenue']
        eps = row['EPS']
        revenue_analysts = row['ForwardRevenueAnalysts']
        eps_analysts = row['ForwardEPSAnalysts']
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

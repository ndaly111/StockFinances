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
        rows = []
        for tr in table.find_all('tr'):
            cols = [td.get_text(strip=True) for td in tr.find_all('td')]
            if cols:
                rows.append(cols)
        df = pd.DataFrame(rows, columns=headers)
        df.replace('NA', pd.NA, inplace=True)
        return df.dropna(how='all')

    sales_df = extract_data(sales_table)
    earnings_df = extract_data(earnings_table)

    def get_estimates_and_counts(df):
        estimates = df[df.iloc[:, 0].str.contains("Zacks Consensus Estimate", na=False)].iloc[:, -2:].values.flatten()
        counts = df[df.iloc[:, 0].str.contains("# of Estimates", na=False)].iloc[:, -2:].values.flatten()
        return estimates, counts

    sales_estimates, sales_counts = get_estimates_and_counts(sales_df)
    earnings_estimates, earnings_counts = get_estimates_and_counts(earnings_df)

    headers = [th.get_text(strip=True) for th in sales_table.find_all('th') if "Year" in th.get_text()]

    def convert_to_date(header):
        parts = header.split('(')[-1].strip(')').split('/')
        if len(parts) != 2:
            return None
        year = parts[1]
        month = parts[0]
        last_days = {
            '01': '31', '02': '28', '03': '31', '04': '30', '05': '31', '06': '30',
            '07': '31', '08': '31', '09': '30', '10': '31', '11': '30', '12': '31'
        }
        day = last_days[month.zfill(2)]
        return f"{year}-{month.zfill(2)}-{day}"

    headers = [convert_to_date(header) for header in headers[-2:] if header and header != 'ND']

    # Debugging prints
    print("Sales Estimates Length:", len(sales_estimates))
    print("Earnings Estimates Length:", len(earnings_estimates))
    print("Sales Counts Length:", len(sales_counts))
    print("Earnings Counts Length:", len(earnings_counts))
    print("Headers Length:", len(headers))

    if len(headers) == 0:
        print("No valid headers found.")
        return pd.DataFrame()

    # Ensure all arrays are the same length
    min_length = min(len(headers), len(sales_estimates), len(earnings_estimates), len(sales_counts), len(earnings_counts))
    headers = headers[:min_length]
    sales_estimates = sales_estimates[:min_length]
    earnings_estimates = earnings_estimates[:min_length]
    sales_counts = sales_counts[:min_length]
    earnings_counts = earnings_counts[:min_length]

    combined_df = pd.DataFrame({
        'Year': headers,
        'Revenue': sales_estimates,
        'EPS': earnings_estimates,
        'ForwardRevenueAnalysts': sales_counts,
        'ForwardEPSAnalysts': earnings_counts
    })

    print("Combined DataFrame before conversion:", combined_df)

    def convert_to_float(value):
        if pd.isna(value):
            return pd.NA
        try:
            if 'B' in value:
                return float(value.replace('B', '')) * 1e9
            elif 'M' in value:
                return float(value.replace('M', '')) * 1e6
            else:
                return float(value)
        except:
            return pd.NA

    combined_df['Revenue'] = combined_df['Revenue'].apply(convert_to_float)
    combined_df['EPS'] = combined_df['EPS'].apply(convert_to_float)
    combined_df['ForwardRevenueAnalysts'] = combined_df['ForwardRevenueAnalysts'].astype(int, errors='ignore')
    combined_df['ForwardEPSAnalysts'] = combined_df['ForwardEPSAnalysts'].astype(int, errors='ignore')

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

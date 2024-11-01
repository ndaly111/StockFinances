import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3
from datetime import datetime
import logging
import calendar
from typing import Optional

logging.basicConfig(level=logging.INFO)


def update_database_schema(db_path: str, table_name: str) -> None:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [column[1] for column in cursor.fetchall()]
    if 'ForwardEPSAnalysts' not in columns:
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN ForwardEPSAnalysts INTEGER")
    if 'ForwardRevenueAnalysts' not in columns:
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN ForwardRevenueAnalysts INTEGER")
    conn.commit()
    conn.close()


def extract_data(table) -> pd.DataFrame:
    headers = [th.get_text(strip=True) for th in table.find_all('th')]
    rows = table.find_all('tr')[1:]
    data = []
    for row in rows:
        cells = row.find_all('td')
        if len(cells) != len(headers):
            continue
        data.append([cell.get_text(strip=True) for cell in cells])
    return pd.DataFrame(data, columns=headers)


def get_last_day_of_month(date_str: str) -> str:
    month, year = date_str.split('/')
    year = int(year)
    month = int(month)
    last_day = calendar.monthrange(year, month)[1]
    return f"{year}-{month:02d}-{last_day:02d}"


def scrape_annual_estimates(ticker: str) -> pd.DataFrame:
    logging.info("Scraping annual estimates from Zacks")
    ticker = ticker.replace('-', '.')
    url = f'https://www.zacks.com/stock/quote/{ticker}/detailed-earnings-estimates'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTP error occurred: {http_err}")
        return pd.DataFrame()
    except Exception as err:
        logging.error(f"Other error occurred: {err}")
        return pd.DataFrame()

    soup = BeautifulSoup(response.content, 'html.parser')

    sales_table = soup.select_one('#detailed_estimate_full_body #detailed_earnings_estimates:nth-of-type(1) table')
    earnings_table = soup.select_one('#detailed_estimate_full_body #detailed_earnings_estimates:nth-of-type(2) table')

    if not sales_table or not earnings_table:
        logging.warning("Data tables not found.")
        return pd.DataFrame()

    sales_df = extract_data(sales_table)
    earnings_df = extract_data(earnings_table)

    combined_df = pd.concat([sales_df, earnings_df], axis=1)

    logging.info(f"Columns before renaming: {combined_df.columns}")

    combined_df.columns = [
        'Period', 'CurrentQtr_Revenue', 'NextQtr_Revenue', 'CurrentYear_Revenue', 'NextYear_Revenue',
        'Period2', 'CurrentQtr_EPS', 'NextQtr_EPS', 'CurrentYear_EPS', 'NextYear_EPS'
    ]

    combined_df = combined_df[['Period', 'CurrentYear_Revenue', 'NextYear_Revenue', 'CurrentYear_EPS', 'NextYear_EPS']]

    combined_df.fillna(0, inplace=True)

    def convert_to_numeric(value: str) -> float:
        if isinstance(value, str):
            value = value.replace('B', '').replace('M', '')
            try:
                return float(value) * (10 ** 9 if 'B' in value else 10 ** 6)
            except ValueError:
                return 0
        return value

    combined_df['CurrentYear_Revenue'] = combined_df['CurrentYear_Revenue'].apply(convert_to_numeric)
    combined_df['NextYear_Revenue'] = combined_df['NextYear_Revenue'].apply(convert_to_numeric)
    combined_df['CurrentYear_EPS'] = combined_df['CurrentYear_EPS'].astype(float, errors='ignore')
    combined_df['NextYear_EPS'] = combined_df['NextYear_EPS'].astype(float, errors='ignore')

    combined_df['CurrentYear_Date'] = combined_df['Period'].apply(
        lambda x: get_last_day_of_month(x.split('(')[-1][:-1]) if isinstance(x, str) and '/' in x else None)

    if 'Period2' in combined_df.columns:
        combined_df['NextYear_Date'] = combined_df['Period2'].apply(
            lambda x: get_last_day_of_month(x.split('(')[-1][:-1]) if isinstance(x, str) and '/' in x else None)
    else:
        combined_df['NextYear_Date'] = None

    logging.info("Scraped DataFrame:\n%s", combined_df)
    return combined_df


def scrape_and_prepare_data(ticker: str) -> pd.DataFrame:
    data_df = scrape_annual_estimates(ticker)
    if data_df.empty:
        logging.info(f"No forecast data found for {ticker}.")
        return pd.DataFrame()
    logging.info(f"Prepared DataFrame for {ticker}:\n%s", data_df)
    return data_df.reset_index(drop=True)


def store_in_database(df: pd.DataFrame, ticker: str, db_path: str, table_name: str) -> None:
    logging.info(f"Storing data in database for {ticker}")
    update_database_schema(db_path, table_name)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    for _, row in df.iterrows():
        current_year_date: Optional[str] = row['CurrentYear_Date']
        next_year_date: Optional[str] = row['NextYear_Date']
        if current_year_date is None or next_year_date is None:
            logging.warning(f"Skipping row due to missing date: {row}")
            continue
        revenue_current: float = row['CurrentYear_Revenue']
        revenue_next: float = row['NextYear_Revenue']
        eps_current: float = row['CurrentYear_EPS']
        eps_next: float = row['NextYear_EPS']
        revenue_analysts = 0  # Placeholder as it's not clear from the data
        eps_analysts = 0  # Placeholder as it's not clear from the data
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
            current_year_date,
            eps_current,
            revenue_current,
            last_updated,
            eps_analysts,
            revenue_analysts
        ))
        cursor.execute(insert_query, (
            ticker,
            next_year_date,
            eps_next,
            revenue_next,
            last_updated,
            eps_analysts,
            revenue_analysts
        ))
    conn.commit()
    conn.close()
    logging.info("Data stored successfully.")

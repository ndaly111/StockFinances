import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3
from datetime import datetime
import logging
import calendar
from typing import Optional

logging.basicConfig(level=logging.INFO)

# -----------------------------
# Database Schema Update
# -----------------------------
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

# -----------------------------
# Helper Functions
# -----------------------------
def extract_data(table) -> pd.DataFrame:
    headers = [th.get_text(strip=True) for th in table.find_all('th')]
    rows = table.find_all('tr')[1:]
    data = []
    for row in rows:
        cells = row.find_all('td')
        if len(cells) != len(headers):
            continue
        data.append([cell.get_text(strip=True) for cell in cells])
    df = pd.DataFrame(data, columns=headers)
    logging.info(f"Extracted DataFrame columns: {df.columns}")
    logging.info(f"Extracted DataFrame head:\n{df.head()}")
    return df

def get_last_day_of_month(date_str: str) -> str:
    month, year = date_str.split('/')
    year = int(year)
    month = int(month)
    last_day = calendar.monthrange(year, month)[1]
    return f"{year}-{month:02d}-{last_day:02d}"

def convert_to_numeric(value: str) -> float:
    logging.info(f"Converting value: {value}")
    if isinstance(value, str):
        value = value.replace(',', '').strip()
        multiplier = 1
        if 'B' in value:
            multiplier = 10**9
            value = value.replace('B', '')
        elif 'M' in value:
            multiplier = 10**6
            value = value.replace('M', '')
        elif 'T' in value:
            multiplier = 10**12
            value = value.replace('T', '')
        try:
            return float(value) * multiplier
        except ValueError:
            logging.warning(f"Value conversion error for: {value}")
            return 0.0
    return float(value)

# -----------------------------
# Scrape Consensus Estimates
# -----------------------------
def scrape_annual_estimates_from_web(ticker: str) -> pd.DataFrame:
    logging.info(f"Scraping annual estimates from Zacks for ticker: {ticker}")
    ticker = ticker.replace('-', '.')
    url = f'https://www.zacks.com/stock/quote/{ticker}/detailed-earning-estimates'
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
    sections = soup.find_all('section', id='detailed_earnings_estimates')
    if len(sections) < 2:
        logging.warning("Relevant sections not found.")
        return pd.DataFrame()

    sales_section = sections[0]
    earnings_section = sections[1]

    sales_table = sales_section.find('table')
    earnings_table = earnings_section.find('table')
    if not sales_table or not earnings_table:
        logging.warning("Relevant tables not found.")
        return pd.DataFrame()

    sales_df = extract_data(sales_table)
    earnings_df = extract_data(earnings_table)

    logging.info(f"Sales DataFrame columns: {sales_df.columns}")
    logging.info(f"Earnings DataFrame columns: {earnings_df.columns}")
    if sales_df.empty or earnings_df.empty:
        logging.warning("No relevant data found in the extracted tables.")
        return pd.DataFrame()

    # Extract analyst counts from rows containing "# of Estimates"
    revenue_analysts_row = sales_df[sales_df[sales_df.columns[0]].str.contains('# of Estimates', case=False, na=False)]
    eps_analysts_row = earnings_df[earnings_df[earnings_df.columns[0]].str.contains('# of Estimates', case=False, na=False)]
    if revenue_analysts_row.empty or eps_analysts_row.empty:
        logging.warning("Analysts count not found.")
        return pd.DataFrame()

    revenue_analysts = revenue_analysts_row.iloc[0, 1].replace(',', '')
    eps_analysts = eps_analysts_row.iloc[0, 1].replace(',', '')
    if not revenue_analysts.isdigit() or not eps_analysts.isdigit():
        logging.warning("Analysts count is not numeric.")
        return pd.DataFrame()
    revenue_analysts = int(revenue_analysts)
    eps_analysts = int(eps_analysts)

    # Determine the forecast periods (assuming columns at index 3 and 4 contain the period info)
    current_year_period = sales_df.columns[3]
    next_year_period = sales_df.columns[4]
    current_year_date = get_last_day_of_month(current_year_period.split('(')[-1][:-1])
    next_year_date = get_last_day_of_month(next_year_period.split('(')[-1][:-1])

    # Helper function for safe numeric conversion
    def safe_convert(value):
        try:
            return float(value.replace(',', ''))
        except ValueError:
            return None

    # Filter to get only the consensus row from each DataFrame
    consensus_revenue_row = sales_df[sales_df[sales_df.columns[0]].str.contains('Consensus', case=False, na=False)]
    consensus_eps_row = earnings_df[earnings_df[earnings_df.columns[0]].str.contains('Consensus', case=False, na=False)]
    if consensus_revenue_row.empty or consensus_eps_row.empty:
        logging.warning("Consensus estimates not found.")
        return pd.DataFrame()

    # Extract consensus estimates (assuming current year is in column index 3 and next year in index 4)
    current_year_revenue = convert_to_numeric(consensus_revenue_row.iloc[0, 3])
    next_year_revenue = convert_to_numeric(consensus_revenue_row.iloc[0, 4])
    current_year_eps = safe_convert(consensus_eps_row.iloc[0, 3])
    next_year_eps = safe_convert(consensus_eps_row.iloc[0, 4])

    # Build a DataFrame with two rows (current and next year) using the consensus estimates.
    # Note: The updated DataFrame will have a single 'Date' column for each row.
    data = [
        {
            'Period': 'Current Year',
            'Date': current_year_date,
            'ForwardRevenue': current_year_revenue,
            'ForwardEPS': current_year_eps,
            'ForwardRevenueAnalysts': revenue_analysts,
            'ForwardEPSAnalysts': eps_analysts
        },
        {
            'Period': 'Next Year',
            'Date': next_year_date,
            'ForwardRevenue': next_year_revenue,
            'ForwardEPS': next_year_eps,
            'ForwardRevenueAnalysts': revenue_analysts,
            'ForwardEPSAnalysts': eps_analysts
        }
    ]
    combined_df = pd.DataFrame(data)
    logging.info("Scraped Consensus Estimates DataFrame:\n%s", combined_df)
    return combined_df

# -----------------------------
# Store Data in Database
# -----------------------------
def store_in_database(df: pd.DataFrame, ticker: str, db_path: str, table_name: str) -> None:
    """
    Inserts the scraped consensus estimates into the database.
    Expects the DataFrame to have the following columns:
      'Date', 'ForwardRevenue', 'ForwardEPS', 'ForwardRevenueAnalysts', 'ForwardEPSAnalysts'
    Inserts records into the table with columns:
      Ticker, Date, ForwardEPS, ForwardRevenue, LastUpdated, ForwardEPSAnalysts, ForwardRevenueAnalysts
    """
    logging.info(f"Storing data in database for {ticker}")
    update_database_schema(db_path, table_name)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Optional: Delete existing records for this ticker (legacy code deletes by ticker)
    today = datetime.now().strftime('%Y-%m-%d')
    cursor.execute(f"DELETE FROM {table_name} WHERE Ticker = ? AND Date < ?", (ticker, today))
    cursor.execute(f"DELETE FROM {table_name} WHERE Ticker = ?", (ticker,))

    # Iterate through each row of the new DataFrame
    for _, row in df.iterrows():
        date_val: Optional[str] = row['Date']
        if not date_val:
            logging.warning(f"Skipping row due to missing date: {row}")
            continue
        revenue: float = row['ForwardRevenue']
        eps: float = row['ForwardEPS']
        revenue_analysts: int = row['ForwardRevenueAnalysts']
        eps_analysts: int = row['ForwardEPSAnalysts']
        last_updated = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        data = (
            ticker,
            date_val,
            eps,
            revenue,
            last_updated,
            eps_analysts,
            revenue_analysts
        )
        logging.info(f"Inserting data: {data}")
        insert_query = f'''
            INSERT INTO {table_name} (Ticker, Date, ForwardEPS, ForwardRevenue, LastUpdated, ForwardEPSAnalysts, ForwardRevenueAnalysts)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(Ticker, Date) DO UPDATE SET
                ForwardEPS = EXCLUDED.ForwardEPS,
                ForwardRevenue = EXCLUDED.ForwardRevenue,
                ForwardEPSAnalysts = EXCLUDED.ForwardEPSAnalysts,
                ForwardRevenueAnalysts = EXCLUDED.ForwardRevenueAnalysts,
                LastUpdated = EXCLUDED.LastUpdated;
        '''
        cursor.execute(insert_query, data)
    conn.commit()
    conn.close()
    logging.info("Data stored successfully.")

def scrape_forward_data(ticker: str, db_path: str, table_name: str) -> None:
    data_df = scrape_annual_estimates_from_web(ticker)
    if data_df.empty:
        logging.info(f"No forecast data found for {ticker}.")
    else:
        logging.info(f"Prepared DataFrame for {ticker}:\n{data_df}")
        store_in_database(data_df, ticker, db_path, table_name)

# -----------------------------
# Example usage (Legacy Code)
# -----------------------------
ticker = 'AAPL'
db_path = 'Stock Data.db'
table_name = 'ForwardFinancialData'
# Uncomment the line below to run the scraping and database insertion
# scrape_forward_data(ticker, db_path, table_name)

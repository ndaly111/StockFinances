import yfinance as yf
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import logging
import numpy as np
import matplotlib.pyplot as plt
import os

# Database connection setup
# Database connection setup with indexing
def get_db_connection(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_symbol ON Annual_Data(Symbol);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_symbol_quarter ON TTM_Data(Symbol, Quarter);")
    cursor.connection.commit()
    return conn


def fetch_ticker_data(ticker, cursor):
    try:
        cursor.execute("PRAGMA table_info(Annual_Data)")
        columns = [col[1] for col in cursor.fetchall()]
        cursor.execute("SELECT * FROM Annual_Data WHERE Symbol = ? ORDER BY Date ASC", (ticker,))
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        return results if results else None
    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
        return None

def fetch_ttm_data(ticker, cursor):
    try:
        cursor.execute("PRAGMA table_info(TTM_Data)")
        columns = [col[1] for col in cursor.fetchall()]
        cursor.execute("SELECT * FROM TTM_Data WHERE Symbol = ? ORDER BY Quarter DESC", (ticker,))
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        return results if results else None
    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
        return None

def get_latest_annual_data_date(ticker_data):
    # Check if ticker_data is a pandas DataFrame
    if isinstance(ticker_data, pd.DataFrame):
        if ticker_data.empty:
            logging.warning("Ticker data is an empty DataFrame. Returning None.")
            return None  # Handle empty DataFrame safely
    elif not ticker_data:  # Handle case where ticker_data is a list or other type
        logging.warning("Ticker data is empty or None. Returning None.")
        return None

    # Get the first row for reference
    first_entry = ticker_data.iloc[0] if isinstance(ticker_data, pd.DataFrame) else ticker_data[0]

    if isinstance(first_entry, dict):
        if 'Date' not in first_entry or 'Symbol' not in first_entry:
            logging.warning("Missing expected keys in ticker_data dictionary. Returning None.")
            return None  # Avoid KeyError
        
        try:
            dates = [
                datetime.strptime(row['Date'], '%Y-%m-%d') 
                for row in ticker_data 
                if row.get('Symbol') == first_entry['Symbol']
            ]
        except Exception as e:
            logging.error(f"Error parsing dates: {e}")
            return None

    elif isinstance(first_entry, (list, tuple)) and len(first_entry) > 2:
        try:
            dates = [
                datetime.strptime(row[2], '%Y-%m-%d') 
                for row in ticker_data 
                if row[0] == first_entry[0]
            ]
        except Exception as e:
            logging.error(f"Error parsing dates from list/tuple format: {e}")
            return None

    elif isinstance(ticker_data, pd.DataFrame) and 'Date' in ticker_data.columns:
        try:
            dates = pd.to_datetime(ticker_data['Date'], format='%Y-%m-%d')
            latest_date = dates.max()
            logging.debug(f"Latest annual data date (DataFrame): {latest_date}")
            return latest_date
        except Exception as e:
            logging.error(f"Error parsing dates from DataFrame: {e}")
            return None

    logging.warning("Unexpected data format in ticker_data. Returning None.")
    return None


from datetime import timedelta

def calculate_next_check_date(latest_date, months):
    if latest_date is None:
        return None  # Prevents TypeError if no date is available

    return latest_date + timedelta(days=months * 30)

def needs_update(latest_date, months):
    if latest_date is None:
        return True  # If there's no last update date, we assume an update is needed

    next_check_date = calculate_next_check_date(latest_date, months)
    return next_check_date is None or next_check_date <= datetime.now()

def check_null_fields(data, fields):
    for entry in data:
        for field in fields:
            if entry.get(field) in [None, '']:
                logging.debug(f"Null field found: {field} in entry {entry}")
                return True
    return True

def clean_financial_data(df):
    # Drop rows where all specified columns are NaN
    df.dropna(axis=0, how='all', subset=['Revenue', 'Net_Income', 'EPS'], inplace=True)
    # Fill NaN values using forward fill, then backward fill as a fallback
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df.infer_objects(copy=False)  # Ensure future behavior for downcasting
    return df

from functools import lru_cache

@lru_cache(maxsize=32)
def fetch_annual_data_from_yahoo(ticker):
    logging.info("Fetching annual data from Yahoo Finance")
    try:
        stock = yf.Ticker(ticker)
        financials = stock.financials
        if financials.empty:
            return pd.DataFrame()

        financials = financials.T
        financials['Date'] = financials.index

        column_mapping = {
            'Total Revenue': 'Revenue',
            'Net Income': 'Net_Income',
            'Basic EPS': 'EPS'
        }
        renamed_columns = {yahoo: db for yahoo, db in column_mapping.items() if yahoo in financials.columns}
        if len(renamed_columns) < len(column_mapping):
            missing_columns = set(column_mapping.values()) - set(renamed_columns.values())
            logging.warning(f"Missing required columns for {ticker}: {missing_columns}")
            return pd.DataFrame()

        financials.rename(columns=renamed_columns, inplace=True)
        financials = clean_financial_data(financials)
        return financials
    except Exception as e:
        logging.error(f"Error fetching data from Yahoo Finance for {ticker}: {e}")
        return pd.DataFrame()

@lru_cache(maxsize=32)
def fetch_ttm_data_from_yahoo(ticker):
    logging.info("Fetching TTM data from Yahoo Finance")
    try:
        stock = yf.Ticker(ticker)
        ttm_financials = stock.quarterly_financials
        if ttm_financials is None or ttm_financials.empty:
            logging.info(f"No TTM financial data available for {ticker}.")
            return None

        ttm_data = {}
        try:
            ttm_data['TTM_Revenue'] = ttm_financials.loc['Total Revenue', :].iloc[:4].sum()
            ttm_data['TTM_Net_Income'] = ttm_financials.loc['Net Income', :].iloc[:4].sum()
        except KeyError:
            ttm_data['TTM_Revenue'] = None
            ttm_data['TTM_Net_Income'] = None

        ttm_data['TTM_EPS'] = stock.info.get('trailingEps', None)
        ttm_data['Shares_Outstanding'] = stock.info.get('sharesOutstanding', None)
        ttm_data['Quarter'] = ttm_financials.columns[0].strftime('%Y-%m-%d')
        return ttm_data
    except Exception as e:
        logging.error(f"Error fetching TTM data from Yahoo Finance for {ticker}: {e}")
        return None


def fetch_ttm_data_from_yahoo(ticker):
    logging.info("Fetching TTM data from Yahoo Finance")
    try:
        stock = yf.Ticker(ticker)
        ttm_financials = stock.quarterly_financials
        if ttm_financials is None or ttm_financials.empty:
            logging.info(f"No TTM financial data available for {ticker}.")
            return None

        ttm_data = {}
        try:
            ttm_data['TTM_Revenue'] = ttm_financials.loc['Total Revenue', :].iloc[:4].sum()
            ttm_data['TTM_Net_Income'] = ttm_financials.loc['Net Income', :].iloc[:4].sum()
        except KeyError:
            ttm_data['TTM_Revenue'] = None
            ttm_data['TTM_Net_Income'] = None

        ttm_data['TTM_EPS'] = stock.info.get('trailingEps', None)
        ttm_data['Shares_Outstanding'] = stock.info.get('sharesOutstanding', None)
        ttm_data['Quarter'] = ttm_financials.columns[0].strftime('%Y-%m-%d')
        return ttm_data
    except Exception as e:
        logging.error(f"Error fetching TTM data from Yahoo Finance for {ticker}: {e}")
        return None

def store_annual_data(ticker, annual_data, cursor):
    logging.info("Storing annual data")
    for index, row in annual_data.iterrows():
        date_str = row['Date'].strftime('%Y-%m-%d') if isinstance(row['Date'], pd.Timestamp) else row['Date']
        cursor.execute("""
            SELECT * FROM Annual_Data
            WHERE Symbol = ? AND Date = ? AND Revenue IS NOT NULL AND Net_Income IS NOT NULL AND EPS IS NOT NULL;
        """, (ticker, date_str))
        existing_row = cursor.fetchone()
        if existing_row:
            continue
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
            """, (ticker, date_str, row['Revenue'], row['Net_Income'], row['EPS']))
            cursor.connection.commit()
        except sqlite3.Error as e:
            logging.error(f"Database error while storing/updating annual data for {ticker}: {e}")

def store_ttm_data(ticker, ttm_data, cursor):
    logging.info("Storing TTM data")
    ttm_values = (
        ticker,
        ttm_data['TTM_Revenue'],
        ttm_data['TTM_Net_Income'],
        ttm_data['TTM_EPS'],
        ttm_data.get('Shares_Outstanding'),
        ttm_data['Quarter']
    )
    try:
        cursor.execute("""
            INSERT OR REPLACE INTO TTM_Data (Symbol, TTM_Revenue, TTM_Net_Income, TTM_EPS, Shares_Outstanding, Quarter, Last_Updated)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP);
        """, ttm_values)
        cursor.connection.commit()
    except sqlite3.Error as e:
        logging.error(f"Database error while storing/updating TTM data for {ticker}: {e}")


def handle_ttm_duplicates(ticker, cursor):
    logging.info("Checking for duplicate TTM entries for ticker")
    try:
        # Fetch all rows for the ticker, ordered by date descending (latest first)
        cursor.execute("SELECT * FROM TTM_Data WHERE Symbol = ? ORDER BY Quarter DESC", (ticker,))
        results = cursor.fetchall()

        # Keep the most recent entry and delete the others
        if len(results) > 1:
            logging.info(f"Multiple TTM entries detected for {ticker}")

            # Keep the most recent entry (first one due to ORDER BY DESC)
            most_recent_entry = results[0]
            most_recent_quarter = most_recent_entry[5]  # Assuming Quarter is the 6th column (index 5)

            # Delete all other entries except the most recent one
            cursor.execute(
                "DELETE FROM TTM_Data WHERE Symbol = ? AND Quarter != ?",
                (ticker, most_recent_quarter)
            )
            cursor.connection.commit()
            logging.info(f"Cleared duplicate TTM entries for {ticker}, keeping only the most recent quarter.")
            return True
        return False
    except sqlite3.Error as e:
        logging.error(f"Database error during duplicate check: {e}")
        return False


def chart_needs_update(chart_path, last_data_update, ttm_update=False, annual_update=False):
    print("Checking if chart needs update")

    # If TTM or Annual data has been updated, we need to update the chart
    if ttm_update or annual_update:
        print("---data update true")
        return True

    # If the chart file doesn't exist, we need to create and hence update it
    if not os.path.exists(chart_path):
        print("---chart does not exist true")
        return True

    # If the chart exists, compare the last modified time of the chart with the last data update
    last_chart_update = datetime.fromtimestamp(os.path.getmtime(chart_path))
    if last_data_update > last_chart_update:
        print("---last update was more recent than the chart update true")
        # If the data is more recent than the chart, an update is needed
        return True

    # If none of the above conditions are met, no update is needed
    print("---no chart generation conditions met")
    return False

def create_formatted_dataframe(df):
    def format_value(value):
        """Format the value as currency with appropriate suffix (B for billions, M for millions, K for thousands)."""
        if pd.isna(value):
            return "N/A"
        if abs(value) >= 1e9:
            return f"${value / 1e9:,.1f}B"
        elif abs(value) >= 1e6:
            return f"${value / 1e6:,.1f}M"
        else:
            return f"${value / 1e3:,.1f}K"

    df['Formatted_Revenue'] = df['Revenue'].apply(format_value)
    df['Formatted_Net_Income'] = df['Net_Income'].apply(format_value)
    df['Formatted_EPS'] = df['EPS'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
    return df



def generate_eps_chart(ticker, charts_output_dir, financial_data_df):
    print("Generating EPS chart")
    if financial_data_df.empty:
        return

    eps_chart_path = os.path.join(charts_output_dir, f"{ticker}_eps_chart.png")

    # Ensure the EPS values are numeric
    financial_data_df['EPS'] = pd.to_numeric(financial_data_df['EPS'], errors='coerce')

    positions = np.arange(len(financial_data_df))
    width = 0.4
    fig, ax = plt.subplots(figsize=(8, 5))
    eps_bars = ax.bar(positions, financial_data_df['EPS'], width, label='EPS', color='teal')

    # Adding grid lines for each tick and a thicker line at y=0
    ax.grid(True, which='both', linestyle='--', linewidth='0.5', color='grey', axis='y')
    ax.axhline(0, color='black', linewidth=1)  # Thicker line at y=0
    ax.axhline(0, color='black', linewidth='2')
    ax.set_ylabel('Earnings Per Share (EPS)')
    ax.set_title(f'EPS Chart for {ticker}')
    ax.set_xticks(positions)
    ax.set_xticklabels(financial_data_df['Date'], rotation=0)

    # Calculate the buffer
    eps_values = financial_data_df['EPS'].abs()
    buffer = eps_values.max() * 0.2

    # Calculate upper and lower limits
    max_value = eps_values.max()
    min_value = financial_data_df['EPS'].min()
    upper_limit = max_value + buffer
    lower_limit = min_value - buffer if min_value < 0 else 0

    # Set the new y-axis limits
    ax.set_ylim(lower_limit, upper_limit)

    add_eps_value_labels(ax, eps_bars, financial_data_df)
    plt.tight_layout()

    # Ensure the directory exists
    os.makedirs(charts_output_dir, exist_ok=True)

    plt.savefig(eps_chart_path)
    plt.close(fig)

def add_eps_value_labels(ax, bars, df):
    for bar in bars:
        height = bar.get_height()
        # Use the formatted EPS values directly from the DataFrame
        label = df.loc[df['EPS'] == height, 'Formatted_EPS'].values[0]
        y_offset = 12 if height >= 0 else -12
        ax.annotate(label,
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, y_offset),
                    textcoords="offset points",
                    ha='center', va='bottom', color="black")

def generate_revenue_net_income_chart(financial_data_df, ticker, revenue_chart_path):
    print("Generating revenue and net income chart")
    df = create_formatted_dataframe(financial_data_df)

    # Ensure the Revenue and Net_Income values are numeric
    df['Revenue'] = pd.to_numeric(df['Revenue'].replace('[\$,]', '', regex=True), errors='coerce')
    df['Net_Income'] = pd.to_numeric(df['Net_Income'].replace('[\$,]', '', regex=True), errors='coerce')

    df['Date'] = df['Date'].astype(str)
    positions = np.arange(len(df))
    width = 0.3
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate scale factor and label ending
    max_net_income = df['Net_Income'].max()
    if abs(max_net_income) >= 1e9:
        scale_factor = 1e9
        label_ending = 'B'
        ylabel = 'Amount (Billions $)'
    else:
        scale_factor = 1e6
        label_ending = 'M'
        ylabel = 'Amount (Millions $)'

    # Calculate the buffer
    revenue_net_income_values = pd.concat([df['Revenue'], df['Net_Income']]) / scale_factor
    abs_rev_net_income_values = revenue_net_income_values.abs()
    buffer = abs_rev_net_income_values.max() * 0.2

    # Calculate upper and lower limits
    max_value = revenue_net_income_values.max()
    min_value = df['Net_Income'].min() / scale_factor
    upper_limit = max_value + buffer
    lower_limit = min_value - buffer if min_value < 0 else 0

    # Create bars
    bars1 = ax.bar(positions - width / 2, df['Revenue'] / scale_factor, width, label=f'Revenue ({label_ending})', color='green')
    bars2 = ax.bar(positions + width / 2, df['Net_Income'] / scale_factor, width, label=f'Net Income ({label_ending})', color='blue')
    ax.set_ylabel(ylabel)

    # Set the new y-axis limits
    ax.set_ylim(lower_limit, upper_limit)
    ax.set_title(f'Revenue and Net Income for {ticker}')
    ax.set_xticks(positions)
    ax.set_xticklabels(df['Date'], rotation=0)
    ax.legend()

    # Adding grid lines for each tick and a thicker line at y=0
    ax.grid(True, which='both', linestyle='--', linewidth='0.5', color='grey', axis='y')
    ax.axhline(0, color='black', linewidth=1)  # Thicker line at y=0

    add_value_labels(ax, bars1, df, 'Formatted_Revenue', scale_factor)
    add_value_labels(ax, bars2, df, 'Formatted_Net_Income', scale_factor)
    plt.tight_layout()

    # Ensure the directory exists
    os.makedirs(os.path.dirname(revenue_chart_path) or ".", exist_ok=True)

    plt.savefig(revenue_chart_path)
    plt.close()

def add_value_labels(ax, bars, df, column, scale_factor):
    for bar in bars:
        height = bar.get_height()
        # Scale back to match the original values
        scaled_value = height * scale_factor
        # Use the correct column for comparison
        column_name = 'Net_Income' if 'Net_Income' in column else 'Revenue'
        label = df.loc[np.isclose(df[column_name], scaled_value, atol=1e-2), column].values
        if len(label) > 0:
            label = label[0]
        else:
            label = "N/A"  # Fallback label in case no match is found
        y_offset = 3 if height >= 0 else -12
        ax.annotate(label, xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, y_offset),
                    textcoords="offset points", ha='center', va='bottom')

def generate_financial_charts(ticker, charts_output_dir, financial_data):
    print("Generating financial charts")

    if financial_data.empty:
        print("---chart data is empty so return")
        return

    revenue_chart_path = os.path.join(charts_output_dir, f"{ticker}_revenue_net_income_chart.png")
    eps_chart_path = os.path.join(charts_output_dir, f"{ticker}_eps_chart.png")
    print("--- define chart path")

    if 'Last_Updated' in financial_data.columns:
        financial_data['Last_Updated'] = pd.to_datetime(financial_data['Last_Updated'], errors='coerce')
        last_data_update = financial_data['Last_Updated'].max()
    else:
        print("Warning: 'Last_Updated' column not found. Proceeding without it.")
        last_data_update = datetime.now()  # Use current time as a fallback

    if chart_needs_update(revenue_chart_path, last_data_update):
        generate_revenue_net_income_chart(financial_data, ticker, revenue_chart_path)
        print("--- needed a rev/net income chart update, generated")

    if chart_needs_update(eps_chart_path, last_data_update):
        generate_eps_chart(ticker, charts_output_dir, financial_data)
        print("---needed a new eps chart, generated")

    # Generate financial data table in HTML format
    generate_financial_data_table_html(ticker, financial_data, charts_output_dir)




def calculate_and_format_changes(df):
    # Ensure the DataFrame is sorted by 'Date' to calculate changes correctly
    df.sort_values('Date', ascending=True, inplace=True)

    # Define the columns to calculate yearly changes
    financial_columns = ['Revenue', 'Net_Income', 'EPS']

    # Convert columns to float if they are not already, assuming they are strings with $ and M symbols
    for column in financial_columns:
        if df[column].dtype == 'object':
            df[column] = df[column].replace('[\$,MK]', '', regex=True).astype(float) * 1e3

    # Calculate and format the yearly changes
    for column in financial_columns:
        change_column = f"{column}_Change"
        df[change_column] = df[column].pct_change(fill_method=None) * 100

        # Format the changes as percentages with one decimal place
        df[change_column] = df[change_column].apply(lambda x: f"{x:.1f}%" if pd.notnull(x) else "N/A")

    return df

def style_changes(val):
    if isinstance(val, str) and '%' in val:
        color = 'red' if '-' in val else 'green'
        return f'color: {color};'
    return ''

def generate_financial_data_table_html(ticker, df, charts_output_dir):
    df = calculate_and_format_changes(df)

    # Drop raw data columns, keep only formatted and change columns
    columns_to_keep = ['Date', 'Formatted_Revenue', 'Formatted_Net_Income', 'Formatted_EPS', 'Revenue_Change', 'Net_Income_Change', 'EPS_Change']
    df = df[columns_to_keep]

    # Rename columns for better headers
    df.columns = ['Date', 'Revenue', 'Net Income', 'EPS', 'Revenue Change', 'Net Income Change', 'EPS Change']

    # Calculate average changes and add as the last row
    avg_changes = df[['Revenue Change', 'Net Income Change', 'EPS Change']].replace('N/A', np.nan).apply(lambda x: pd.to_numeric(x.str.replace('%', ''), errors='coerce')).mean()
    avg_changes = avg_changes.apply(lambda x: f"{x:.1f}%" if pd.notnull(x) else "N/A")
    avg_row = pd.Series(['Average'] + [''] * 3 + avg_changes.tolist(), index=df.columns)
    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

    # Create styled HTML table
    styled_table = df.style.applymap(style_changes, subset=['Revenue Change', 'Net Income Change', 'EPS Change'])
    html_table = styled_table.to_html()

    # Save the HTML table to a file
    table_file_path = os.path.join(charts_output_dir, f"{ticker}_rev_net_table.html")
    os.makedirs(charts_output_dir, exist_ok=True)  # Ensure the directory exists
    with open(table_file_path, 'w', encoding='utf-8') as f:
        f.write(html_table)
    print(f"Financial data table for {ticker} saved to {table_file_path}")



def prepare_data_for_charts(ticker, cursor):
    print("Preparing data for charts")
    # Fetch annual data including Last_Updated
    cursor.execute("SELECT Date, Revenue, Net_Income, EPS, Last_Updated FROM Annual_Data WHERE Symbol = ? ORDER BY Date", (ticker,))
    annual_data = cursor.fetchall()
    print("---fetching all annual data", annual_data)

    # Fetch TTM data including Last_Updated
    cursor.execute("SELECT 'TTM' AS Date, TTM_Revenue AS Revenue, TTM_Net_Income AS Net_Income, TTM_EPS AS EPS, Last_Updated FROM TTM_Data WHERE Symbol = ?", (ticker,))
    ttm_data = cursor.fetchall()
    print("---fetching all ttm data from database", ttm_data)

    # Fetch TTM data including Last_Updated
    cursor.execute(
        "SELECT 'TTM' AS Date, TTM_Revenue AS Revenue, TTM_Net_Income AS Net_Income, TTM_EPS AS EPS, Quarter AS Quarter, Last_Updated FROM TTM_Data WHERE Symbol = ?",
        (ticker,))
    ttm_datab = cursor.fetchall()
    print("---fetching all ttm data from database", ttm_data)

    # Convert to DataFrame and ensure correct data types
    annual_df = pd.DataFrame(annual_data, columns=['Date', 'Revenue', 'Net_Income', 'EPS', 'Last_Updated'])
    ttm_df = pd.DataFrame(ttm_data, columns=['Date', 'Revenue', 'Net_Income', 'EPS', 'Last_Updated'])
    ttm_dfb = pd.DataFrame(ttm_datab, columns=['Date', 'Revenue', 'Net_Income', 'EPS', 'Quarter', 'Last_Updated'])
    print("---converting df to correct names", annual_df, ttm_df)

    # Ensure all 'Last_Updated' entries are Timestamps
    annual_df['Last_Updated'] = pd.to_datetime(annual_df['Last_Updated'])
    ttm_df['Last_Updated'] = pd.to_datetime(ttm_df['Last_Updated'])
    print("---converting all last updated entries are timestamps")

    # Assuming 'Last_Updated' is in the format 'YYYY-MM-DD HH:MM:SS'
    if not ttm_df.empty and 'Last_Updated' in ttm_df:
        print("---checking if ttm df is not empty")

        # Extract the last quarter end date
        last_quarter_end = ttm_dfb.loc[0, 'Quarter']
        print("---last quarter", last_quarter_end)
        # Check if 'Date' already formatted with 'TTM' to prevent "TTM TTM"
        if not str(last_quarter_end).startswith('TTM'):
            ttm_df.at[0, 'Date'] = f'TTM {last_quarter_end}'
        print('---extracting last quarter date', last_quarter_end)

    # Check if annual_data and ttm_data are empty
    if not annual_data and not ttm_data:
        print("---checking if annual and ttm data is empty", pd)

        return pd.DataFrame()

    # Exclude empty or all-NA columns before concatenation
    annual_df.dropna(axis=1, how='all', inplace=True)
    ttm_df.dropna(axis=1, how='all', inplace=True)

    # Combine annual data and TTM data
    combined_df = pd.concat([annual_df, ttm_df], ignore_index=True)
    print("---combining annual df and ttm df")

    # Handle nulls and ensure correct data types
    combined_df['Revenue'] = pd.to_numeric(combined_df['Revenue'], errors='coerce')
    combined_df['Net_Income'] = pd.to_numeric(combined_df['Net_Income'], errors='coerce')
    combined_df['EPS'] = pd.to_numeric(combined_df['EPS'], errors='coerce')
    print("---handling null data")

    # Clean the combined DataFrame
    combined_df = clean_financial_data(combined_df)

    # Sort the DataFrame to ensure 'TTM' appears last
    combined_df['Date'] = combined_df['Date'].astype(str)
    combined_df.sort_values(by='Date', inplace=True)
    print("sorting df with TTM last")

    # Create formatted values
    combined_df = create_formatted_dataframe(combined_df)

    print("---combined df: ", combined_df)

    return combined_df

def annual_and_ttm_update(ticker, db_path):
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    # Fetch existing data
    annual_data = fetch_ticker_data(ticker, cursor)
    if annual_data is None or len(annual_data) == 0:
        # Fetch new data from Yahoo Finance if not present in the database
        new_annual_data = fetch_annual_data_from_yahoo(ticker)
        if not new_annual_data.empty:
            store_annual_data(ticker, new_annual_data, cursor)
            annual_data = new_annual_data

    ttm_data = fetch_ttm_data(ticker, cursor)
    if ttm_data is None or len(ttm_data) == 0:
        # Fetch new data from Yahoo Finance if not present in the database
        new_ttm_data = fetch_ttm_data_from_yahoo(ticker)
        if new_ttm_data:
            store_ttm_data(ticker, new_ttm_data, cursor)
            ttm_data = [new_ttm_data]

    # Handle duplicates in TTM data
    handle_ttm_duplicates(ticker, cursor)

    # Check for updates
    annual_update_needed = False
    ttm_update_needed = False

    if annual_data is not None and not pd.DataFrame(annual_data).empty:
        latest_annual_date = get_latest_annual_data_date(annual_data)
        annual_update_needed = needs_update(latest_annual_date, 13) or check_null_fields(annual_data, ['Revenue', 'Net_Income', 'EPS'])

    if ttm_data is not None and not pd.DataFrame(ttm_data).empty:
        latest_ttm_date = max([datetime.strptime(row['Quarter'], '%Y-%m-%d') for row in ttm_data])
        ttm_update_needed = needs_update(latest_ttm_date, 4) or check_null_fields(ttm_data, ['TTM_Revenue', 'TTM_Net_Income', 'TTM_EPS'])

    # Fetch and store new data if needed
    if annual_update_needed:
        new_annual_data = fetch_annual_data_from_yahoo(ticker)
        if not new_annual_data.empty:
            store_annual_data(ticker, new_annual_data, cursor)

    if ttm_update_needed:
        new_ttm_data = fetch_ttm_data_from_yahoo(ticker)
        if new_ttm_data:
            store_ttm_data(ticker, new_ttm_data, cursor)

    combined_df = prepare_data_for_charts(ticker, cursor)
    charts_output_dir = "charts"
    generate_financial_charts(ticker, charts_output_dir, combined_df)

    conn.close()
    logging.debug(f"Update for {ticker} completed")




if __name__ == "__main__":
    ticker = "PG"  # Example ticker, replace with desired ticker
    db_path = "Stock Data.db"
    charts_output_dir = "charts"
    annual_and_ttm_update(ticker, db_path)

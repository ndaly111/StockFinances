import yfinance as yf
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
from functools import lru_cache

# Database connection setup
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
    if isinstance(ticker_data, pd.DataFrame):
        if ticker_data.empty:
            logging.warning("Ticker data is an empty DataFrame. Returning None.")
            return None
    elif not ticker_data:
        logging.warning("Ticker data is empty or None. Returning None.")
        return None

    first_entry = ticker_data.iloc[0] if isinstance(ticker_data, pd.DataFrame) else ticker_data[0]

    if isinstance(first_entry, dict):
        if 'Date' not in first_entry or 'Symbol' not in first_entry:
            logging.warning("Missing expected keys in ticker_data dictionary. Returning None.")
            return None
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

def calculate_next_check_date(latest_date, months):
    if latest_date is None:
        return None
    return latest_date + timedelta(days=months * 30)

def needs_update(latest_date, months):
    if latest_date is None:
        return True
    next_check_date = calculate_next_check_date(latest_date, months)
    return next_check_date is None or next_check_date <= datetime.now()

def check_null_fields(data, fields):
    if not isinstance(data, list):
        print(f"Warning: Expected list, got {type(data)} instead. Returning False.")
        return False
    for entry in data:
        if not isinstance(entry, dict):
            print(f"Warning: Expected dictionary, got {type(entry)} instead. Skipping entry.")
            continue
        for field in fields:
            if entry.get(field) in [None, '']:
                print(f"Null or empty value found for field '{field}' in {entry}. Update needed.")
                return True
    return False

def clean_financial_data(df):
    df.dropna(axis=0, how='all', subset=['Revenue', 'Net_Income', 'EPS'], inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df.infer_objects(copy=False)
    return df

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
            missing = set(column_mapping.values()) - set(renamed_columns.values())
            logging.warning(f"Missing required columns for {ticker}: {missing}")
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

def store_annual_data(ticker, annual_data, cursor):
    logging.info("Storing annual data")
    for _, row in annual_data.iterrows():
        date_str = row['Date'].strftime('%Y-%m-%d') if isinstance(row['Date'], pd.Timestamp) else row['Date']
        cursor.execute("""
            SELECT * FROM Annual_Data
            WHERE Symbol = ? AND Date = ? AND Revenue IS NOT NULL AND Net_Income IS NOT NULL AND EPS IS NOT NULL;
        """, (ticker, date_str))
        if cursor.fetchone():
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
    vals = (
        ticker,
        ttm_data['TTM_Revenue'],
        ttm_data['TTM_Net_Income'],
        ttm_data['TTM_EPS'],
        ttm_data.get('Shares_Outstanding'),
        ttm_data['Quarter']
    )
    try:
        cursor.execute("""
            INSERT OR REPLACE INTO TTM_Data
              (Symbol, TTM_Revenue, TTM_Net_Income, TTM_EPS, Shares_Outstanding, Quarter, Last_Updated)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP);
        """, vals)
        cursor.connection.commit()
    except sqlite3.Error as e:
        logging.error(f"Database error while storing/updating TTM data for {ticker}: {e}")

def handle_ttm_duplicates(ticker, cursor):
    logging.info("Checking for duplicate TTM entries for ticker")
    try:
        cursor.execute("SELECT * FROM TTM_Data WHERE Symbol = ? ORDER BY Quarter DESC", (ticker,))
        results = cursor.fetchall()
        if len(results) > 1:
            most_recent = results[0][5]  # Quarter is 6th column
            cursor.execute(
                "DELETE FROM TTM_Data WHERE Symbol = ? AND Quarter != ?",
                (ticker, most_recent)
            )
            cursor.connection.commit()
            logging.info(f"Cleared duplicate TTM entries for {ticker}, keeping only {most_recent}.")
            return True
        return False
    except sqlite3.Error as e:
        logging.error(f"Database error during duplicate check: {e}")
        return False

def chart_needs_update(chart_path, last_data_update, ttm_update=False, annual_update=False):
    print("Checking if chart needs update")
    if ttm_update or annual_update:
        print("---data update true")
        return True
    if not os.path.exists(chart_path):
        print("---chart does not exist true")
        return True
    last_chart_update = datetime.fromtimestamp(os.path.getmtime(chart_path))
    if last_data_update > last_chart_update:
        print("---last update was more recent than the chart update true")
        return True
    print("---no chart generation conditions met")
    return False

def create_formatted_dataframe(df):
    def format_value(value):
        if pd.isna(value):
            return "N/A"
        if abs(value) >= 1e9:
            return f"${value / 1e9:,.1f}B"
        elif abs(value) >= 1e6:
            return f"${value / 1e6:,.1f}M"
        else:
            return f"${value / 1e3:,.1f}K"
    df['Formatted_Revenue']    = df['Revenue'].apply(format_value)
    df['Formatted_Net_Income'] = df['Net_Income'].apply(format_value)
    df['Formatted_EPS']        = df['EPS'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
    return df

def add_eps_value_labels(ax, bars, df):
    for bar in bars:
        h = bar.get_height()
        label = df.loc[df['EPS'] == h, 'Formatted_EPS'].values[0]
        offset = 12 if h >= 0 else -12
        ax.annotate(label, xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, offset), textcoords="offset points",
                    ha='center', va='bottom', color="black")

def generate_eps_chart(ticker, charts_output_dir, financial_data_df):
    print("Generating EPS chart")
    if financial_data_df.empty:
        return
    eps_chart_path = os.path.join(charts_output_dir, f"{ticker}_eps_chart.png")
    financial_data_df['EPS'] = pd.to_numeric(financial_data_df['EPS'], errors='coerce')
    positions = np.arange(len(financial_data_df))
    width = 0.4
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(positions, financial_data_df['EPS'], width, label='EPS', color='teal')
    ax.grid(True, which='both', linestyle='--', linewidth='0.5', color='grey', axis='y')
    ax.axhline(0, color='black', linewidth=2)
    ax.set_ylabel('Earnings Per Share (EPS)')
    ax.set_title(f'EPS Chart for {ticker}')
    ax.set_xticks(positions)
    ax.set_xticklabels(financial_data_df['Date'], rotation=0)
    vals = financial_data_df['EPS'].abs()
    buffer = vals.max() * 0.2
    max_v = vals.max()
    min_v = financial_data_df['EPS'].min()
    ax.set_ylim(min_v - buffer if min_v < 0 else 0, max_v + buffer)
    add_eps_value_labels(ax, bars, financial_data_df)
    plt.tight_layout()
    os.makedirs(charts_output_dir, exist_ok=True)
    plt.savefig(eps_chart_path)
    plt.close(fig)

def add_value_labels(ax, bars, df, column, scale_factor):
    for bar in bars:
        height = bar.get_height()
        scaled = height * scale_factor
        col = 'Net_Income' if 'Net_Income' in column else 'Revenue'
        matches = df.loc[np.isclose(df[col], scaled, atol=1e-2), column]
        label = matches.values[0] if not matches.empty else "N/A"
        offset = 3 if height >= 0 else -12
        ax.annotate(label, xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, offset), textcoords="offset points",
                    ha='center', va='bottom')

def generate_revenue_net_income_chart(financial_data_df, ticker, revenue_chart_path):
    print("Generating revenue and net income chart")
    df = create_formatted_dataframe(financial_data_df.copy())
    df['Revenue']    = pd.to_numeric(df['Revenue'].replace('[\$,]', '', regex=True), errors='coerce')
    df['Net_Income'] = pd.to_numeric(df['Net_Income'].replace('[\$,]', '', regex=True), errors='coerce')
    df['Date'] = df['Date'].astype(str)
    positions = np.arange(len(df))
    width     = 0.3
    fig, ax   = plt.subplots(figsize=(10, 6))
    max_net   = df['Net_Income'].max()
    if abs(max_net) >= 1e9:
        sf, le, ylabel = 1e9, 'B', 'Amount (Billions $)'
    else:
        sf, le, ylabel = 1e6, 'M', 'Amount (Millions $)'
    vals = pd.concat([df['Revenue'], df['Net_Income']]) / sf
    buf  = vals.abs().max() * 0.2
    max_v = vals.max()
    min_v = df['Net_Income'].min() / sf
    ax.bar(positions - width/2, df['Revenue']/sf, width, label=f'Revenue ({le})', color='green')
    bars2 = ax.bar(positions + width/2, df['Net_Income']/sf, width, label=f'Net Income ({le})', color='blue')
    ax.set_ylabel(ylabel)
    ax.set_ylim(min_v - buf if min_v < 0 else 0, max_v + buf)
    ax.set_title(f'Revenue and Net Income for {ticker}')
    ax.set_xticks(positions)
    ax.set_xticklabels(df['Date'], rotation=0)
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth='0.5', color='grey', axis='y')
    ax.axhline(0, color='black', linewidth=1)
    bars1 = ax.patches[:len(df)]
    add_value_labels(ax, bars1, df, 'Formatted_Revenue', sf)
    add_value_labels(ax, bars2, df, 'Formatted_Net_Income', sf)
    plt.tight_layout()
    os.makedirs(os.path.dirname(revenue_chart_path) or ".", exist_ok=True)
    plt.savefig(revenue_chart_path)
    plt.close()

def calculate_and_format_changes(df):
    df.sort_values('Date', ascending=True, inplace=True)
    cols = ['Revenue', 'Net_Income', 'EPS']
    for c in cols:
        if df[c].dtype == 'object':
            df[c] = df[c].replace('[\$,MK]', '', regex=True).astype(float)*1e3
    for c in cols:
        ch = f"{c}_Change"
        df[ch] = df[c].pct_change(fill_method=None)*100
        df[ch] = df[ch].apply(lambda x: f"{x:.1f}%" if pd.notnull(x) else "N/A")
    return df

def style_changes(val):
    if isinstance(val, str) and '%' in val:
        return f"color: {'red' if '-' in val else 'green'};"
    return ''

def generate_financial_data_table_html(ticker, df, charts_output_dir):
    df = calculate_and_format_changes(df.copy())
    cols = ['Date','Formatted_Revenue','Formatted_Net_Income','Formatted_EPS',
            'Revenue_Change','Net_Income_Change','EPS_Change']
    df = df[cols]
    df.columns = ['Date','Revenue','Net Income','EPS','Revenue Change','Net Income Change','EPS Change']
    avg = df[['Revenue Change','Net Income Change','EPS Change']].replace('N/A',np.nan)\
           .apply(lambda x: pd.to_numeric(x.str.replace('%',''),errors='coerce')).mean()\
           .apply(lambda x: f"{x:.1f}%" if pd.notnull(x) else "N/A")
    avg_row = pd.Series(['Average']+['']*3+avg.tolist(), index=df.columns)
    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
    styled = df.style.applymap(style_changes, subset=['Revenue Change','Net Income Change','EPS Change'])
    html = styled.to_html()
    path = os.path.join(charts_output_dir, f"{ticker}_rev_net_table.html")
    os.makedirs(charts_output_dir, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Financial data table for {ticker} saved to {path}")

def prepare_data_for_charts(ticker, cursor):
    print("Preparing data for charts")
    cursor.execute("SELECT Date, Revenue, Net_Income, EPS, Last_Updated FROM Annual_Data WHERE Symbol = ? ORDER BY Date", (ticker,))
    annual_data = cursor.fetchall()
    print("---fetching all annual data", annual_data)

    cursor.execute("SELECT 'TTM' AS Date, TTM_Revenue AS Revenue, TTM_Net_Income AS Net_Income, TTM_EPS AS EPS, Last_Updated FROM TTM_Data WHERE Symbol = ?", (ticker,))
    ttm_data = cursor.fetchall()
    print("---fetching all ttm data from database", ttm_data)

    cursor.execute(
        "SELECT 'TTM' AS Date, TTM_Revenue AS Revenue, TTM_Net_Income AS Net_Income, TTM_EPS AS EPS, Quarter AS Quarter, Last_Updated FROM TTM_Data WHERE Symbol = ?",
        (ticker,))
    ttm_datab = cursor.fetchall()
    print("---fetching all ttm data from database", ttm_data)

    annual_df = pd.DataFrame(annual_data, columns=['Date','Revenue','Net_Income','EPS','Last_Updated'])
    ttm_df    = pd.DataFrame(ttm_data, columns=['Date','Revenue','Net_Income','EPS','Last_Updated'])
    ttm_dfb   = pd.DataFrame(ttm_datab, columns=['Date','Revenue','Net_Income','EPS','Quarter','Last_Updated'])
    print("---converting df to correct names", annual_df, ttm_df)

    annual_df['Last_Updated'] = pd.to_datetime(annual_df['Last_Updated'])
    ttm_df['Last_Updated']    = pd.to_datetime(ttm_df['Last_Updated'])
    print("---converting all last updated entries are timestamps")

    if not ttm_df.empty and 'Last_Updated' in ttm_df:
        print("---checking if ttm df is not empty")
        last_quarter_end = ttm_dfb.loc[0, 'Quarter']
        print("---last quarter", last_quarter_end)
        if not str(last_quarter_end).startswith('TTM'):
            ttm_df.at[0, 'Date'] = f'TTM {last_quarter_end}'
        print('---extracting last quarter date', last_quarter_end)

    if not annual_data and not ttm_data:
        print("---checking if annual and ttm data is empty", pd)
        return pd.DataFrame()

    annual_df.dropna(axis=1, how='all', inplace=True)
    ttm_df.dropna(axis=1, how='all', inplace=True)

    combined_df = pd.concat([annual_df, ttm_df], ignore_index=True)
    print("---combining annual df and ttm df")

    combined_df['Revenue']    = pd.to_numeric(combined_df['Revenue'], errors='coerce')
    combined_df['Net_Income'] = pd.to_numeric(combined_df['Net_Income'], errors='coerce')
    combined_df['EPS']        = pd.to_numeric(combined_df['EPS'], errors='coerce')
    print("---handling null data")

    combined_df = clean_financial_data(combined_df)
    combined_df['Date'] = combined_df['Date'].astype(str)
    combined_df.sort_values(by='Date', inplace=True)
    print("sorting df with TTM last")

    combined_df = create_formatted_dataframe(combined_df)
    print("---combined df: ", combined_df)
    return combined_df

def annual_and_ttm_update(ticker, db_path):
    conn    = get_db_connection(db_path)
    cursor  = conn.cursor()

    annual_data = fetch_ticker_data(ticker, cursor)
    if annual_data is None or len(annual_data) == 0:
        new_annual_data = fetch_annual_data_from_yahoo(ticker)
        if not new_annual_data.empty:
            store_annual_data(ticker, new_annual_data, cursor)
            annual_data = new_annual_data

    ttm_data = fetch_ttm_data(ticker, cursor)
    if ttm_data is None or len(ttm_data) == 0:
        new_ttm_data = fetch_ttm_data_from_yahoo(ticker)
        if new_ttm_data:
            store_ttm_data(ticker, new_ttm_data, cursor)
            ttm_data = [new_ttm_data]

    handle_ttm_duplicates(ticker, cursor)

    annual_update_needed = False
    ttm_update_needed    = False

    if annual_data is not None and not pd.DataFrame(annual_data).empty:
        latest_annual_date = get_latest_annual_data_date(annual_data)
        annual_update_needed = (
            needs_update(latest_annual_date, 13)
            or check_null_fields(annual_data, ['Revenue', 'Net_Income', 'EPS'])
        )

    if ttm_data is not None and not pd.DataFrame(ttm_data).empty:
        # — PATCH: filter out None/empty quarters before parsing —
        valid_quarters = [
            row['Quarter']
            for row in ttm_data
            if isinstance(row.get('Quarter'), str) and row['Quarter'].strip()
        ]
        if valid_quarters:
            latest_ttm_date = max(
                datetime.strptime(q, '%Y-%m-%d')
                for q in valid_quarters
            )
        else:
            latest_ttm_date = None
            logging.warning(f"[{ticker}] No valid TTM 'Quarter' strings found; skipping date parse.")

        ttm_update_needed = (
            latest_ttm_date is None
            or needs_update(latest_ttm_date, 4)
            or check_null_fields(ttm_data, ['TTM_Revenue', 'TTM_Net_Income', 'TTM_EPS'])
        )

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
    ticker = "PG"
    db_path = "Stock Data.db"
    annual_and_ttm_update(ticker, db_path)

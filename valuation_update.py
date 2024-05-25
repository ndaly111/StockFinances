import requests
from bs4 import BeautifulSoup
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import os
import yfinance as yf
import csv




def log_valuation_data(ticker, nicks_ttm_valuation, nicks_forward_valuation, finviz_ttm_valuation,
                       finviz_forward_valuation):
    db_path = "Stock Data.db"
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # Create the table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ValuationHistory (
                id INTEGER PRIMARY KEY,
                ticker TEXT,
                date DATE DEFAULT (datetime('now','localtime')),
                nicks_ttm_valuation REAL,
                nicks_forward_valuation REAL,
                finviz_ttm_valuation REAL,
                finviz_forward_valuation REAL
            );
        ''')

        # Insert the new valuation data
        cursor.execute('''
            INSERT INTO ValuationHistory (ticker, nicks_ttm_valuation, nicks_forward_valuation, finviz_ttm_valuation, finviz_forward_valuation)
            VALUES (?, ?, ?, ?, ?);
        ''', (ticker, nicks_ttm_valuation, nicks_forward_valuation, finviz_ttm_valuation, finviz_forward_valuation))

        conn.commit()
        print(f"Inserted valuation data for {ticker} into ValuationHistory.")


def finviz_five_yr(ticker, cursor):
    """Fetches and stores the 5-year EPS growth percentage from Finviz into the database."""
    url = f'https://finviz.com/quote.ashx?t={ticker}'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        estimate_cell = soup.find('td', text='EPS next 5Y')
        if estimate_cell:
            next_td = estimate_cell.find_next_sibling('td')
            estimate_value = next_td.text.strip('%')  # Remove the '%' symbol
            if estimate_value:  # Check if estimate_value is not empty
                try:
                    estimate_value = float(estimate_value)  # Convert to float

                    # Check if ticker exists in the table
                    cursor.execute(f'''
                        SELECT 1 FROM Tickers_Info WHERE ticker = ?;
                    ''', (ticker,))
                    if not cursor.fetchone():
                        # Insert ticker if it does not exist
                        cursor.execute(f'''
                            INSERT INTO Tickers_Info (ticker) VALUES (?);
                        ''', (ticker,))
                        cursor.connection.commit()
                        print(f"Inserted new ticker {ticker} into Tickers_Info.")

                    # Update the table with the new growth estimate
                    cursor.execute(f'''
                        UPDATE 'Tickers_Info'
                        SET FINVIZ_5yr_gwth = ? 
                        WHERE ticker = ?;
                    ''', (estimate_value, ticker))
                    cursor.connection.commit()
                    print(f"Stored Finviz 5-Year Growth for {ticker}: {estimate_value}")
                except ValueError:
                    print(f"Invalid estimate value '{next_td.text}' for ticker {ticker}.")
            else:
                print(f"Empty estimate value for ticker {ticker}.")
        else:
            print(f"Could not find the 5-year EPS growth estimate for {ticker} on Finviz.")
    else:
        print(f"Failed to retrieve Finviz data for {ticker}, status code: {response.status_code}")


def fetch_financial_valuation_data(ticker, db_path):
    print(f"Fetching financial valuation data for: {ticker}")

    stock = yf.Ticker(ticker)
    current_price = stock.info.get('currentPrice')

    with sqlite3.connect(db_path) as conn:
        # Fetch TTM financial data and format it
        ttm_query = """
        SELECT 'TTM' AS Year, TTM_Revenue AS Revenue, TTM_EPS AS EPS
        FROM TTM_Data
        WHERE Symbol = ?
        ORDER BY Last_Updated DESC
        LIMIT 1;
        """
        ttm_data = pd.read_sql_query(ttm_query, conn, params=(ticker,))
        print("TTM data fetched:", ttm_data)

        # Fetch forecast financial data and format it
        forecast_query = """
        SELECT strftime('%Y', Date) AS Year, ForwardRevenue AS Revenue, ForwardEPS AS EPS
        FROM ForwardFinancialData
        WHERE Ticker = ?
        ORDER BY Date;
        """
        forecast_data = pd.read_sql_query(forecast_query, conn, params=(ticker,))
        print("Forecast data fetched:", forecast_data)

        growth_query = """
        SELECT nicks_growth_rate, FINVIZ_5yr_gwth, projected_profit_margin
        FROM Tickers_Info
        WHERE ticker = ?;
        """
        growth_value = pd.read_sql_query(growth_query, conn, params=(ticker,))
        print("growth value", growth_value)

        # Combine TTM and forecast data into a single DataFrame
        combined_data = pd.concat([ttm_data, forecast_data]).reset_index(drop=True)

        return combined_data, growth_value, current_price, forecast_data



def calculate_valuations(combined_data, growth_values, treasury_yield, current_price, marketcap):
    treasury_yield = (float(treasury_yield) / 100)
    print('treasury yield', treasury_yield)

    nicks_growth_rate = float(growth_values['nicks_growth_rate'].iloc[0] if pd.notna(growth_values['nicks_growth_rate'].iloc[0]) else 0) / 100
    finviz_growth_rate = float(growth_values['FINVIZ_5yr_gwth'].iloc[0] if pd.notna(growth_values['FINVIZ_5yr_gwth'].iloc[0]) else 0) / 100
    print("finviz growth rate", finviz_growth_rate)
    projected_profit_margin = float(growth_values['projected_profit_margin'].iloc[0] if pd.notna(growth_values['projected_profit_margin'].iloc[0]) else 0) / 100

    nicks_fair_pe = ((nicks_growth_rate - treasury_yield + 1) ** 10) * 10
    finviz_fair_pe = ((finviz_growth_rate - treasury_yield + 1) ** 10) * 10
    print("finviz fair pe", finviz_fair_pe)

    nicks_fair_ps = (((nicks_growth_rate - treasury_yield + 1) ** 10) * 10) * projected_profit_margin
    finviz_fair_ps = (((finviz_growth_rate - treasury_yield + 1) ** 10) * 10) * projected_profit_margin
    print("finviz fair ps", finviz_fair_ps)

    print("nicks fair pe", nicks_fair_pe)
    print("finviz fair pe", finviz_fair_pe)
    print("nicks fair ps", nicks_fair_ps)
    print("finviz fair ps", finviz_fair_ps)

    # Calculate revenue per share for each year
    combined_data['Revenue_Per_Share'] = (combined_data['Revenue'] / marketcap) * current_price

    def calculate_valuation(row, nicks_fair_pe, nicks_fair_ps, finviz_fair_pe, finviz_fair_ps):
        if row['EPS'] > 0:
            nicks_valuation = row['EPS'] * nicks_fair_pe
            finviz_valuation = row['EPS'] * finviz_fair_pe if finviz_fair_pe else None
        else:
            nicks_valuation = row['Revenue_Per_Share'] * nicks_fair_ps
            finviz_valuation = row['Revenue_Per_Share'] * finviz_fair_ps if finviz_fair_ps else None
            print("finviz valuation", finviz_valuation)
        return {
            'Basis_Value': row['EPS'] if row['EPS'] > 0 else row['Revenue_Per_Share'],
            'Basis_Type': 'EPS' if row['EPS'] > 0 else 'Revenue',
            'Nicks_Valuation': nicks_valuation,
            'Finviz_Valuation': finviz_valuation
        }

    calculated_valuations = combined_data.apply(lambda row: calculate_valuation(row, nicks_fair_pe, nicks_fair_ps, finviz_fair_pe, finviz_fair_ps), axis=1)

    combined_data['Basis_Value'] = calculated_valuations.apply(lambda x: x['Basis_Value'])
    combined_data['Basis_Type'] = calculated_valuations.apply(lambda x: x['Basis_Type'])
    combined_data['Nicks_Valuation'] = calculated_valuations.apply(lambda x: x['Nicks_Valuation'])
    combined_data['Finviz_Valuation'] = calculated_valuations.apply(lambda x: x['Finviz_Valuation'])

    print("Valuation calculations:")
    print(combined_data[['Year', 'Basis_Type', 'Basis_Value', 'Nicks_Valuation', 'Finviz_Valuation']])

    return combined_data, nicks_fair_pe, finviz_fair_pe, nicks_fair_ps, finviz_fair_ps




def plot_valuation_chart(valuation_data, current_price, ticker, growth_value):
    """
    Plots the valuation data with the current stock price.

    Args:
        valuation_data (pd.DataFrame): DataFrame containing years, Nicks_Valuation, and Finviz_Valuation.
        current_price (float): Current price of the stock.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plotting the valuation lines if the growth values are not None
    if not pd.isna(growth_value['nicks_growth_rate'].iloc[0]):
        ax.plot(valuation_data['Year'], valuation_data['Nicks_Valuation'], label='Nicks Valuation', color='blue', marker='o')

    if not pd.isna(growth_value['FINVIZ_5yr_gwth'].iloc[0]):
        ax.plot(valuation_data['Year'], valuation_data['Finviz_Valuation'], label='Finviz Valuation', color='green', marker='o')

    # Adding a scatter plot for the current price at all points for visual comparison
    ax.plot(valuation_data['Year'], [current_price] * len(valuation_data), color='orange', label='Current Price', linestyle='--', marker='x')

    # Adding labels and title
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Valuation (USD)', fontsize=12)
    ax.set_title(f'Valuation Comparison for {ticker}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)

    # Enhance the grid lines
    ax.grid(True, linestyle='--', alpha=0.7)

    # Increase font size for ticks
    ax.tick_params(axis='both', which='major', labelsize=10)

    fig_path = f"charts/{ticker}_valuation_chart.png"
    plt.savefig(fig_path, bbox_inches='tight')
    print(f"Figure saved to {fig_path}")
    plt.close()


def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)
    current_price = stock.info.get('currentPrice')
    forward_eps = stock.info.get('forwardEps')
    pe_ratio = stock.info.get('trailingPE', None)
    price_to_sales = stock.info.get('priceToSalesTrailing12Months', None)
    forward_pe_ratio = current_price / forward_eps if forward_eps else None

    return current_price, pe_ratio, price_to_sales, forward_pe_ratio




def generate_valuation_tables(ticker, combined_data, growth_values, treasury_yield, current_price, nicks_fair_pe, finviz_fair_pe, nicks_fair_ps):
    treasury_yield = float(treasury_yield)
    current_price, pe_ratio, price_to_sales, forward_pe_ratio = fetch_stock_data(ticker)
    nicks_fair_ps = float(nicks_fair_ps)

    current_price_formatted = f"${current_price:,.2f}"
    treasury_yield_formatted = f"{treasury_yield:.1f}%"
    nicks_growth_rate_formatted = f"{growth_values['nicks_growth_rate'].iloc[0]:.0f}%" if pd.notna(growth_values['nicks_growth_rate'].iloc[0]) else "N/A"
    finviz_growth_rate_formatted = f"{growth_values['FINVIZ_5yr_gwth'].iloc[0]:.0f}%" if pd.notna(growth_values['FINVIZ_5yr_gwth'].iloc[0]) else "N/A"
    expected_margin_formatted = f"{growth_values['projected_profit_margin'].iloc[0]:.0f}%" if pd.notna(growth_values['projected_profit_margin'].iloc[0]) else "N/A"

    estimates_string = (
        f"Nicks&nbsp;Growth:&nbsp;{nicks_growth_rate_formatted}<br>"
        f"Nick's&nbsp;Expected&nbsp;Margin:&nbsp;{expected_margin_formatted}<br>"
        f"FINVIZ&nbsp;Growth:&nbsp;{finviz_growth_rate_formatted}"
    )

    fair_pe_string = (
        f"Nicks:&nbsp;{nicks_fair_pe:.0f}<br>"
        f"Finviz:&nbsp;{finviz_fair_pe:.0f}" if finviz_fair_pe is not None else "Finviz: N/A"
    )

    table_1_data = {
        "Share Price": [current_price_formatted],
        "Treasury Yield": [treasury_yield_formatted],
        "Estimates": [estimates_string],
        "Fair Value (P/E)": [fair_pe_string],
        "Fair Value (P/S)": [f"Nick's: {nicks_fair_ps:.3f}"],
        "Current P/S": [f"{price_to_sales:.1f}" if price_to_sales else "N/A"]
    }

    if pe_ratio is not None and pe_ratio > 0:
        table_1_data["Current P/E"] = [f"{pe_ratio:.1f}"]

    table_1_df = pd.DataFrame(table_1_data)
    table_1_html = table_1_df.to_html(index=False, escape=False, classes='table table-striped', justify='left')
    table_1_path = os.path.join('charts', f"{ticker}_valuation_info.html")
    with open(table_1_path, "w") as file:
        file.write(table_1_html)

    print("combined data", combined_data)
    combined_data['Nicks_Valuation'] = combined_data['Nicks_Valuation'].apply(valuation_format)
    if 'Finviz_Valuation' in combined_data.columns:
        combined_data['Finviz_Valuation'] = combined_data['Finviz_Valuation'].apply(valuation_format)

    combined_data['Nicks vs Share Price'] = combined_data['Nicks_Valuation'].apply(
        lambda x: f"{((remove_commas_and_convert(x.strip('$BMK')) / current_price - 1) * 100):.1f}%" if remove_commas_and_convert(x.strip('$BMK')) is not None else 'N/A')
    if 'Finviz_Valuation' in combined_data.columns:
        combined_data['Finviz vs Share Price'] = combined_data['Finviz_Valuation'].apply(
            lambda x: f"{((remove_commas_and_convert(x.strip('$BMK')) / current_price - 1) * 100):.1f}%" if remove_commas_and_convert(x.strip('$BMK')) is not None else 'N/A')

    def format_color(value):
        try:
            value = float(value.strip('%'))
            color = 'red' if value < 0 else 'green'
            return f'<span style="color: {color}">{value:.1f}%</span>'
        except ValueError:
            return value

    combined_data['Nicks vs Share Price'] = combined_data['Nicks vs Share Price'].apply(format_color)
    if 'Finviz vs Share Price' in combined_data.columns:
        combined_data['Finviz vs Share Price'] = combined_data['Finviz vs Share Price'].apply(format_color)

    combined_data['Basis'] = combined_data.apply(lambda row: f"${format_number(row['EPS'])} EPS" if row['EPS'] > 0 else f"${format_number(row['Revenue_Per_Share'])} RevPS", axis=1)

    table_2_data = {
        "Basis": combined_data['Basis'],
        "Year": combined_data['Year'],
        "Nicks Valuation": combined_data['Nicks_Valuation'],
        "Nicks vs Share Price": combined_data['Nicks vs Share Price'],
    }

    # Add Finviz valuations if conditions are met
    if pd.notna(growth_values['FINVIZ_5yr_gwth'].iloc[0]) and 'Finviz_Valuation' in combined_data.columns:
        table_2_data.update({
            "Finviz Valuation": combined_data['Finviz_Valuation'],
            "Finviz vs Share Price": combined_data['Finviz vs Share Price']
        })

    table_2_df = pd.DataFrame(table_2_data)
    table_2_html = table_2_df.to_html(index=False, escape=False, classes='table table-striped', justify='left')
    table_2_path = os.path.join('charts', f"{ticker}_valuation_table.html")
    with open(table_2_path, "w") as file:
        file.write(table_2_html)

    print(f"Saved valuation info to {table_1_path} and valuation table to {table_2_path}")



# Helper function to remove commas and convert to float
def remove_commas_and_convert(value):
    try:
        return float(value.replace(',', ''))
    except ValueError:
        return None


# Helper function to format numbers with appropriate suffixes
def format_number(value):
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    elif value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    elif value >= 1_000:
        return f"{value / 1_000:.2f}K"
    else:
        return f"{value:.2f}"

# Helper function to apply valuation formatting
def valuation_format(value):
    if value is None:
        return "N/A"
    if isinstance(value, str):
        return value
    return f"${value:,.2f}"



def process_update_growth_csv(file_path, db_path):
    """Reads the update_growth.csv file and updates the database with the new growth rates and profit margins."""
    if not os.path.exists(file_path):
        print(f"{file_path} does not exist. No updates to process.")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) == 2:
                ticker, growth_rate = row
                profit_margin = None
            elif len(row) == 3:
                ticker, growth_rate, profit_margin = row
                profit_margin = None if profit_margin == '0' else profit_margin
            else:
                print(f"Invalid row format: {row}")
                continue

            ticker = ticker.upper()  # Normalize ticker to uppercase

            # Check if growth_rate is not an empty string
            if growth_rate.strip():
                try:
                    growth_rate = float(growth_rate)
                except ValueError:
                    print(f"Invalid growth rate '{growth_rate}' for ticker {ticker}. Skipping update.")
                    continue
            else:
                print(f"No growth rate provided for ticker {ticker}. Skipping update.")
                continue

            # Check if the ticker exists in the database
            cursor.execute('SELECT ticker FROM Tickers_Info WHERE ticker = ?', (ticker,))
            result = cursor.fetchone()

            if not result:
                # Insert the new ticker into the database
                cursor.execute('INSERT INTO Tickers_Info (ticker) VALUES (?)', (ticker,))
                print(f"Inserted new ticker: {ticker}")

            print(f"Before Update: {ticker} =>", cursor.execute(
                'SELECT ticker, nicks_growth_rate, projected_profit_margin FROM Tickers_Info WHERE ticker = ?',
                (ticker,)).fetchall())

            print(f"Updating {ticker}: Growth Rate = {growth_rate}, Profit Margin = {profit_margin}")
            try:
                cursor.execute('''
                    UPDATE Tickers_Info
                    SET nicks_growth_rate = ?, projected_profit_margin = ?
                    WHERE ticker = ?;
                ''', (growth_rate, profit_margin, ticker))
                conn.commit()
                print(f"Updated {ticker}: Growth Rate = {growth_rate}, Profit Margin = {profit_margin}")

                print(f"After Update: {ticker} =>", cursor.execute(
                    'SELECT ticker, nicks_growth_rate, projected_profit_margin FROM Tickers_Info WHERE ticker = ?',
                    (ticker,)).fetchall())
            except sqlite3.Error as e:
                print(f"Error updating {ticker}: {e}")

    conn.close()
    # Wipe the file contents
    open(file_path, 'w').close()
    print(f"{file_path} processed and data wiped.")


    conn.close()
    # Wipe the file contents
    open(file_path, 'w').close()
    print(f"{file_path} processed and data wiped.")


def valuation_update(ticker, cursor, treasury_yield, marketcap, dashboard_data):
    db_path = "Stock Data.db"
    finviz_five_yr(ticker, cursor)
    combined_data, growth_values, current_price, forecast_data = fetch_financial_valuation_data(ticker, db_path)
    print('combined data', combined_data)

    if forecast_data.empty:
        print(f"No forecast data for {ticker}. Skipping...")
        return

    if growth_values.empty or (
            pd.isna(growth_values['nicks_growth_rate'].iloc[0]) and pd.isna(growth_values['FINVIZ_5yr_gwth'].iloc[0])):
        print("Growth values are missing or not valid. Skipping valuation.")
        return

    combined_data, nicks_fair_pe, finviz_fair_pe, nicks_fair_ps, _ = calculate_valuations(combined_data, growth_values,
                                                                                          treasury_yield, current_price,
                                                                                          marketcap)

    plot_valuation_chart(combined_data[['Year', 'Nicks_Valuation', 'Finviz_Valuation']], current_price, ticker,
                         growth_values)
    generate_valuation_tables(ticker, combined_data, growth_values, treasury_yield, current_price, nicks_fair_pe,
                              finviz_fair_pe, nicks_fair_ps)

    try:
        nicks_ttm_valuation = float(
            combined_data['Nicks_Valuation'].iloc[0].replace('$', '').replace('B', '').replace('M', '').replace('K',
                                                                                                                '').replace(
                ',', ''))
        nicks_forward_valuation = float(
            combined_data['Nicks_Valuation'].iloc[1].replace('$', '').replace('B', '').replace('M', '').replace('K',
                                                                                                                '').replace(
                ',', ''))
        nicks_ttm_value = ((nicks_ttm_valuation / current_price) - 1) * 100
        nicks_forward_value = ((nicks_forward_valuation / current_price) - 1) * 100

        if pd.notna(growth_values['FINVIZ_5yr_gwth'].iloc[0]):
            finviz_ttm_valuation = float(
                combined_data['Finviz_Valuation'].iloc[0].replace('$', '').replace('B', '').replace('M', '').replace(
                    'K', '').replace(',', ''))
            finviz_forward_valuation = float(
                combined_data['Finviz_Valuation'].iloc[1].replace('$', '').replace('B', '').replace('M', '').replace(
                    'K', '').replace(',', ''))
            finviz_ttm_value = ((finviz_ttm_valuation / current_price) - 1) * 100
            finviz_forward_value = ((finviz_forward_valuation / current_price) - 1) * 100
        else:
            finviz_ttm_valuation = "-"
            finviz_forward_valuation = "-"
            finviz_ttm_value = "-"
            finviz_forward_value = "-"
    except (ValueError, IndexError) as e:
        print(f"Error converting valuation values to float for ticker {ticker}: {e}")
        nicks_ttm_valuation = "-"
        nicks_forward_valuation = "-"
        nicks_ttm_value = "-"
        nicks_forward_value = "-"
        finviz_ttm_valuation = "-"
        finviz_forward_valuation = "-"
        finviz_ttm_value = "-"
        finviz_forward_value = "-"
        #commi

    dashboard_data.append([
        ticker,
        f"${current_price:.2f}",
        f"${nicks_ttm_valuation:.2f}" if isinstance(nicks_ttm_valuation, float) else nicks_ttm_valuation,
        f"{nicks_ttm_value:.1f}%" if isinstance(nicks_ttm_value, float) else nicks_ttm_value,
        f"${nicks_forward_valuation:.2f}" if isinstance(nicks_forward_valuation, float) else nicks_forward_valuation,
        f"{nicks_forward_value:.1f}%" if isinstance(nicks_forward_value, float) else nicks_forward_value,
        f"${finviz_ttm_valuation:.2f}" if isinstance(finviz_ttm_valuation, float) else finviz_ttm_valuation,
        f"{finviz_ttm_value:.1f}%" if isinstance(finviz_ttm_value, float) else finviz_ttm_value,
        f"${finviz_forward_valuation:.2f}" if isinstance(finviz_forward_valuation, float) else finviz_forward_valuation,
        f"{finviz_forward_value:.1f}%" if isinstance(finviz_forward_value, float) else finviz_forward_value
    ])

import requests
from bs4 import BeautifulSoup
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import os
import yfinance as yf
import csv

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
            try:
                estimate_value = float(estimate_value)  # Convert to float
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

        return combined_data, growth_value, current_price

def determine_valuation_method(combined_data):
    """Determines the valuation method based on the EPS values in the combined data."""
    if combined_data.empty or 'EPS' not in combined_data.columns:
        print("No data available to determine the valuation method.")
        return None

    if len(combined_data) >= 2 and all(combined_data.loc[:1, 'EPS'] < 0):
        print("First two EPS values are negative. Using sales valuation method.")
        return "sales valuation"
    else:
        print("EPS values are not consistently negative in the first two rows. Using EPS valuation method.")
        return "eps valuation"

def calculate_fair_pe(combined_data, growth_values, treasury_yield):
    treasury_yield = (float(treasury_yield) / 100)
    print('treasury yield', treasury_yield)

    nicks_growth_rate = float(growth_values['nicks_growth_rate'].iloc[0] if growth_values['nicks_growth_rate'].iloc[0] is not None else 0) / 100
    finviz_growth_rate = float(growth_values['FINVIZ_5yr_gwth'].iloc[0] if growth_values['FINVIZ_5yr_gwth'].iloc[0] is not None else 0) / 100

    nicks_fair_pe = ((nicks_growth_rate - treasury_yield + 1) ** 10) * 10
    print("nicks fair pe", nicks_fair_pe)
    finviz_fair_pe = ((finviz_growth_rate - treasury_yield + 1) ** 10) * 10
    print("finviz fair pe", finviz_fair_pe)

    combined_data['Nicks_Valuation'] = combined_data['EPS'] * nicks_fair_pe
    combined_data['Finviz_Valuation'] = combined_data['EPS'] * finviz_fair_pe

    print("Valuation calculations based on EPS valuation method:")
    print(combined_data[['Year', 'Nicks_Valuation', 'Finviz_Valuation']])

    return combined_data[['Year', 'Nicks_Valuation', 'Finviz_Valuation']], nicks_fair_pe, finviz_fair_pe

def calculate_fair_ps(combined_data, growth_values, treasury_yield):
    treasury_yield = (float(treasury_yield) / 100)
    print('treasury yield', treasury_yield)

    nicks_growth_rate = float(growth_values['nicks_growth_rate'].iloc[0] if growth_values['nicks_growth_rate'].iloc[0] is not None else 0) / 100
    finviz_growth_rate = float(growth_values['FINVIZ_5yr_gwth'].iloc[0] if growth_values['FINVIZ_5yr_gwth'].iloc[0] is not None else 0) / 100
    projected_profit_margin = float(growth_values['projected_profit_margin'].iloc[0] if growth_values['projected_profit_margin'].iloc[0] is not None else 0) / 100

    nicks_fair_ps = (((nicks_growth_rate - treasury_yield + 1) ** 10) * 10) * projected_profit_margin
    print("nicks fair ps", nicks_fair_ps)
    finviz_fair_ps = (((finviz_growth_rate - treasury_yield + 1) ** 10) * 10) * projected_profit_margin
    print("finviz fair ps", finviz_fair_ps)

    combined_data['Nicks_Valuation'] = combined_data['Revenue'] * nicks_fair_ps
    combined_data['Finviz_Valuation'] = combined_data['Revenue'] * finviz_fair_ps

    print("Valuation calculations based on Sales valuation method:")
    print(combined_data[['Year', 'Nicks_Valuation', 'Finviz_Valuation']])

    return combined_data[['Year', 'Nicks_Valuation', 'Finviz_Valuation']], nicks_fair_ps, finviz_fair_ps

def plot_valuation_chart(valuation_data, current_price, ticker, growth_value):
    """
    Plots the valuation data with the current stock price.

    Args:
        valuation_data (pd.DataFrame): DataFrame containing years, Nicks_Valuation, and Finviz_Valuation.
        current_price (float): Current price of the stock.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plotting the valuation lines if the growth values are not None
    if not pd.isna(growth_value['nicks_growth_rate'].iloc[0]):
        ax.plot(valuation_data['Year'], valuation_data['Nicks_Valuation'], label='Nicks Valuation', color='blue', marker='o')

    if not pd.isna(growth_value['FINVIZ_5yr_gwth'].iloc[0]):
        ax.plot(valuation_data['Year'], valuation_data['Finviz_Valuation'], label='Finviz Valuation', color='green', marker='o')

    # Adding a scatter plot for the current price at all points for visual comparison
    ax.plot(valuation_data['Year'], [current_price] * len(valuation_data), color='orange', label='Current Price')

    # Adding labels and title
    ax.set_xlabel('Year')
    ax.set_ylabel('Valuation')
    ax.set_title(f'Valuation Comparison for {ticker}')
    ax.legend()

    # Show the plot
    plt.grid(True)

    fig_path = f"charts/{ticker}_valuation_chart.png"
    plt.savefig(fig_path)
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

def format_number(value):
    """Formats numbers to billions, millions, or thousands with appropriate suffixes."""
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    elif value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    elif value >= 1_000:
        return f"{value / 1_000:.2f}K"
    else:
        return f"{value:.2f}"

def format_currency(value):
    """Formats numbers as currency with billions, millions, or thousands with appropriate suffixes."""
    if value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"
    elif value >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    elif value >= 1_000:
        return f"${value / 1_000:.2f}K"
    else:
        return f"${value:.2f}"

def generate_valuation_tables(ticker, combined_data, growth_values, treasury_yield, current_price, nicks_fair_val, finviz_fair_val, valuation_method):
    # Ensure treasury_yield is a float
    treasury_yield = float(treasury_yield)

    # Fetch stock data
    current_price, pe_ratio, price_to_sales, forward_pe_ratio = fetch_stock_data(ticker)

    # Format the necessary values
    current_price_formatted = f"${current_price:.2f}"
    nicks_fair_val_formatted = f"{nicks_fair_val:.0f}"
    finviz_fair_val_formatted = f"{finviz_fair_val:.0f}"
    treasury_yield_formatted = f"{treasury_yield:.1f}%"
    nicks_growth_rate_formatted = f"{growth_values['nicks_growth_rate'].iloc[0]:.0f}%"
    finviz_growth_rate_formatted = f"{growth_values['FINVIZ_5yr_gwth'].iloc[0]:.0f}%"

    if valuation_method == "eps valuation":
        current_valuation_metric = f"{pe_ratio:.1f}" if pe_ratio else "N/A"
        valuation_metric_label = "Current P/E"
    else:
        current_valuation_metric = f"{price_to_sales:.1f}" if price_to_sales else "N/A"
        valuation_metric_label = "Current P/S"

    table_1_data = {
        "Share Price": [current_price_formatted],
        "Treasury Yield": [treasury_yield_formatted],
        "Growth Rate": [f"Nicks: {nicks_growth_rate_formatted} Finviz: {finviz_growth_rate_formatted}"],
        "Fair P/E": [f"Nicks: {nicks_fair_val_formatted} Finviz: {finviz_fair_val_formatted}"],
        valuation_metric_label: [current_valuation_metric]
    }
    table_1_df = pd.DataFrame(table_1_data)

    table_1_html = table_1_df.to_html(index=False, escape=False)
    table_1_path = os.path.join('charts', f"{ticker}_valuation_info.html")
    with open(table_1_path, "w") as file:
        file.write(table_1_html)

    # Apply formatting to valuation data
    combined_data['Nicks_Valuation'] = combined_data['Nicks_Valuation'].apply(format_currency)
    combined_data['Finviz_Valuation'] = combined_data['Finviz_Valuation'].apply(format_currency)
    combined_data['Nicks vs Share Price'] = combined_data['Nicks_Valuation'].apply(lambda x: f"{((float(x.strip('$BMK')) / current_price - 1) * 100):.1f}%")
    combined_data['Finviz vs Share Price'] = combined_data['Finviz_Valuation'].apply(lambda x: f"{((float(x.strip('$BMK')) / current_price - 1) * 100):.1f}%")

    if valuation_method == "eps valuation":
        combined_data['Basis'] = combined_data['EPS'].apply(lambda x: f"{x:.2f} EPS" if pd.notna(x) else "")
    else:  # sales valuation
        combined_data['Basis'] = combined_data['Revenue'].apply(lambda x: f"{x:.2f} Revenue" if pd.notna(x) else "")

    table_2_data = {
        "Basis": combined_data['Basis'],
        "Year": combined_data['Year'],
        "Nicks Valuation": combined_data['Nicks_Valuation'],
        "Nicks vs Share Price": combined_data['Nicks vs Share Price'],
        "Finviz Valuation": combined_data['Finviz_Valuation'],
        "Finviz vs Share Price": combined_data['Finviz vs Share Price']
    }
    table_2_df = pd.DataFrame(table_2_data)

    table_2_html = table_2_df.to_html(index=False, escape=False)
    table_2_path = os.path.join('charts', f"{ticker}_valuation_table.html")
    with open(table_2_path, "w") as file:
        file.write(table_2_html)

    print(f"Saved valuation info to {table_1_path} and valuation table to {table_2_path}")



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
            growth_rate = float(growth_rate)
            if profit_margin is not None:
                profit_margin = float(profit_margin)

            print(f"Before Update: {ticker} =>", cursor.execute('SELECT ticker, nicks_growth_rate, projected_profit_margin FROM Tickers_Info WHERE ticker = ?', (ticker,)).fetchall())

            print(f"Updating {ticker}: Growth Rate = {growth_rate}, Profit Margin = {profit_margin}")
            try:
                cursor.execute('''
                    UPDATE Tickers_Info
                    SET nicks_growth_rate = ?, projected_profit_margin = ?
                    WHERE ticker = ?;
                ''', (growth_rate, profit_margin, ticker))
                conn.commit()
                print(f"Updated {ticker}: Growth Rate = {growth_rate}, Profit Margin = {profit_margin}")

                print(f"After Update: {ticker} =>", cursor.execute('SELECT ticker, nicks_growth_rate, projected_profit_margin FROM Tickers_Info WHERE ticker = ?', (ticker,)).fetchall())
            except sqlite3.Error as e:
                print(f"Error updating {ticker}: {e}")

    conn.close()
    # Wipe the file contents
    open(file_path, 'w').close()
    print(f"{file_path} processed and data wiped.")




def valuation_update(ticker, cursor, treasury_yield):
    db_path = "Stock Data.db"
    """Updates the Finviz 5-year EPS growth data for the given ticker and determines the valuation method."""
    finviz_five_yr(ticker, cursor)
    combined_data, growth_values, current_price = fetch_financial_valuation_data(ticker, db_path)

    if growth_values.empty or pd.isna(growth_values['nicks_growth_rate'].iloc[0]) or pd.isna(growth_values['FINVIZ_5yr_gwth'].iloc[0]):
        print("Growth values are missing or not valid. Skipping valuation.")
    else:
        valuation_method = determine_valuation_method(combined_data)
        print(f"Valuation Method for {ticker}: {valuation_method}")
        if valuation_method == "eps valuation":
            valuation_data, nicks_fair_pe, finviz_fair_pe = calculate_fair_pe(combined_data, growth_values, treasury_yield)
            plot_valuation_chart(valuation_data, current_price, ticker, growth_values)
            generate_valuation_tables(ticker, combined_data, growth_values, treasury_yield, current_price,
                                      nicks_fair_pe, finviz_fair_pe, valuation_method)
        elif valuation_method == "sales valuation":
            valuation_data, nicks_fair_ps, finviz_fair_ps = calculate_fair_ps(combined_data, growth_values, treasury_yield)
            plot_valuation_chart(valuation_data, current_price, ticker, growth_values)
            generate_valuation_tables(ticker, combined_data, growth_values, treasury_yield, current_price,
                                      nicks_fair_ps, finviz_fair_ps, valuation_method)



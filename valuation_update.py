# valuation_update.py

import requests
from bs4 import BeautifulSoup
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

def finviz_five_yr(ticker, cursor):
    """Fetches and stores the 5-year EPS growth percentage from Finviz into the database."""
    url = f'https://finviz.com/quote.ashx?t={ticker}'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        # XPath equivalent: //td[text()='EPS next 5Y']/following-sibling::td
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
    import yfinance as yf

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
        SELECT nicks_growth_rate, FINVIZ_5yr_gwth
        FROM Tickers_Info
        WHERE ticker = ?;
        """
        growth_value = pd.read_sql_query(growth_query, conn, params=(ticker,))
        print("growth value",growth_value)


        # Combine TTM and forecast data into a single DataFrame
        combined_data = pd.concat([ttm_data, forecast_data]).reset_index(drop=True)

        return combined_data, growth_value, current_price


def determine_valuation_method(combined_data):
    """
    Determines the valuation method based on the EPS values in the combined data.

    Args:
        combined_data (pd.DataFrame): DataFrame containing the combined financial data including EPS.

    Returns:
        str: The valuation method based on EPS.
    """
    # Check if the DataFrame is empty or if the EPS column doesn't exist
    if combined_data.empty or 'EPS' not in combined_data.columns:
        print("No data available to determine the valuation method.")
        return None

    # Check the EPS values in the first two rows
    if len(combined_data) >= 2 and all(combined_data.loc[:1, 'EPS'] < 0):
        print("First two EPS values are negative. Using sales valuation method.")
        return "sales valuation"
    else:
        print("EPS values are not consistently negative in the first two rows. Using EPS valuation method.")
        return "eps valuation"

def calculate_fair_pe(combined_data, growth_values, treasury_yield):
    # Ensure treasury_yield is a float
    treasury_yield = (float(treasury_yield)/100)
    print('treasury yield', treasury_yield)

    # Handle possible None values and ensure type correctness
    nicks_growth_rate = float(growth_values['nicks_growth_rate'].iloc[0] if growth_values['nicks_growth_rate'].iloc[0] is not None else 0)
    nicks_growth_rate = nicks_growth_rate/100
    finviz_growth_rate = float(growth_values['FINVIZ_5yr_gwth'].iloc[0] if growth_values['FINVIZ_5yr_gwth'].iloc[0] is not None else 0)
    finviz_growth_rate = finviz_growth_rate/100
    # Calculate fair P/E ratios
    nicks_fair_pe = ((nicks_growth_rate - treasury_yield + 1)**10) * 10
    print("nicks fair pe",nicks_fair_pe)
    finviz_fair_pe = ((finviz_growth_rate - treasury_yield + 1)**10) * 10
    print("finviz fair pe", finviz_fair_pe)

    # Calculate valuations
    combined_data['Nicks_Valuation'] = combined_data['EPS'] * nicks_fair_pe
    combined_data['Finviz_Valuation'] = combined_data['EPS'] * finviz_fair_pe

    print("Valuation calculations based on EPS valuation method:")
    print(combined_data[['Year', 'Nicks_Valuation', 'Finviz_Valuation']])

    return combined_data[['Year', 'Nicks_Valuation', 'Finviz_Valuation']]


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
        ax.plot(valuation_data['Year'], valuation_data['Nicks_Valuation'], label='Nicks Valuation', color='blue',
                marker='o')

    if not pd.isna(growth_value['FINVIZ_5yr_gwth'].iloc[0]):
        ax.plot(valuation_data['Year'], valuation_data['Finviz_Valuation'], label='Finviz Valuation', color='green',
                marker='o')

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


def valuation_update(ticker, cursor, treasury_yield):
    db_path = "Stock Data.db"
    """Updates the Finviz 5-year EPS growth data for the given ticker and determines the valuation method."""
    finviz_five_yr(ticker, cursor)
    combined_data, growth_value, current_price = fetch_financial_valuation_data(ticker, db_path)  # Ensure this function returns combined_data

    # Check if growth values are present
    if not growth_value.empty and not (pd.isna(growth_value['nicks_growth_rate'].iloc[0]) and pd.isna(growth_value['FINVIZ_5yr_gwth'].iloc[0])):
        valuation_method = determine_valuation_method(combined_data)  # Use the combined_data to determine the valuation method
        print(f"Valuation Method for {ticker}: {valuation_method}")
        if valuation_method == "eps valuation":
            valuation_data = calculate_fair_pe(combined_data, growth_value, treasury_yield)
            plot_valuation_chart(valuation_data, current_price, ticker, growth_value)
    else:
        print("Growth values are missing. Skipping valuation.")

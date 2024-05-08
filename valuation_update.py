# valuation_update.py

import requests
from bs4 import BeautifulSoup
import sqlite3

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


def valuation_update(ticker, cursor):
    """Updates the Finviz 5-year EPS growth data for the given ticker."""
    finviz_five_yr(ticker, cursor)

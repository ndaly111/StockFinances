import yfinance as yf
import os
from bs4 import BeautifulSoup

def fetch_stock_data(ticker, treasury_yield):
    """
    Safely fetches stock info using yfinance, avoiding 'NoneType' errors and
    also handling fallback logic for currentPrice.
    """
    try:
        stock = yf.Ticker(ticker)
        raw_info = stock.info  # This can return None or raise an error
        if raw_info is not None:
            info = raw_info
        else:
            info = {}
            print(f"Warning: yfinance returned None info for ticker '{ticker}'")
    except Exception as e:
        print(f"Error retrieving market data for {ticker}: {e}")
        # Fallback to an empty dict to avoid crashing
        info = {}

    # Safely pull values from the 'info' dictionary
    current_price = info.get('currentPrice', None)
    # Fallback attempts
    if current_price is None:
        current_price = info.get('regularMarketPrice', None)
    if current_price is None:
        current_price = info.get('previousClose', None)
    if current_price is None:
        bid = info.get('bid', None)
        ask = info.get('ask', None)
        if bid and ask:
            current_price = (bid + ask) / 2

    forward_eps = info.get('forwardEps', None)
    pe_ratio = info.get('trailingPE', None)
    price_to_book = info.get('priceToBook', None)
    marketcap = info.get('marketCap', None)

    if current_price is not None and forward_eps:
        forward_pe_ratio = current_price / forward_eps
    else:
        forward_pe_ratio = None

    # Ensure treasury_yield is a float and convert from percentage to decimal if possible
    if treasury_yield and treasury_yield != '-':
        treasury_yield = float(treasury_yield) / 100
    else:
        treasury_yield = None

    # Calculate implied growth for trailing P/E
    implied_growth = calculate_implied_growth(pe_ratio, treasury_yield) if pe_ratio is not None else '-'
    implied_growth_formatted = f"{implied_growth * 100:.1f}%" if implied_growth != '-' else 'N/A'

    # Calculate implied growth for forward P/E
    implied_forward_growth = calculate_implied_growth(forward_pe_ratio, treasury_yield) \
        if forward_pe_ratio is not None else '-'
    implied_forward_growth_formatted = f"{implied_forward_growth * 100:.1f}%" if implied_forward_growth != '-' else '-'

    # Format close price or placeholder if None
    formatted_close_price = f"${current_price:.2f}" if current_price is not None else '-'

    # Create the data dictionary
    data = {
        'Close Price': formatted_close_price,
        'Market Cap': marketcap,
        'P/E Ratio': "{:.1f}".format(pe_ratio) if pe_ratio is not None else '-',
        'Forward P/E Ratio': "{:.1f}".format(forward_pe_ratio) if forward_pe_ratio is not None else '-',
        'Implied Growth*': implied_growth_formatted,
        'Implied Forward Growth*': implied_forward_growth_formatted,
        'P/B Ratio': "{:.1f}".format(price_to_book) if price_to_book is not None else '-',
    }

    return data, marketcap


def calculate_implied_growth(pe_ratio, treasury_yield):
    """
    Calculates an 'implied growth' figure based on P/E ratio and risk-free rate.
    """
    if pe_ratio is None or treasury_yield is None:
        return '-'
    else:
        # Simple example of a custom implied-growth formula
        return ((pe_ratio / 10) ** (1/10)) + treasury_yield - 1


def format_number(value):
    """
    Formats large numbers into a more readable form (e.g., in millions, billions, or trillions).
    """
    if value is None:
        return "N/A"
    if abs(value) >= 1e12:
        return f"${value / 1e12:.2f}T"
    elif abs(value) >= 1e9:
        return f"${value / 1e9:.2f}B"
    elif abs(value) >= 1e6:
        return f"${value / 1e6:.2f}M"
    else:
        return f"${value:.2f}"


def prepare_data_for_display(ticker, treasury_yield):
    """
    Fetches the stock data (price, ratios, etc.) safely for further processing.
    """
    fetched_data, marketcap = fetch_stock_data(ticker, treasury_yield)
    return fetched_data, marketcap


def generate_html_table(data, ticker):
    """
    Generates a minimalist horizontal HTML table from a dictionary of data
    and saves it to an HTML file, applying minor styling.
    """
    html_content = """
    <style>
        table {
            width: 80%;
            margin-left: auto;
            margin-right: auto;
            border-collapse: collapse;
            text-align: center;
            font-family: 'Arial', sans-serif;
        }
        th, td {
            padding: 8px 12px;
        }
    </style>
    <table>
    <tr>"""

    # Headers
    for key in data:
        html_content += f"<th>{key}</th>"
    html_content += "</tr><tr>"

    # Values
    for key, value in data.items():
        if key == 'Market Cap' and isinstance(value, (int, float)):
            formatted_value = format_number(value)
        else:
            formatted_value = value
        html_content += f"<td>{formatted_value}</td>"
    html_content += "</tr></table>"

    file_path = f"charts/{ticker}_ticker_info.html"
    with open(file_path, 'w') as file:
        file.write(html_content)

    return file_path


if __name__ == "__main__":
    # Example usage
    ticker = 'AAPL'
    treasury_yield = '3.5'  # e.g. 3.5 means 3.5% yield

    prepared_data, marketcap = prepare_data_for_display(ticker, treasury_yield)
    html_file_path = generate_html_table(prepared_data, ticker)
    print(f"HTML content has been written to {html_file_path}")

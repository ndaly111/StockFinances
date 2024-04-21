import yfinance as yf
import os

from bs4 import BeautifulSoup


def fetch_stock_data(ticker, treasury_yield):
    stock = yf.Ticker(ticker)
    current_price = stock.info.get('currentPrice')
    forward_eps = stock.info.get('forwardEps')
    pe_ratio = stock.info.get('trailingPE', None)
    forward_pe_ratio = current_price / forward_eps if forward_eps else None

    # Ensure treasury_yield is a float and convert from percentage to decimal
    treasury_yield = float(treasury_yield) / 100 if treasury_yield and treasury_yield != '-' else None

    # Calculate implied growth for trailing P/E
    implied_growth = calculate_implied_growth(pe_ratio, treasury_yield) if pe_ratio is not None else '-'
    implied_growth_formatted = f"{implied_growth * 100:.1f}%" if implied_growth != '-' else 'N/A'

    # Calculate implied growth for forward P/E
    implied_forward_growth = calculate_implied_growth(forward_pe_ratio,
                                                      treasury_yield) if forward_pe_ratio is not None else '-'
    implied_forward_growth_formatted = f"{implied_forward_growth * 100:.1f}%" if implied_forward_growth != '-' else '-'

    data = {
        'Close Price': current_price,
        'Market Cap': stock.info.get('marketCap'),
        'P/E Ratio': "{:.2f}".format(pe_ratio) if pe_ratio is not None else '-',
        'Forward P/E Ratio': "{:.2f}".format(forward_pe_ratio) if forward_pe_ratio is not None else '-',
        'Implied Growth*': implied_growth_formatted,
        'Implied Forward Growth*': implied_forward_growth_formatted,
    }
    return data


def calculate_implied_growth(pe_ratio, treasury_yield):
    if pe_ratio is None or treasury_yield is None:
        return '-'
    else:
        return ((pe_ratio / 10) ** (1/10)) + treasury_yield - 1


def format_number(value):
    """
    Formats large numbers into a more readable form (e.g., in millions, billions, or trillions).

    Args:
        value (int or float): The value to format.

    Returns:
        str: Formatted string representing the number.
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
    fetched_data = fetch_stock_data(ticker, treasury_yield)
    return fetched_data


def generate_html_table(data, ticker):
    """
    Generates a minimalist horizontal HTML table from a dictionary of data and saves it to an HTML file.

    Args:
        data (dict): Data to be formatted into an HTML table.
        ticker (str): Stock ticker symbol used to name the file.

    Returns:
        str: The filename of the saved HTML file.
    """
    # Define minimalist CSS for the table
    html_content = """
    <style>
        table {
            width: 80%;
            margin-left: auto;
            margin-right: auto;
            border-collapse: collapse;
            text-align: center;
            font-family: 'Arial', sans-serif; /* Using Arial as an example of a sans-serif font */
        }
        th, td {
            padding: 8px 12px;
        }
    </style>
    <table>
    <tr>"""

    # Add headers to the first row
    for key in data:
        html_content += f"<th>{key}</th>"
    html_content += "</tr><tr>"

    # Add values to the second row, only format 'Market Cap'
    for key, value in data.items():
        if key == 'Market Cap' and isinstance(value, (int, float)):
            formatted_value = format_number(value)
        else:
            formatted_value = value
        html_content += f"<td>{formatted_value}</td>"
    html_content += "</tr></table>"

    # Save the HTML content to a file
    file_path = f"charts/{ticker}_ticker_info.html"
    with open(file_path, 'w') as file:
        file.write(html_content)

    return file_path



if __name__ == "__main__":
    ticker = 'AAPL'  # Example ticker
    user_pe = 15.0  # Example fair P/E provided by the user
    user_ps = 4.0  # Example fair P/S provided by the user
    growth_rate = '5%'  # Example growth rate provided by the user

    # Prepare and fetch data
    prepared_data = prepare_data_for_display(ticker, treasury_yield)

    # Generate HTML table and save to file
    html_file_path = generate_html_table(prepared_data, ticker)
    print(f"HTML content has been written to {html_file_path}")

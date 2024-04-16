import yfinance as yf
import os


def fetch_stock_data(ticker):
    """
    Fetches financial data for a given ticker using yfinance.

    Args:
        ticker (str): The stock ticker symbol.

    Returns:
        dict: A dictionary containing the fetched stock data.
    """
    stock = yf.Ticker(ticker)
    data = {
        'Close Price': stock.info.get('previousClose'),
        'Market Cap': stock.info.get('marketCap'),
        'P/E Ratio': "{:.2f}".format(stock.info.get('trailingPE')) if stock.info.get(
            'trailingPE') is not None else 'N/A',
        'P/S Ratio': "{:.2f}".format(stock.info.get('priceToSalesTrailing12Months')) if stock.info.get(
            'priceToSalesTrailing12Months') is not None else 'N/A'
    }
    return data


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


def prepare_data_for_display(ticker, user_pe, user_ps, growth_rate):
    """
    Integrates fetched stock data with user-provided estimates and prepares it for display.

    Args:
        ticker (str): The stock ticker symbol.
        user_pe (float): User-provided fair P/E ratio.
        user_ps (float): User-provided fair P/S ratio.
        growth_rate (str): User-provided estimated growth rate.

    Returns:
        dict: A dictionary with combined stock data and user inputs.
    """
    fetched_data = fetch_stock_data(ticker)
    fetched_data.update({
        'Fair P/E': user_pe,
        'Fair P/S': user_ps,
        'Estimated Growth Rate': growth_rate
    })
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
    prepared_data = prepare_data_for_display(ticker, user_pe, user_ps, growth_rate)

    # Generate HTML table and save to file
    html_file_path = generate_html_table(prepared_data, ticker)
    print(f"HTML content has been written to {html_file_path}")

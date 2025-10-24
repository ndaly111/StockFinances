#start of html_generator.py

from jinja2 import Environment, FileSystemLoader
import logging
import os
import pandas as pd
import yfinance as yf



env = Environment(loader=FileSystemLoader(os.path.dirname(os.path.abspath(__file__))))

# gate to match Aug 10 rollback
VERSION = "SEGMENTS v2025-08-10b"

logger = logging.getLogger(__name__)


def format_to_millions(value):
    """Format a numeric value into millions with a dollar sign and commas."""
    try:
        value_in_millions = value / 1e6  # Convert to millions
        formatted_value = f"${value_in_millions:,.0f}M"
        logger.debug("Formatted value %s to %s", value, formatted_value)
        return formatted_value
    except (TypeError, ValueError):
        logger.debug("Unable to format value %s to millions; returning 'N/A'", value)
        return "N/A"



def prepare_financial_data(df):
    # Define columns that need to be converted to numeric
    financial_columns = ['Revenue', 'Net_Income', 'EPS']

    # Convert financial figure columns from string to float
    for column in financial_columns:
        # Remove dollar signs, commas, and "M" (if your data represents millions, you might want to divide by 1e6 to convert to actual values)
        df[column] = df[column].replace('[\$,M]', '', regex=True).astype(float)
        # If your data uses "M" to represent millions, you might want to scale the numbers accordingly
        # df[column] = df[column] / 1e6  # Uncomment this line if necessary

    # Your existing logic to calculate percentage changes
    for column in financial_columns:
        change_column_name = f'{column}_Change'
        # Calculate percentage change and store in a new column
        df[change_column_name] = df[column].pct_change(periods=-1) * 100

    return df








def ensure_template_exists(template_path, template_content):
    logger.debug("Ensuring template exists at %s", template_path)
    if not os.path.exists(template_path):
        with open(template_path, 'w') as file:
            file.write(template_content)
        logger.info("Created default template at %s", template_path)



# Define the content of your template.html with the table style placeholder
template_html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Financial Charts</title>
    <style>
        body {
            font-family: 'Arial', sans-serif;
            color: #333;
            margin: 0;
            padding: 0;
            overflow-x: hidden; /* Prevent horizontal scrolling */
        }
        .navigation {
            text-align: center; /* Center navigation links */
            padding: 10px 0; /* Padding for the navigation bar */
            background: #f2f2f2; /* Light grey background for the navigation bar */
        }
        .carousel-container {
            white-space: nowrap; /* Ensure the charts don't wrap */
            overflow-x: auto; /* Enable horizontal scrolling */
            -webkit-overflow-scrolling: touch; /* Smooth scrolling on iOS */
            margin: 20px auto; /* Center horizontally */
            padding: 10px 0; /* Padding to prevent content touching the edges */
            scroll-snap-type: x mandatory; /* Enables snap scrolling on the x-axis */
            display: flex; /* Use flex display to manage child elements */
            flex-direction: row; /* Arrange items in a row */
            gap: 20px; /* This can replace margin-right on carousel-item for spacing between items */
        }
        .carousel-item {
            display: inline-block; /* Display items in a line */
            width: 100%; /* Full width of the container */
            vertical-align: top; /* Align items to the top */
            margin-right: 20px; /* Margin between items */
        }
        .chart-container, .financial-table-container, .balance-sheet-container {
            text-align: center; /* Center content vertically */
            margin-bottom: 20px; /* Space below each container */
            scroll-snap-align: center; /* Aligns the snapping point to the center of the element */
            /* It's important to give the item a defined width, less than 100% if you want it to be less than full width */
            width: 90%; /* Example width, adjust as necessary for your layout */
            margin-left: auto; /* Centers the item in the carousel if width is less than 100% */
            margin-right: auto; /* Centers the item in the carousel if width is less than 100% */
        }
        .chart, .financial-table-container img, .balance-sheet-container img {
            max-width: 100%; /* Maximum width of images */
            height: auto; /* Maintain aspect ratio */
        }
        .balance-sheet-container {
            display: flex; /* Flex container for layout */
            justify-content: space-between; /* Space between child elements */
            flex-wrap: wrap; /* Allow items to wrap if needed */
        }
        .balance-sheet-table, .balance-sheet-chart {
            flex: 1; /* Allow flex items to grow to fill available space */
            max-width: calc(50% - 10px); /* Maximum width with spacing */
            box-sizing: border-box; /* Include padding and border in width calculation */
        }
        .balance-sheet-chart img {
            max-width: 80%; /* Limit width of balance sheet charts */
        }
        .home-button {
            padding: 10px 20px;
            font-size: 18px;
            background-color: #008CBA;
            color: white;
            border: none;
            border-radius: 5px;
            text-decoration: none;
            display: inline-block;
            margin: 5px;
            cursor: pointer;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            transition: background-color 0.3s, box-shadow 0.3s;
        }
        .home-button:hover {
            background-color: #003f4b;
            box-shadow: 0 6px 12px rgba(0,0,0,0.3);
        }
    </style>
</head>
<body>
    <div id="top-of-page"></div>
    <div class="navigation">
        {{ nav_links | safe }}
    </div>

    {% for ticker_data in tickers_data %}
        <div class="ticker-section" id="{{ ticker_data.ticker }}">
            <h2>{{ ticker_data.ticker }}</h2>
            <a href="#top-of-page" class="home-button">Home</a>

            <div>
                {{ ticker_data.ticker_info | safe }}
                <img src="{{ ticker_data.revenue_net_income_chart_path }}" alt="Revenue and Net Income Chart" align="center">
                <img src="{{ ticker_data.eps_chart_path }}" alt="EPS Chart" align="center">
                {{ ticker_data.financial_table | safe }}
            </div>

            <div class="carousel-container">
                <div class="carousel-item">
                    <img src="{{ ticker_data.forecast_rev_net_chart_path }}" alt="Revenue and Net Income Forecast Chart">
                </div>
                <div class="carousel-item">
                    <img src="{{ ticker_data.forecast_eps_chart_path }}" alt="EPS Forecast Chart">
                </div>
                <div class="carousel-item">
                    {{ ticker_data.yoy_growth_table_html | safe }}
                </div>
            </div>

            <div class="balance-sheet-container">
                <div class="balance-sheet-table">
                    {{ ticker_data.balance_sheet_table_html | safe }}
                </div>
                <div class="balance-sheet-chart">
                    <img src="{{ ticker_data.balance_sheet_chart_path }}" alt="{{ ticker_data.ticker }} Balance Sheet Chart" style="max-width: 80%;">
                </div>
            </div>
            <hr>
        </div>
    {% endfor %}
</body>
</html>
"""
logger.debug("Template HTML definition loaded.")

# Path to your template file
template_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'template.html')
logger.debug("Template path resolved to %s", template_file_path)


# Ensure the template exists before rendering
ensure_template_exists(template_file_path, template_html_content)
logger.debug("Verified template exists at %s", template_file_path)





def calculate_and_format_changes(df):
    # Ensure the DataFrame is sorted by 'Date' to calculate changes correctly
    df.sort_values('Date', ascending=True, inplace=True)

    # Define the columns to calculate yearly changes
    financial_columns = ['Revenue', 'Net_Income', 'EPS']

    # Convert columns to float if they are not already, assuming they are strings with $ and M symbols
    for column in financial_columns:
        if df[column].dtype == 'object':
            df[column] = df[column].replace('[\$,M]', '', regex=True).astype(float) * 1e6

    # Calculate and format the yearly changes
    for column in financial_columns:
        change_column = f"{column}_Change"
        df[change_column] = df[column].pct_change(fill_method=None) * 100

        # Format the changes as percentages with one decimal place
        df[change_column] = df[change_column].apply(lambda x: f"{x:.1f}%" if pd.notnull(x) else "N/A")

    # Format the financial numbers in millions and EPS as specified
    for column in financial_columns:
        if column != 'EPS':  # For Revenue and Net_Income
            df[column] = df[column].apply(lambda x: f"${x/1e6:,.0f}M")
        else:  # For EPS
            df[column] = df[column].apply(lambda x: f"${x:,.2f}")

    return df





def print_dataframe_to_console(df, message):
    logger.debug("%s:\n%s", message, df)


def get_file_content_or_placeholder(file_path, placeholder="No data available", is_binary=False):
    """Gets the content of a file if it exists, otherwise returns a placeholder.
       For binary files (like images), set is_binary to True."""
    try:
        if is_binary:
            with open(file_path, 'rb') as file:  # Open as binary
                return file.read()
        else:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
    except FileNotFoundError:
        return f"<div style='text-align: center;'><strong>{placeholder}</strong></div>"
    except UnicodeDecodeError as e:
        return f"<div style='text-align: center;'><strong>Error reading file: {e}</strong></div>"



# Function to create HTML content
def create_html_for_tickers(current_tickers, financial_data, charts_output_dir, html_file='index.html'):
    charts_output_dir = "charts/"
    logger.debug("Creating HTML for %d tickers", len(current_tickers))
    # Ensure charts_output_dir ends with a slash
    charts_output_dir = charts_output_dir.rstrip('/') + '/'

    # Sorting the tickers
    sorted_tickers = sorted(current_tickers)

    # Building navigation links - Update to ensure correct IDs and styling
    nav_links = " | ".join(
        f'<a href="#{ticker}" class="home-button">{ticker}</a>' for ticker in sorted_tickers
    )

    # Debug print to check nav_links content
    logger.debug("Generated navigation links for %d tickers", len(sorted_tickers))
    # Load the template
    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template('template.html')
    logger.debug("Loading HTML template")

    # Define your table styles
    table_styles = """
    <style>
    .financial-data {
        font-size: 14px;
        font-family: 'Arial', sans-serif;
        width: 100%;
        border-collapse: collapse;
        table-layout: fixed;
    }
    .financial-data th, .financial-data td {
        text-align: right;
        padding: 8px;
        border: none;
    }
    .financial-data th {
        background-color: #f2f2f2;
    }
    .financial-data tbody tr:hover {
        background-color: #f5f5f5;
    }
    img.chart {
        max-width: 100%;
        height: auto;
        display: block;
        margin: 10px auto;
    }
    .balance-sheet-container {
        display: flex;
        justify-content: space-between;
        margin-bottom: 20px;
    }
    .balance-sheet-table, .balance-sheet-chart {
        flex: 1;
        padding: 10px;
    }
    .balance-sheet-table, .balance-sheet-chart {
        overflow-x: auto;
    }
    @media (max-width: 768px) {
        .balance-sheet-container {
            flex-direction: column;
        }
    }
    </style>
    """

    data_for_rendering = {
        'tickers_data': [],
        'nav_links': nav_links,
        'table_styles': table_styles
    }

    for ticker in sorted_tickers:
        if ticker in financial_data:
            df = financial_data[ticker]
            df = calculate_and_format_changes(df)
            rendered_table = df.to_html(classes="financial-data", border=0, na_rep='N/A')

            ticker_data = {
                'ticker': ticker,
                'ticker_info': open(f"{charts_output_dir}{ticker}_ticker_info.html").read() if os.path.exists(
                    f"{charts_output_dir}{ticker}_ticker_info.html") else "Ticker info not available",
                'revenue_net_income_chart_path': f"{charts_output_dir}{ticker}_revenue_net_income_chart.png",
                'eps_chart_path': f"{charts_output_dir}{ticker}_eps_chart.png",
                'financial_table': rendered_table,
                'forecast_rev_net_chart_path': get_file_content_or_placeholder(
                    f"{charts_output_dir}{ticker}_Revenue_Net_Income_Forecast.png",
                    "No Revenue & Net Income Forecast data available"),
                'forecast_eps_chart_path': get_file_content_or_placeholder(
                    f"{charts_output_dir}{ticker}_EPS_Forecast.png", "No EPS Forecast data available"),
                'yoy_growth_table_html': get_file_content_or_placeholder(
                    f"{charts_output_dir}{ticker}_yoy_growth_tbl.html", "No Year-Over-Year Growth data available"),
                'balance_sheet_chart_path': f"{charts_output_dir}{ticker}_balance_sheet_chart.png",
                'balance_sheet_table_html': open(
                    f"{charts_output_dir}{ticker}_balance_sheet_table.html").read() if os.path.exists(
                    f"{charts_output_dir}{ticker}_balance_sheet_table.html") else "Balance sheet data not available"
            }

            data_for_rendering['tickers_data'].append(ticker_data)

    html_content = template.render(data_for_rendering)

    with open(html_file, 'w') as file:
        file.write(html_content)

    logger.info("HTML content has been written to %s", html_file)

    return html_content

#end of html_generator.py

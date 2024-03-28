#start of html_generator.py

from jinja2 import Environment, FileSystemLoader
import os
import pandas as pd
import yfinance as yf



env = Environment(loader=FileSystemLoader(os.path.dirname(os.path.abspath(__file__))))

def format_to_millions(value):
    print("html_generator 1 format to millions")
    """
    Formats a numerical value to a string representing the value in millions
    with a dollar sign and commas.
    """
    print("html generator 1 formatting to millions", value)
    try:
        print("---value",value)
        # Assume value is already a float representing the total amount (not in millions)
        value_in_millions = value / 1e6  # Convert to millions
        formatted_value = f"${value_in_millions:,.0f}M"
        return formatted_value
    except ValueError:
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


def append_yearly_changes(df):
    """
    Appends yearly changes for 'Revenue', 'Net_Income', and 'EPS' as formatted strings
    to the DataFrame. Changes are calculated year-over-year and formatted as percentages
    and values in millions.

    Args:
        df (pd.DataFrame): DataFrame containing the financial data.

    Returns:
        pd.DataFrame: DataFrame with appended yearly change columns, formatted as strings.
    """
    # Ensure the DataFrame is sorted by 'Date' to calculate changes correctly
    df.sort_values('Date', ascending=False, inplace=True)

    # Define the columns to calculate yearly changes
    financial_columns = ['Revenue', 'Net_Income', 'EPS']

    for column in financial_columns:
        change_column_name = f'{column}_Change'
        # Calculate year-over-year percentage change
        df[change_column_name] = df[column].pct_change(periods=-1) * 100

        # Format the changes as strings with percentage sign and two decimals
        # For values, divide by 1 million to represent values in millions
        df[change_column_name] = df[change_column_name].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A")

    return df





def ensure_template_exists(template_path, template_content):
    print("html generator 3 ensuring template exists")
    if not os.path.exists(template_path):
        with open(template_path, 'w') as file:
            file.write(template_content)



# Define the content of your template.html with the table style placeholder
template_html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Financial Charts</title>
    {{ table_styles | safe }}
    <style>
        body {
            font-family: 'Arial', sans-serif; /* Example font */
            color: #333;
        }
        .ticker-section h2 {
            display: inline-block;
            margin-right: 10px; /* Adjust the spacing between the header and the button */
        } /* Closing bracket added here */

        .home-button {
            padding: 10px 20px; /* Increase padding for a larger button */
            font-size: 18px; /* Increase font size for better visibility */
            background-color: #008CBA; /* Change the background color for higher contrast */
            color: white; /* Keep the text color white for contrast */
            border: none; /* Remove any borders */
            border-radius: 5px; /* Rounded corners for a modern look */
            text-decoration: none; /* Remove underline from links */
            display: inline-block; /* Align inline with other elements */
            margin: 5px 0; /* Adjust margins around the button */
            cursor: pointer; /* Change cursor to pointer on hover */
            box-shadow: 0 4px 8px rgba(0,0,0,0.2); /* Add a shadow for depth */
            transition: background-color 0.3s, box-shadow 0.3s; /* Smooth transition for hover effect */
        }

        .home-button:hover {
            background-color: #003f4b; /* Even darker on hover */
            box-shadow: 0 6px 12px rgba(0,0,0,0.3);
        }


        /* Page break style */
        @media print {
            hr {
                page-break-after: always; /* Force page break after hr for print */
            }
        }

        /* Rest of your CSS */
    </style>
</head>
<body>
    <!-- Target for "Home" button navigation -->
    <div id="top-of-page"></div>

    <div class="navigation">
        {{ nav_links | safe }}
    </div>

    {% for ticker_data in tickers_data %}
        <!-- Insert an anchor tag with the ticker ID right before the section content -->
        <a id="{{ ticker_data.ticker }}" class="section-anchor"></a>
        <div class="ticker-section">
            <h2>{{ ticker_data.ticker }}</h2>
            <!-- Insert the Home button right after the ticker header -->
            <a href="#top-of-page" class="home-button">Home</a>
            <!-- The rest of the ticker section content -->
            <div>
                <img src="{{ ticker_data.revenue_net_income_chart_path }}" alt="Revenue and Net Income Chart" class="chart">
                <img src="{{ ticker_data.eps_chart_path }}" alt="EPS Chart" class="chart">
            </div>
            {{ ticker_data.financial_table | safe }}
            <!-- Balance Sheet Section -->
            <div class="balance-sheet-container">
                <div class="balance-sheet-table" align=left>
                    {{ ticker_data.balance_sheet_table_html | safe }}
                </div>
                <div class="balance-sheet-chart">
                    <img src="{{ ticker_data.balance_sheet_chart_path }}" alt="{{ ticker_data.ticker }} Balance Sheet Chart" class="chart" align=right>
                </div>
            </div>
        </div>
        <hr> <!-- Page break will occur here when printed -->
    {% endfor %}
    
</body>
</html>
"""
print("html generator 4 defined template.html")

# Path to your template file
template_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'template.html')
print("html generator 5 defined template path")


# Ensure the template exists before rendering
ensure_template_exists(template_file_path, template_html_content)
print("html generator 6 ensuring templat exists")





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
    print("html generator 8 print dataframe to console")
    print(f"\n{message}:\n")
    print(df)


# Function to create HTML content
def create_html_for_tickers(current_tickers, financial_data, charts_output_dir, html_file='index.html'):
    charts_output_dir = "charts/"
    print("HTML generator 9 creating HTML for tickers")
    # Ensure charts_output_dir ends with a slash
    charts_output_dir = charts_output_dir.rstrip('/') + '/'

    # Sorting the tickers
    sorted_tickers = sorted(current_tickers)

    # Building navigation links
    home_button = '<a href="#top-of-page" class="home-button">Home</a>'
    nav_links = home_button + " | " + " | ".join(
        f'<a href="#{ticker}" class="home-button">{ticker}</a>' for ticker in sorted_tickers)


    # Load the template
    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template('template.html')
    print("---loading HTML template")

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

            # Read the balance sheet table HTML file
            with open(f"charts/{ticker}_balance_sheet_table.html", 'r') as bs_table_file:
                balance_sheet_table_html = bs_table_file.read()

            # Add balance sheet table and chart for each ticker
            balance_sheet_html = f"""
                    <div class="balance-sheet-container">
                        <div class="balance-sheet-table">
                            <!-- Your balance sheet table HTML here -->
                            {rendered_table}  <!-- For example purposes, using the same table as above -->
                        </div>
                        <div class="balance-sheet-chart">
                            <img src="charts/{ticker}_balance_sheet_chart.png" alt="Balance Sheet Chart">
                        </div>
                    </div>
                    """

            ticker_data = {
                'ticker': ticker,
                'revenue_net_income_chart_path': f"{charts_output_dir}{ticker}_revenue_net_income_chart.png",
                'eps_chart_path': f"{charts_output_dir}{ticker}_eps_chart.png",
                'financial_table': rendered_table,
                # Adding paths for balance sheet chart and table HTML
                'balance_sheet_chart_path': f"{charts_output_dir}{ticker}_balance_sheet_chart.png",
                'balance_sheet_table_html': balance_sheet_table_html
                # This is the HTML content of the balance sheet table
            }

            data_for_rendering['tickers_data'].append(ticker_data)

    html_content = template.render(data_for_rendering)

    with open(html_file, 'w') as file:
        file.write(html_content)

    print(f"HTML content has been written to {html_file}")

    return html_content

#end of html_generator.py

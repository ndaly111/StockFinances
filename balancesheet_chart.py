import sqlite3
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# Constants
DB_PATH = 'Stock Data.db'
charts_output_dir = 'charts/'

# Ensure the charts output directory exists
os.makedirs(charts_output_dir, exist_ok=True)

# Fetch data from the database
def fetch_balance_sheet_data(ticker):
    print("balance sheet chart 1 fetch bs data")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT Total_Assets, Total_Liabilities, Total_Shareholder_Equity, Total_Debt, Cash_and_Cash_Equivalents
        FROM BalanceSheetData
        WHERE Symbol = ?
        ORDER BY Date DESC
        LIMIT 1;""", (ticker,))
    data = cursor.fetchone()
    print("---data fetched",data)
    conn.close()
    if data:
        return {
            'Total_Assets': data[0],
            'Total_Liabilities': data[1],
            'Total_Equity': data[2],
            'Total_Debt': data[3],
            'Cash': data[4]
        }
    else:
        return None

# Plot the chart
def plot_chart(data, output_dir, ticker):
    print("Plotting Balance Sheet Chart")
    # Calculate non-cash assets and non-debt liabilities
    non_cash_assets = data['Total_Assets'] - data['Cash']
    non_debt_liabilities = data['Total_Liabilities'] - data['Total_Debt']

    # Set up the figure and axes
    fig, ax = plt.subplots()

    # Define bar locations and width
    bar_locs = np.array([0, 1, 3, 4])
    bar_width = 0.4

    # Plot stacked bars for cash and non-cash assets
    ax.bar(bar_locs[0], data['Cash'], label='Cash', color='#77dd77', width=bar_width)
    ax.bar(bar_locs[0], non_cash_assets, bottom=data['Cash'], label='Non-Cash Assets', color='#3498db', width=bar_width)

    # Plot stacked bars for debt and non-debt liabilities
    ax.bar(bar_locs[1], data['Total_Debt'], label='Debt', color='#800000', width=bar_width)
    ax.bar(bar_locs[1], non_debt_liabilities, bottom=data['Total_Debt'], label='Non-Debt Liabilities', color='#ff7f50', width=bar_width)

    # Plot bars for total equity and total debt
    ax.bar(bar_locs[2], data['Total_Equity'], label='Total Equity', color='#3498db', width=bar_width)
    ax.bar(bar_locs[3], data['Total_Debt'], label='Total Debt', color='#fa8072', width=bar_width)

    ax.axhline(0, color='black', linewidth='2')

    # Add an x-axis line if there is negative equity
    if data['Total_Equity'] < 0:
        ax.axhline(0, color='grey', linewidth=0.8)

    # Add labels, title, and legend
    ax.set_ylabel('Amount ($)')
    ax.set_title(f'Balance Sheet Breakdown for {ticker}')  # Dynamic title
    ax.set_xticks(bar_locs)
    ax.set_xticklabels(['Assets', 'Liabilities', 'Equity', 'Debt'])
    ax.legend()

    # Save the figure
    chart_path = os.path.join(output_dir, f"{ticker}_balance_sheet_chart.png")
    plt.savefig(chart_path)
    print(f"Chart saved to {chart_path}")
    plt.close(fig)

# Create a DataFrame
def format_value(value):
    """Format the financial figures and ratios."""
    if isinstance(value, float):  # Assuming all floats should be formatted to two decimal places
        return f"{value:,.2f}"
    elif isinstance(value, int):
        return f"${value / 1_000_000:,.0f}M"
    return value

# Apply coloring based on Debt_to_Equity_Ratio value

# Apply coloring based on Debt_to_Equity_Ratio value
def create_and_save_table(data, output_dir, ticker):
    print("balance sheet chart 3 creating table")
    print("Data received for table:", data)

    df = pd.DataFrame([data], columns=[
        'Total_Assets',
        'Cash',
        'Total_Liabilities',
        'Total_Debt',
        'Total_Equity',
        'Debt_to_Equity_Ratio'
    ])

    # Function to apply color formatting to Debt to Equity Ratio
    def color_debt_to_equity_ratio(val):
        try:
            val = float(val)
            if val > 1:
                color = 'red'
            if val < 0:
                color = 'red'
            elif val < 1:
                color = 'green'
            else:
                color = 'black'
            return f'color: {color}'
        except ValueError:
            return ''

    # Assuming you have a dictionary named 'data' containing your financial data
    # Calculate Debt to Equity Ratio
    debt_to_equity_ratio = data['Total_Debt'] / data['Total_Equity']

    # Prepare the data for DataFrame
    data_for_df = {
        'Metric': ['Total Assets', 'Cash', 'Total Liabilities', 'Total Debt', 'Total Equity', 'Debt to Equity Ratio'],
        'Value': [
            f"${data['Total_Assets'] / 1_000_000:,.0f}M",
            f"${data['Cash'] / 1_000_000:,.0f}M",
            f"${data['Total_Liabilities'] / 1_000_000:,.0f}M",
            f"${data['Total_Debt'] / 1_000_000:,.0f}M",
            f"${data['Total_Equity'] / 1_000_000:,.0f}M",
            f"{debt_to_equity_ratio:,.2f}"  # Use the calculated debt_to_equity_ratio
        ]
    }

    # Create DataFrame
    df = pd.DataFrame(data_for_df)

    # Apply color formatting to Debt to Equity Ratio column
    styled_df = df.style.map(color_debt_to_equity_ratio, subset=['Value'])

    # Convert DataFrame to HTML
    html_content = styled_df.to_html(index=False)

    # Save the styled DataFrame to an HTML file
    html_file_path = os.path.join(output_dir, f"{ticker}_balance_sheet_table.html")
    with open(html_file_path, 'w') as f:
        f.write(html_content)
    print(f"Table saved to {html_file_path}")

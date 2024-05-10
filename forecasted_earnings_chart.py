import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as mpatches
import numpy as np
import os
from matplotlib.ticker import FuncFormatter, AutoMinorLocator
import yfinance as yf
import shutil  # Import shutil for file operations




def millions_formatter(x, pos):
    print("millions formatter 1 forecasted earnings chart")
    """Formats numbers as millions with a dollar sign."""
    return f'${int(x / 1e6)}M'




def format_axis(ax, max_value):
    print("format axis 2 forecasted earnings chart")
    # Add a buffer to the maximum value to make space for labels
    buffer = max_value * 0.1  # 10% buffer for the y-axis
    max_lim = max_value + buffer

    threshold = 1e9
    if max_value >= threshold:
        formatter = FuncFormatter(lambda x, pos: f'${int(x / 1e9)}B')
        ax.yaxis.set_major_formatter(formatter)
        ax.set_ylabel('USD (Billions)')
    else:
        formatter = FuncFormatter(lambda x, pos: f'${int(x / 1e6)}M')
        ax.yaxis.set_major_formatter(formatter)
        ax.set_ylabel('USD (Millions)')

    ax.set_ylim(top=max_lim)  # Set the top limit with buffer



def fetch_financial_data(ticker, db_path):
    print("Fetching financial data for:", ticker)

    # Database table names
    historical_table = 'Annual_Data'
    forecast_table = 'ForwardFinancialData'

    with sqlite3.connect(db_path) as conn:
        # Fetch historical financial data
        historical_query = f"""
        SELECT Date, Revenue, Net_Income, EPS 
        FROM {historical_table} 
        WHERE Symbol = ? 
        ORDER BY Date;
        """
        historical_data = pd.read_sql_query(historical_query, conn, params=(ticker,))

        # Fetch forecast financial data
        forecast_query = f"""
        SELECT Date, ForwardRevenue AS Revenue, ForwardEPS AS EPS 
        FROM {forecast_table} 
        WHERE Ticker = ? 
        ORDER BY Date;
        """
        forecast_data = pd.read_sql_query(forecast_query, conn, params=(ticker,))

        # Fetch analyst count data
        analyst_count_query = f"""
        SELECT Date, ForwardEPSAnalysts, ForwardRevenueAnalysts 
        FROM {forecast_table} WHERE Ticker = ? ORDER BY Date;
        """
        analyst_counts = pd.read_sql_query(analyst_count_query, conn, params=(ticker,))

        # Fetch shares outstanding data
        shares_outstanding_query = """
        SELECT Shares_Outstanding FROM TTM_Data WHERE Symbol = ? 
        ORDER BY Last_Updated DESC LIMIT 1;
        """
        shares_outstanding_result = pd.read_sql_query(shares_outstanding_query, conn, params=(ticker,))
        print("shares outstanding results",shares_outstanding_result)

        # Determine shares outstanding value
        shares_outstanding = shares_outstanding_result.iloc[0]['Shares_Outstanding'] if not shares_outstanding_result.empty else None

        if shares_outstanding is not None:
            # Convert EPS to numeric and calculate Net_Income for forecast data
            forecast_data['EPS'] = pd.to_numeric(forecast_data['EPS'], errors='coerce')
            forecast_data['Net_Income'] = forecast_data['EPS'] * shares_outstanding
        else:
            print(f"Shares outstanding data is missing for {ticker}, unable to calculate 'Net_Income' for forecast data.")
            forecast_data['Net_Income'] = pd.NA  # Use pandas NA for missing data. This requires pandas version 1.0.0 or later.
        print("shares outstanding",shares_outstanding)
        return historical_data, forecast_data, analyst_counts, shares_outstanding





def prepare_data_for_plotting(historical_data, forecast_data, shares_outstanding, ticker):
    print("prepare data for plotting 4 forecasted earnings chart")
    # Ensure that ticker is a string
    if not isinstance(ticker, str):
        print("Error: Ticker must be a string.")
        return None

    # Fetch current market data for the ticker using yfinance
    market_data = yf.Ticker(ticker)
    print("---market data", market_data)

    # Attempt to fetch various market data points
    current_price = market_data.info.get('regularMarketPrice', None)
    if not current_price:  # Fallback to previous close if regular market price is not available
        current_price = market_data.info.get('previousClose', None)
    
    # If both regular and previous close prices are unavailable, try average of bid and ask
    if not current_price:
        bid = market_data.info.get('bid', None)
        ask = market_data.info.get('ask', None)
        if bid and ask:
            current_price = (bid + ask) / 2
    
    print("---current price", current_price)
    market_cap = market_data.info.get('marketCap', None)
    print("---market cap", market_cap)

    # Assign types to differentiate data in plots
    historical_data['Type'] = 'Historical'
    forecast_data['Type'] = 'Forecast'

    # Convert 'EPS' to numeric to avoid calculation errors
    forecast_data['EPS'] = pd.to_numeric(forecast_data['EPS'], errors='coerce')

    # Calculate Net Income based on new formula
    if current_price and market_cap:
        forecast_data['Net_Income'] = (forecast_data['EPS'] / current_price) * market_cap
    else:
        print("Current price or market cap is missing or invalid. Unable to calculate 'Net_Income' for forecast data.")
        forecast_data['Net_Income'] = pd.NA  # Use pandas NA for missing data

    # Combine historical and forecast data for plotting
    combined_data = pd.concat([historical_data, forecast_data])
    combined_data.sort_values(by=['Date', 'Type'], inplace=True)

    return combined_data


def plot_bars(ax, combined_data, bar_width, analyst_counts):
    print("plot bars 5 forecasted earnings chart")

    # Get the maximum and minimum values for Revenue and Net Income
    max_revenue = combined_data['Revenue'].max()
    min_revenue = combined_data['Revenue'].min()
    max_net_income = combined_data['Net_Income'].max()
    min_net_income = combined_data['Net_Income'].min()

    # Calculate the absolute maximum value for padding calculation
    max_abs_value = max(abs(max_revenue), abs(min_revenue), abs(max_net_income), abs(min_net_income))
    padding = max_abs_value * 0.2  # 20% padding based on the largest absolute value

    # Determine scale based on the data magnitude
    scale = 1e9 if max_abs_value >= 1e9 else 1e6
    unit = 'B' if scale == 1e9 else 'M'

    # Set the limits for the y-axis to include negative values if present
    y_lower_limit = min(min_revenue, min_net_income) - padding  # Subtract padding for the lower limit
    y_upper_limit = max(max_revenue, max_net_income) + padding  # Add padding for the upper limit

    ax.set_ylim(y_lower_limit, y_upper_limit)

    # Unique dates for the x-axis.
    unique_dates = combined_data['Date'].unique()
    # Number of groups of bars.
    n_dates = len(unique_dates)
    # The x position of the groups.
    positions = np.arange(n_dates) * bar_width * 3  # Spacing between groups

    bar_settings = {
        'width': bar_width,
        'align': 'center'
    }

    # Generate custom tick labels with the date and the number of analysts for forecast only
    custom_xtick_labels = []
    for index, date in enumerate(unique_dates):
        date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
        label = date_str

        if date in analyst_counts['Date'].values:
            revenue_analyst_count = analyst_counts.loc[analyst_counts['Date'] == date, 'ForwardRevenueAnalysts'].iloc[0]
            eps_analyst_count = analyst_counts.loc[analyst_counts['Date'] == date, 'ForwardEPSAnalysts'].iloc[0]
            label += f"\n({revenue_analyst_count}) / ({eps_analyst_count})"

        custom_xtick_labels.append(label)

        date_data = combined_data[combined_data['Date'] == date]
        group_offset = positions[index] - bar_width / 2

        if 'Historical' in date_data['Type'].values:
            historical_data = date_data[date_data['Type'] == 'Historical']
            ax.bar(group_offset, historical_data['Revenue'], color='green', **bar_settings)
            ax.bar(group_offset + bar_width, historical_data['Net_Income'], color='blue', **bar_settings)

        if 'Forecast' in date_data['Type'].values:
            forecast_data = date_data[date_data['Type'] == 'Forecast']
            ax.bar(group_offset, forecast_data['Revenue'], color='#b6d7a8', **bar_settings)
            ax.bar(group_offset + bar_width, forecast_data['Net_Income'], color='#a4c2f4', **bar_settings)

    ax.set_xticks(positions)
    ax.set_xticklabels(custom_xtick_labels)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best')

    ax.axhline(0, color='black', linewidth=0.8)
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('USD (Millions)', fontsize=12)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${int(x / 1e6)}M'))
    ax.set_title(f'Revenue and Net Income (Historical & Forecasted)', fontsize=14)

    return ax



def add_value_labels(ax):
    """Adds value labels to the bars."""
    for rect in ax.patches:
        height = rect.get_height()
        x = rect.get_x() + rect.get_width() / 2
        # Offset from the bar, tweak as needed
        offset = 0.05 * max(ax.get_ylim())
        y = height - offset if height < 0 else height + offset
        label_text = f'{height / 1e6:.1f}M' if abs(height) < 1e9 else f'{height / 1e9:.1f}B'
        ax.text(x, y, label_text, ha='center', va='bottom' if height > 0 else 'top', color='black')




def format_chart(ax, combined_data, output_path, ticker):
    print("format chart 8 forecasted earnings chart")
    """Formats the chart with titles, axis labels, and saves the figure."""
    ax.set_title(f'{ticker} Revenue and Net Income (Historical & Forecasted)')
    ax.set_xlabel('Date')
    ax.set_ylabel('USD (Millions or Billions)')

    # Determine the maximum value in the dataset for scaling
    max_value = combined_data[['Revenue', 'Net_Income']].max().max()

    if max_value >= 1e9:  # if the max value is in the billions, scale to billions
        formatter = FuncFormatter(lambda x, p: f'${int(x / 1e9)}B')
        ax.set_ylabel('USD (Billions)')
    else:  # otherwise, scale to millions
        formatter = FuncFormatter(lambda x, p: f'${int(x / 1e6)}M')
        ax.set_ylabel('USD (Millions)')

    ax.yaxis.set_major_formatter(formatter)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    try:
        plt.tight_layout()
    except Exception as e:
        print(f"An error occurred while applying tight layout: {e}")

    fig_path = f"{output_path}{ticker}_Revenue_Net_Income_Forecast.png"
    plt.savefig(fig_path)
    print(f"Figure saved to {fig_path}")
    plt.close()


def plot_eps(ticker, ax, combined_data, analyst_counts, bar_width):
    # Define colors for EPS bars
    historical_eps_color = '#2c3e50'  # Darker color for historical EPS
    forecast_eps_color = '#74a9cf'    # Same color as before for forecast EPS

    # Calculate the y-axis limits based on EPS values with padding
    max_eps = combined_data['EPS'].max()
    min_eps = combined_data['EPS'].min() if combined_data['EPS'].min() < 0 else 0
    padding = max(abs(max_eps), abs(min_eps)) * 0.2  # 20% of the larger of max/min EPS
    ax.set_ylim(min_eps - padding, max_eps + padding)

    # Set the positions for the EPS bars
    unique_dates = combined_data['Date'].unique()
    positions = np.arange(len(unique_dates)) * (bar_width * 3)

    # Loop through each unique date and plot bars based on 'Type'
    for date in unique_dates:
        date_data = combined_data[combined_data['Date'] == date]
        group_offset = positions[list(unique_dates).index(date)] - bar_width / 2

        if 'Historical' in date_data['Type'].values:
            historical_eps = date_data[date_data['Type'] == 'Historical']
            ax.bar(group_offset, historical_eps['EPS'], width=bar_width, color=historical_eps_color, label='Historical EPS', align='center')
        
        if 'Forecast' in date_data['Type'].values:
            forecast_eps = date_data[date_data['Type'] == 'Forecast']
            ax.bar(group_offset + bar_width, forecast_eps['EPS'], width=bar_width, color=forecast_eps_color, label='Forecast EPS', align='center')

    # Add value labels for EPS
    for rect in ax.patches:
        height = rect.get_height()
        x = rect.get_x() + rect.get_width() / 2
        y = height
        label_text = f'{height:.2f}'  # Format label with two decimal places
        ax.text(x, y, label_text, ha='center', va='bottom')

    # Set labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel('Earnings Per Share (EPS)')
    ax.set_title(f"{ticker} EPS (Historical & Forecasted)")
    ax.axhline(y=0, color='black', linewidth=1)

    # Generate custom tick labels with the date and the number of EPS analysts for forecast only
    custom_xtick_labels = []
    for date in unique_dates:
        date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
        label = date_str
        if date in analyst_counts['Date'].values:
            eps_analyst_count = analyst_counts.loc[analyst_counts['Date'] == date, 'ForwardEPSAnalysts'].iloc[0]
            label += f"\n({eps_analyst_count} analysts)"
        custom_xtick_labels.append(label)

    ax.set_xticks(positions)
    ax.set_xticklabels(custom_xtick_labels)

    # Ensure each label is only added once to the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best')

    return ax



def generate_financial_forecast_chart(ticker, combined_data, charts_output_dir,db_path,historical_data,forecast_data,analyst_counts):
    print("generate financial forecast chart 10 forecasted earnings chart")
    print("---prepare for plotting (within generate financial forecast chart function)")

    # Calculate the maximum value for the axis
    max_revenue = combined_data['Revenue'].max()
    max_net_income = combined_data['Net_Income'].max()
    max_value = max(max_revenue, max_net_income)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    bar_width = 0.3
    plot_bars(ax1, combined_data, bar_width, analyst_counts)  # Call without 'positions'

    # Call format_axis with the max value
    format_axis(ax1, max_value)

    # Generate Revenue and Net Income Chart
    fig, ax1 = plt.subplots(figsize=(10, 6))
    bar_width = 0.3
    plot_bars(ax1, combined_data, bar_width, analyst_counts)  # Call without 'positions'
    add_value_labels(ax1)
    format_chart(ax1, combined_data, charts_output_dir, ticker)

    fig, ax2 = plt.subplots(figsize=(10, 6))
    plot_eps(ticker,ax2, combined_data, analyst_counts, bar_width)
    plt.tight_layout()
    plt.savefig(f"{charts_output_dir}{ticker}_EPS_Forecast.png")
    plt.close(fig)


def calculate_yoy_growth(combined_data, analyst_counts):
    print("calculate yoy growth 11 forecasted earnings chart")
    combined_data['Year'] = pd.to_datetime(combined_data['Date']).dt.year
    combined_data.sort_values(by='Date', inplace=True)

    # Calculate YoY Growth for Revenue and Net Income
    combined_data['Revenue_YoY'] = combined_data['Revenue'].pct_change() * 100
    combined_data['Net_Income_YoY'] = combined_data['Net_Income'].pct_change() * 100

    # Get the last record for each year for Revenue and Net Income
    yoy_table = combined_data.groupby('Year').tail(1).set_index('Year')
    yoy_table = yoy_table[['Revenue_YoY', 'Net_Income_YoY']]

    # Joining the analysts count data
    analyst_counts['Year'] = pd.to_datetime(analyst_counts['Date']).dt.year
    analyst_counts_grouped = analyst_counts.groupby('Year').tail(1).set_index('Year')[['ForwardRevenueAnalysts', 'ForwardEPSAnalysts']]
    yoy_table = yoy_table.join(analyst_counts_grouped, how='left')

    # Format the numeric values as strings with one decimal place for percentages
    yoy_table['Revenue_YoY'] = yoy_table['Revenue_YoY'].map(lambda x: f'{x:.1f}%' if not pd.isnull(x) else '')
    yoy_table['Net_Income_YoY'] = yoy_table['Net_Income_YoY'].map(lambda x: f'{x:.1f}%' if not pd.isnull(x) else '')

    # For ForwardRevenueAnalysts
    yoy_table['ForwardRevenueAnalysts'] = yoy_table['ForwardRevenueAnalysts'].fillna(0).astype(int)
    yoy_table['ForwardEPSAnalysts'] = yoy_table['ForwardEPSAnalysts'].fillna(0).astype(int)


    yoy_table = yoy_table[['Revenue_YoY', 'ForwardRevenueAnalysts', 'Net_Income_YoY', 'ForwardEPSAnalysts']]

    # Renaming columns for clarity
    yoy_table.rename(columns={
        'Revenue_YoY': 'Revenue Growth (%)',
        'ForwardRevenueAnalysts': 'Revenue Analysts (#)',
        'Net_Income_YoY': 'EPS Growth (%)',
        'ForwardEPSAnalysts': 'EPS Analysts (#)'
    }, inplace=True)

    # Transpose the table for the desired format
    yoy_table_transposed = yoy_table.T

    return yoy_table_transposed




def save_yoy_growth_to_html(yoy_growth_table, charts_output_dir, ticker):
    print("save yoy growth to html 12 forecasted earnings chart")
    filename = f"{ticker}_yoy_growth_tbl"

    # Convert the DataFrame to an HTML table string directly without transposing again
    html_table = yoy_growth_table.to_html(classes='table table-striped', justify='center')

    # Adding some basic styling to format the table
    html_string = f'''
    <html>
    <head>
    <title>YoY Growth Rates</title>
    <style>
        .table {{
            width: 80%;
            margin-left: auto;
            margin-right: auto;
            border-collapse: collapse;
        }}
        th, td {{
            text-align: center;
            padding: 8px;
            border: 1px solid #ddd;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        body {{
            font-family: Arial, sans-serif;
        }}
    </style>
    </head>
    <body>
        <h2 style="text-align:center;">{ticker} Year-over-Year Growth</h2>
        {html_table}
    </body>
    </html>
    '''

    # Ensure the output path exists
    os.makedirs(charts_output_dir, exist_ok=True)

    # Write the HTML string to a file
    full_path = os.path.join(charts_output_dir, f"{filename}.html")
    with open(full_path, 'w') as f:
        f.write(html_string)

    print(f"YoY Growth Table saved to {full_path}")



# Example usage
#ticker = 'AAPL'
#db_path = 'Stock Data.db'
historical_table_name = 'Annual_Data'
forecast_table_name = 'ForwardFinancialData'
#output_chart_path = 'charts/'


import matplotlib.pyplot as plt
import pandas as pd

def generate_yoy_line_chart(data, title, ylabel, output_path, analyst_counts_df=None, analyst_column=None):
    """
    Generates a line chart showing year-over-year (YoY) percentage changes with dynamic y-axis limits.

    Args:
        data (pd.Series): A pandas Series containing the percentage changes.
        title (str): The title of the chart.
        ylabel (str): The label for the y-axis.
        output_path (str): The path where the chart image will be saved.
        analyst_counts_df (pd.DataFrame, optional): DataFrame with analyst count data.
        analyst_column (str, optional): Column name to use for analyst counts.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    years = data.index
    values = data.values

    # Plot the data as a line graph
    ax.plot(years, values, marker='o', linestyle='-', color='blue')

    # Add labels for each data point
    for i, (year, value) in enumerate(zip(years, values)):
        # Create the display label based on whether the value exceeds the threshold
        if value > 95:
            label = '95%'
            display_value = 95
        elif value < -95:
            label = '-95%'
            display_value = -95
        else:
            label = f'{value:.1f}%'
            display_value = value

        # Adjust offset direction based on the original value (not clamped)
        y_offset = 1 if value > 0 else -1

        # Display the label at the clamped position
        ax.text(year, display_value + y_offset, label, ha='center', va='bottom' if display_value > 0 else 'top', fontsize=10)

    # Set custom x-axis labels to include analyst counts where available
    if analyst_counts_df is not None and analyst_column:
        # Ensure years are integers for indexing
        analyst_counts_df['Year'] = pd.to_datetime(analyst_counts_df['Date']).dt.year
        year_analysts = analyst_counts_df.groupby('Year')[analyst_column].last().to_dict()

        x_labels = [
            f"{year}* ({year_analysts.get(year, 'N/A')})" if year in year_analysts else str(year)
            for year in years
        ]
        ax.set_xticks(years)
        ax.set_xticklabels(x_labels)

    ax.set_title(title)
    ax.set_xlabel('Year')
    ax.set_ylabel(ylabel)
    ax.axhline(y=0, color='black', linewidth=0.8)  # Horizontal line at y=0
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Determine dynamic y-axis limits
    max_y_value = min(max(values) + 5, 100)  # Add a buffer, up to a maximum of 100
    min_y_value = max(min(values) - 5, -100)  # Add a buffer to the bottom, but not below -100

    ax.set_ylim(min_y_value, max_y_value)

    fig.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Chart saved to {output_path}")



def generate_revenue_yoy_change_chart(yoy_table, ticker, output_dir, analyst_counts, analyst_column):
    """
    Generates a line chart for Revenue YoY percentage change.

    Args:
        yoy_table (pd.DataFrame): Year-over-Year growth table containing revenue data.
        ticker (str): The stock ticker symbol.
        output_dir (str): The directory where the chart will be saved.
        analyst_counts (pd.DataFrame): Analyst count data.
        analyst_column (str): The column to use for analyst counts.
    """
    revenue_changes = pd.to_numeric(yoy_table.loc['Revenue Growth (%)'].replace('%', '', regex=True), errors='coerce').dropna()
    output_path = f"{output_dir}{ticker}_revenue_yoy_change.png"
    generate_yoy_line_chart(revenue_changes, f"{ticker} Revenue Year-over-Year Change", "Revenue YoY (%)", output_path, analyst_counts, analyst_column)

def generate_eps_yoy_change_chart(yoy_table, ticker, output_dir, analyst_counts, analyst_column):
    """
    Generates a line chart for EPS YoY percentage change.

    Args:
        yoy_table (pd.DataFrame): Year-over-Year growth table containing EPS data.
        ticker (str): The stock ticker symbol.
        output_dir (str): The directory where the chart will be saved.
        analyst_counts (pd.DataFrame): Analyst count data.
        analyst_column (str): The column to use for analyst counts.
    """
    eps_changes = pd.to_numeric(yoy_table.loc['EPS Growth (%)'].replace('%', '', regex=True), errors='coerce').dropna()
    output_path = f"{output_dir}{ticker}_eps_yoy_change.png"
    generate_yoy_line_chart(eps_changes, f"{ticker} EPS Year-over-Year Change", "EPS YoY (%)", output_path, analyst_counts, analyst_column)




def generate_forecast_charts_and_tables(ticker, db_path, charts_output_dir):
    print(f"Generating forecast charts and tables for {ticker}...")

    # Fetch financial data
    historical_data, forecast_data, analyst_counts, shares_outstanding = fetch_financial_data(ticker, db_path)

    # Prepare combined data for plotting
    combined_data = prepare_data_for_plotting(historical_data, forecast_data, shares_outstanding, ticker)

    # Paths for placeholder and target chart filenames
    placeholder_image_path = os.path.join(charts_output_dir, 'No_forecast_data.png')
    revenue_forecast_path = os.path.join(charts_output_dir, f"{ticker}_Revenue_Net_Income_Forecast.png")
    eps_forecast_path = os.path.join(charts_output_dir, f"{ticker}_EPS_Forecast.png")
    revenue_yoy_path = os.path.join(charts_output_dir, f"{ticker}_revenue_yoy_change.png")
    eps_yoy_path = os.path.join(charts_output_dir, f"{ticker}_eps_yoy_change.png")

    # Check if there is forecast data available
    if forecast_data.empty:
        print(f"No forecast data available for {ticker}.")
        # Copy placeholders for the forecast charts
        shutil.copy(placeholder_image_path, revenue_forecast_path)
        shutil.copy(placeholder_image_path, eps_forecast_path)

        # Copy placeholders for the YoY growth charts
        shutil.copy(placeholder_image_path, revenue_yoy_path)
        shutil.copy(placeholder_image_path, eps_yoy_path)
    else:
        # Generate forecast charts if data is available
        generate_financial_forecast_chart(
            ticker, combined_data, charts_output_dir, db_path, historical_data,
            forecast_data, analyst_counts
        )

        # Proceed with YoY growth calculation and charts generation
        yoy_growth_table = calculate_yoy_growth(combined_data, analyst_counts)
        save_yoy_growth_to_html(yoy_growth_table, charts_output_dir, ticker)

        # Revenue YoY Change Chart
        generate_revenue_yoy_change_chart(yoy_growth_table, ticker, charts_output_dir, analyst_counts, 'ForwardRevenueAnalysts')

        # EPS YoY Change Chart
        generate_eps_yoy_change_chart(yoy_growth_table, ticker, charts_output_dir, analyst_counts, 'ForwardEPSAnalysts')

    print(f"Completed generating charts and tables for {ticker}.")


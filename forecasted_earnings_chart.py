import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as mpatches
import numpy as np
import os
from matplotlib.ticker import FuncFormatter, AutoMinorLocator




def millions_formatter(x, pos):
    print("millions formatter 1 forecasted earnings chart")
    """Formats numbers as millions with a dollar sign."""
    return f'${int(x / 1e6)}M'




def format_axis(ax, max_value):
    print("format axis 2 forecasted earnings chart")
    """Dynamically format the axis based on the maximum value."""
    threshold = 1e9
    if max_value >= threshold:
        formatter = FuncFormatter(lambda x, pos: f'${int(x / 1e9)}B')
        ax.yaxis.set_major_formatter(formatter)
        ax.set_ylabel('USD (Billions)')
    else:
        formatter = FuncFormatter(lambda x, pos: f'${int(x / 1e6)}M')
        ax.yaxis.set_major_formatter(formatter)
        ax.set_ylabel('USD (Millions)')


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



def prepare_data_for_plotting(historical_data, forecast_data, shares_outstanding):
    print("prepare data for plotting 4 forecasted earnings chart")
    # Assign types to differentiate data in plots
    historical_data['Type'] = 'Historical'
    forecast_data['Type'] = 'Forecast'

    # Convert 'EPS' to numeric to avoid calculation errors
    forecast_data['EPS'] = pd.to_numeric(forecast_data['EPS'], errors='coerce')

    # Ensure 'shares_outstanding' is not None and is a numeric value
    if shares_outstanding and pd.notnull(shares_outstanding):
        forecast_data['Net_Income'] = forecast_data['EPS'] * shares_outstanding
    else:
        print("Shares outstanding data is missing or invalid. Unable to calculate Net Income for forecast data.")
        forecast_data['Net_Income'] = pd.NA  # Use pandas NA for missing data

    # Combine historical and forecast data for plotting
    combined_data = pd.concat([historical_data, forecast_data])
    combined_data.sort_values(by=['Date', 'Type'], inplace=True)

    # Optionally, you can drop or fill NA values in 'Net_Income' here if needed

    return combined_data



def plot_bars(ax, combined_data, bar_width, analyst_counts):
    print("plot bars 5 forecasted earnings chart")
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
        # Initialize default label
        label = date_str

        # If this date has forecast data, add the analyst counts
        if date in analyst_counts['Date'].values:
            revenue_analyst_count = analyst_counts.loc[analyst_counts['Date'] == date, 'ForwardRevenueAnalysts'].iloc[0]
            eps_analyst_count = analyst_counts.loc[analyst_counts['Date'] == date, 'ForwardEPSAnalysts'].iloc[0]
            label += f"\n({revenue_analyst_count}) / ({eps_analyst_count})"

        custom_xtick_labels.append(label)

        # Plotting logic remains the same
        date_data = combined_data[combined_data['Date'] == date]
        group_offset = positions[index] - bar_width / 2
        # Plot historical bars
        if 'Historical' in date_data['Type'].values:
            historical_data = date_data[date_data['Type'] == 'Historical']
            ax.bar(group_offset, historical_data['Revenue'], color='green', **bar_settings)
            ax.bar(group_offset + bar_width, historical_data['Net_Income'], color='blue', **bar_settings)
        # Plot forecast bars
        if 'Forecast' in date_data['Type'].values:
            forecast_data = date_data[date_data['Type'] == 'Forecast']
            ax.bar(group_offset, forecast_data['Revenue'], color='#b6d7a8', **bar_settings)
            ax.bar(group_offset + bar_width, forecast_data['Net_Income'], color='#a4c2f4', **bar_settings)

    ax.set_xticks(positions)
    ax.set_xticklabels(custom_xtick_labels)

    # Adjust the legend to show only one entry per label
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best')

    # Add a horizontal line at y=0
    ax.axhline(0, color='black', linewidth=0.8)

    # Add minor tick marks on the y-axis
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    # Improve formatting
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
    ax.set_ylabel('USD (Millions)')
    formatter = FuncFormatter(lambda x, p: f'${int(x / 1e6)}M')
    ax.yaxis.set_major_formatter(formatter)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    try:
        plt.tight_layout()
    except Exception as e:
        print(f"An error occurred while applying tight layout: {e}")
        # Possibly more diagnostics here

    fig_path = f"{output_path}{ticker}_Revenue_Net_Income_Forecast.png"
    plt.savefig(fig_path)
    print(f"Figure saved to {fig_path}")
    plt.close()


def plot_eps(ticker,ax, combined_data, analyst_counts):
    # Define color for EPS bars
    eps_color = '#74a9cf'

    # Bar settings
    bar_settings = {
        'width': 0.35,
        'align': 'center',
        'color': eps_color,
        'label': 'EPS',
        'alpha': 0.7
    }

    # Plot EPS bars
    ax.bar(combined_data['Date'], combined_data['EPS'], **bar_settings)

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
    unique_dates = combined_data['Date'].unique()
    custom_xtick_labels = []
    for date in unique_dates:
        date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
        label = date_str

        # If this date has forecast data, add the EPS analyst counts
        if date in analyst_counts['Date'].values:
            eps_analyst_count = analyst_counts.loc[analyst_counts['Date'] == date, 'ForwardEPSAnalysts'].iloc[0]
            label += f"\n({eps_analyst_count} analysts)"
        custom_xtick_labels.append(label)

    ax.set_xticks(combined_data['Date'])
    ax.set_xticklabels(custom_xtick_labels)

    return ax



def generate_financial_forecast_chart(ticker, combined_data, charts_output_dir,db_path,historical_data,forecast_data,analyst_counts):
    print("generate financial forecast chart 10 forecasted earnings chart")
    print("---prepare for plotting (within generate financial forecast chart function)")

    # Generate Revenue and Net Income Chart
    fig, ax1 = plt.subplots(figsize=(10, 6))
    bar_width = 0.3
    plot_bars(ax1, combined_data, bar_width, analyst_counts)  # Call without 'positions'
    add_value_labels(ax1)
    format_chart(ax1, combined_data, charts_output_dir, ticker)

    fig, ax2 = plt.subplots(figsize=(10, 6))
    plot_eps(ticker,ax2, combined_data, analyst_counts)
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
        'Net_Income_YoY': 'Net Income Growth (%)',
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


def generate_forecast_charts_and_tables(ticker, db_path, charts_output_dir):
    print("generate forecast charts and tables 13 forecasted earnings chart")
    historical_table_name = 'Annual_Data'
    forecast_table_name = 'ForwardFinancialData'

    # Fetch the necessary data only once
    historical_data, forecast_data, analyst_counts, shares_outstanding = fetch_financial_data(ticker, db_path)

    # Prepare combined data for plotting only once
    combined_data = prepare_data_for_plotting(historical_data, forecast_data, shares_outstanding)

    # Pass the prepared combined_data to the chart generation function
    generate_financial_forecast_chart(ticker, combined_data, charts_output_dir,db_path,historical_data,forecast_data,analyst_counts)

    # Proceed with YOY growth calculation and HTML generation using combined_data
    yoy_growth_table = calculate_yoy_growth(combined_data, analyst_counts)
    save_yoy_growth_to_html(yoy_growth_table, charts_output_dir, ticker)

    print(f"Completed generating charts and tables for {ticker}.")


#This is chart_generator

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os
from datetime import datetime

from split_utils import apply_split_adjustments, ensure_splits_table




def chart_needs_update(chart_path, last_data_update, ttm_update=False, annual_update=False, debug_this=False):
    print("chart generator 3 does chart need an update?")

    # If debug mode is on, always update
    if debug_this:
        print("---debug mode true")
        return True

    # If TTM or Annual data has been updated, we need to update the chart
    if ttm_update or annual_update:
        print("---data update true")
        return True

    # If the chart file doesn't exist, we need to create and hence update it
    if not os.path.exists(chart_path):
        print("---chart does not exist true")
        return True

    # If the chart exists, compare the last modified time of the chart with the last data update
    last_chart_update = datetime.fromtimestamp(os.path.getmtime(chart_path))
    if last_data_update > last_chart_update:
        print("---last update was more recent than the chart update true")
        # If the data is more recent than the chart, an update is needed
        return True

    # If none of the above conditions are met, no update is needed
    print("---no chart generation conditions met")
    return False


def prepare_data_for_charts(ticker, cursor):
    print("chart generator 4 preparing data for charts")
    ensure_splits_table(cursor)
    if apply_split_adjustments(ticker, cursor):
        print(f"[{ticker}] Split adjustments applied prior to chart generation.")

    # Fetch annual data including Last_Updated
    cursor.execute("SELECT Date, Revenue, Net_Income, EPS, Last_Updated FROM Annual_Data WHERE Symbol = ? ORDER BY Date", (ticker,))
    annual_data = cursor.fetchall()
    print("---fetching all annual data", annual_data)

    # Fetch TTM data including Last_Updated
    cursor.execute("SELECT 'TTM' AS Date, TTM_Revenue AS Revenue, TTM_Net_Income AS Net_Income, TTM_EPS AS EPS, Last_Updated FROM TTM_Data WHERE Symbol = ?", (ticker,))
    ttm_data = cursor.fetchall()
    print("---fetching all ttm data from database", ttm_data)

    # Fetch TTM data including Last_Updated
    cursor.execute(
        "SELECT 'TTM' AS Date, TTM_Revenue AS Revenue, TTM_Net_Income AS Net_Income, TTM_EPS AS EPS, Quarter AS Quarter, Last_Updated FROM TTM_Data WHERE Symbol = ?",
        (ticker,))
    ttm_datab = cursor.fetchall()
    print("---fetching all ttm data from database", ttm_data)


    # Convert to DataFrame and ensure correct data types
    annual_df = pd.DataFrame(annual_data, columns=['Date', 'Revenue', 'Net_Income', 'EPS', 'Last_Updated'])
    ttm_df = pd.DataFrame(ttm_data, columns=['Date', 'Revenue', 'Net_Income', 'EPS', 'Last_Updated'])
    ttm_dfb = pd.DataFrame(ttm_datab, columns=['Date', 'Revenue', 'Net_Income', 'EPS', 'Quarter','Last_Updated'])
    print("---converting df to correct names", annual_df,ttm_df)

    # Ensure all 'Last_Updated' entries are Timestamps
    annual_df['Last_Updated'] = pd.to_datetime(annual_df['Last_Updated'])
    ttm_df['Last_Updated'] = pd.to_datetime(ttm_df['Last_Updated'])
    print("---converting all last updated entries are timestamps")

    # Assuming 'Last_Updated' is in the format 'YYYY-MM-DD HH:MM:SS'
    if not ttm_df.empty and 'Last_Updated' in ttm_df:
        print("---checking if ttm df is not empty")

        # Extract the last quarter end date
        last_quarter_end = ttm_dfb.loc[0, 'Quarter']
        print("---last quarter",last_quarter_end)
        # Check if 'Date' already formatted with 'TTM' to prevent "TTM TTM"
        if not str(last_quarter_end).startswith('TTM'):
            ttm_df.at[0, 'Date'] = f'TTM {last_quarter_end}'
        print('---extracting last quarter date', last_quarter_end)

    # Check if annual_data and ttm_data are empty
    if not annual_data and not ttm_data:
        print("---checking if annual and ttm data is empty",pd)

        return pd.DataFrame()


    # Combine annual data and TTM data
    combined_df = pd.concat([annual_df, ttm_df], ignore_index=True)
    print("---combining annual df and ttm df")

    # Handle nulls and ensure correct data types
    combined_df['Revenue'] = pd.to_numeric(combined_df['Revenue'], errors='coerce')
    combined_df['Net_Income'] = pd.to_numeric(combined_df['Net_Income'], errors='coerce')
    combined_df['EPS'] = pd.to_numeric(combined_df['EPS'], errors='coerce')
    print("---handling null data")

    # Sort the DataFrame to ensure 'TTM' appears last
    combined_df['Date'] = combined_df['Date'].astype(str)
    combined_df.sort_values(by='Date', inplace=True)
    print("sorting df with TTM last")

    print("---combined df: ", combined_df)


    return combined_df



# Usage in the main chart generation function
def generate_revenue_net_income_chart(financial_data_df, ticker, revenue_chart_path):
    print("chart generator 5 generating rev and net inc chart")
    df = financial_data_df
    print("---revenue net income chart data frame", df)

    df['Date'] = df['Date'].astype(str)
    positions = np.arange(len(df))
    width = 0.3
    fig, ax = plt.subplots(figsize=(10, 6))

    max_net_income = df['Net_Income'].max()
    if abs(max_net_income) >= 1e9:
        scale_factor = 1e9
        label_ending = 'B'
        ylabel = 'Amount (Billions $)'
    else:
        scale_factor = 1e6
        label_ending = 'M'
        ylabel = 'Amount (Millions $)'

    bars1 = ax.bar(positions - width / 2, df['Revenue'] / scale_factor, width, label=f'Revenue ({label_ending})', color='green')
    bars2 = ax.bar(positions + width / 2, df['Net_Income'] / scale_factor, width, label=f'Net Income ({label_ending})', color='blue')
    ax.set_ylabel(ylabel)

    max_revenue = max(df['Revenue'] / scale_factor)
    min_net_income = min(df['Net_Income'] / scale_factor)
    upper_limit = max_revenue * 1.2
    lower_limit = min_net_income * 1.2 if min_net_income < 0 else 0

    ax.set_ylim(lower_limit, upper_limit)
    ax.set_title(f'Revenue and Net Income for {ticker}')
    ax.set_xticks(positions)
    ax.set_xticklabels(df['Date'], rotation=0)
    ax.legend()

    # Adding grid lines for each tick and a thicker line at y=0
    ax.grid(True, which='both', linestyle='--', linewidth='0.5', color='grey', axis='y')
    ax.axhline(0, color='black', linewidth=1)  # Thicker line at y=0

    def add_value_labels(ax, bars):
        for bar in bars:
            height = bar.get_height()
            label = f'{height:.1f}{label_ending}'
            y_offset = 3 if height >= 0 else -12
            ax.annotate(label, xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, y_offset),
                        textcoords="offset points", ha='center', va='bottom')

    add_value_labels(ax, bars1)
    add_value_labels(ax, bars2)
    plt.tight_layout()
    plt.savefig(revenue_chart_path)
    plt.close()





def generate_eps_chart(ticker, charts_output_dir, financial_data_df):
    print("chart generator 6 generating eps chart")
    if financial_data_df.empty:
        return

    eps_chart_path = f'{charts_output_dir}/{ticker}_eps_chart.png'


    positions = np.arange(len(financial_data_df))
    width = 0.4
    fig, ax = plt.subplots(figsize=(8, 5))
    eps_bars = ax.bar(positions, financial_data_df['EPS'], width, label='EPS', color='teal')

    ax.grid(True, linestyle='-', linewidth='0.5', color='grey')
    ax.axhline(0, color='black', linewidth='2')
    ax.set_ylabel('Earnings Per Share (EPS)')
    ax.set_title(f'EPS Chart for {ticker}')
    ax.set_xticks(positions)
    ax.set_xticklabels(financial_data_df['Date'], rotation=0)


    def add_eps_value_labels(ax, bars):
        for bar in bars:
            height = bar.get_height()
            label = f'${height:.2f}'

            label_y_offset = 12

            if height < 0:  # For negative values
                label_y_offset = -12

            ax.annotate(label,
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, label_y_offset),
                        textcoords="offset points",
                        ha='center', va='bottom', color="black")

    add_eps_value_labels(ax, eps_bars)

    # Calculate y-axis limits
    max_eps = max(financial_data_df['EPS'])
    min_eps = min(financial_data_df['EPS'])

    if max_eps <0:
        upper_limit = 0
    else:
        upper_limit = max_eps * 1.25
    if min_eps < 0:
        adjusted_eps_limit = min_eps * 1.25
        print("adjusted eps limit",adjusted_eps_limit)
        adjusted_max_eps_limit = 0 - (max_eps * 1.25)
        print("adjusted max eps limit",adjusted_max_eps_limit)
        print
        lower_limit = min(adjusted_eps_limit, adjusted_max_eps_limit)
    else:
        lower_limit = 0

    # Set the new y-axis limits
    ax.set_ylim(lower_limit, upper_limit)

    plt.tight_layout()
    plt.savefig(eps_chart_path)
    plt.close(fig)



def append_yearly_changes(df):
    """
    Appends yearly percentage changes to financial columns.
    """
    df.sort_values('Date', ascending=True, inplace=True)
    financial_columns = ['Revenue', 'Net_Income', 'EPS']
    for column in financial_columns:
        change_column = f"{column}_Change"
        df[change_column] = df[column].pct_change() * 100
        df[change_column] = df[change_column].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A")

    return df


def prepare_financial_data(df):
    """
    Prepare the financial data by converting strings to numeric and handling nulls.
    """
    financial_columns = ['Revenue', 'Net_Income', 'EPS']
    for column in financial_columns:
        # Convert to float, assuming the columns may have commas or dollar signs
        df[column] = pd.to_numeric(df[column].replace('[\$,]', '', regex=True), errors='coerce')

    return df


def format_currency(value, millions=True):
    """Format the value as currency with thousands separators and M (millions) or B (billions)."""
    if pd.isna(value):
        return "N/A"
    if millions:
        value /= 1e6
        suffix = 'M'
    else:
        value /= 1e9
        suffix = 'B'
    return f"${value:,.2f}{suffix}"

def format_percentage_change(value):
    """Format the percentage change."""
    if pd.isna(value):
        return "N/A"
    return f"{value:.2f}%"

def format_eps(value):
    """Format the EPS value."""
    if pd.isna(value):
        return "N/A"
    return f"${value:.2f}"

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


def append_yearly_changes(df):
    """
    Appends yearly percentage changes to financial columns.
    Ensure all data are numeric and handle NaN values appropriately.
    """
    financial_columns = ['Revenue', 'Net_Income', 'EPS']
    for column in financial_columns:
        # Convert column to numeric, forcing errors to NaN to avoid type issues
        df[column] = pd.to_numeric(df[column], errors='coerce')

        # Calculate the percentage change, specifying no fill method for NAs
        change_column = f"{column}_Change"
        df[change_column] = df[column].pct_change(fill_method=None) * 100  # Explicitly handle NA values

        # Format the change as a string with two decimal places, handling NaNs
        df[change_column] = df[change_column].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A")

    return df


def generate_financial_data_table_html(ticker, df, charts_output_dir):
    df = calculate_and_format_changes(df)
    html_table = df.to_html(classes="financial-data", border=0, na_rep='N/A')

    # Save the HTML table to a file
    table_file_path = os.path.join(charts_output_dir, f"{ticker}_rev_net_table.html")
    with open(table_file_path, 'w', encoding='utf-8') as f:
        f.write(html_table)
    print(f"Financial data table for {ticker} saved to {table_file_path}")



def generate_financial_charts(ticker, charts_output_dir, financial_data):
    print("chart generator 7 generating financial charts")

    if financial_data.empty:
        print("---chart data is empty so return")
        return

    revenue_chart_path = os.path.join(charts_output_dir, f"{ticker}_revenue_net_income_chart.png")
    eps_chart_path = os.path.join(charts_output_dir, f"{ticker}_eps_chart.png")
    print("--- define chart path")

    if 'Last_Updated' in financial_data.columns:
        financial_data['Last_Updated'] = pd.to_datetime(financial_data['Last_Updated'], errors='coerce')
        last_data_update = financial_data['Last_Updated'].max()
    else:
        print("Warning: 'Last_Updated' column not found. Proceeding without it.")
        last_data_update = datetime.now()  # Use current time as a fallback

    if chart_needs_update(revenue_chart_path, last_data_update):
        generate_revenue_net_income_chart(financial_data, ticker, revenue_chart_path)
        print("--- needed a rev/net income chart update, generated")

    if chart_needs_update(eps_chart_path, last_data_update):
        generate_eps_chart(ticker, charts_output_dir, financial_data)
        print("---needed a new eps chart, generated")

    # Generate financial data table in HTML format
    generate_financial_data_table_html(ticker, financial_data, charts_output_dir)

#end of chart_generator.py

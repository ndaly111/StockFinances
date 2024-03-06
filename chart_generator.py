#This is chart_generator

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os
from datetime import datetime



def format_axis_as_billions(ax):
    print("chart generator 1 format axis as billions")
    ax.get_yaxis().set_major_formatter(
        mtick.FuncFormatter(lambda x, p: format(int(x), ',.0f')))

def format_axis_as_millions(ax):
    print("chart generator 2 format axis as millions")
    ax.get_yaxis().set_major_formatter(
        mtick.FuncFormatter(lambda x, p: format(int(x) / 1e6, ',.0f') + 'M'))


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
    """
    Generates the revenue and net income chart for a given ticker.
    :param df: DataFrame containing financial data.
    :param ticker: The ticker symbol.
    :param cursor: Database cursor object.
    """

    df = financial_data_df
    print("---revenue net income chart data frame", df)


    # Convert 'Year' to a string if it's not already
    df['Date'] = df['Date'].astype(str)


    # Set the positions and width for the bars
    positions = np.arange(len(df))
    width = 0.3

    fig, ax = plt.subplots(figsize=(10, 6))

    # Determine if Net Income should be in millions or billions
    net_income_max = df['Net_Income'].max()
    print("---net income max", net_income_max)
    if abs(net_income_max) < 1e9: # Default scale for billions
        print("---net income is more than 1B, format in millions")
        scale_factor = 1e6  # This is the new variable for scale factor
        label_ending = 'M'

        bars1 = ax.bar(positions - width / 2, df['Revenue'] / 1e6, width, label='Revenue (Millions)', color='green')
        bars2 = ax.bar(positions + width / 2, df['Net_Income'] / 1e6, width, label='Net Income (Millions)',
                       color='blue')
        ax.set_ylabel('Amount (Millions $)')
        format_axis_as_millions(ax)  # Format axis as millions


    else:  # Otherwise, use billions
        print("---else (meaning max net income is more than $1B, formatting in millions")
        bars1 = ax.bar(positions - width / 2, df['Revenue'] / 1e9, width, label='Revenue (Billions)', color='green')
        bars2 = ax.bar(positions + width / 2, df['Net_Income'] / 1e9, width, label='Net Income (Billions)',
                       color='blue')
        ax.set_ylabel('Amount (Billions $)')
        format_axis_as_billions(ax)  # Format axis as billions
        scale_factor = 1e9  # This is the new variable for scale factor
        label_ending = 'B'



    # Adding grid lines for each tick and a thicker line at y=0
    ax.grid(True, which='both', linestyle='--', linewidth='0.5', color='grey', axis='y')
    ax.axhline(0, color='black', linewidth='1')  # Thicker line at y=0

    # Adding labels, title, and legend

    ax.set_title(f'Revenue and Net Income for {ticker}')
    ax.set_xticks(positions)
    ax.set_xticklabels(df['Date'], rotation=0)
    ax.legend(loc='upper left', bbox_to_anchor=(0, -.05))

    # Adding value labels on top of the bars
    def add_value_labels(ax, bars, label_ending, scale_factor):
        for bar in bars:
            height = bar.get_height()
            label = f'{height:.1f}{label_ending}'  # Apply scale_factor here

            # Determine the y offset for the label
            label_y_offset = 12 if height >= 0 else -12

            ax.annotate(label,
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, label_y_offset),
                        textcoords="offset points",
                        ha='center', va='bottom', color="black")

    add_value_labels(ax, bars1, label_ending, scale_factor)
    add_value_labels(ax, bars2, label_ending, scale_factor)

    plt.tight_layout()

    # Find the tallest bar for Revenue and lowest for Net Income
    max_revenue = max(df['Revenue'] / scale_factor)
    min_net_income = min(df['Net_Income'] / scale_factor)

    # Calculate new limits
    upper_limit = max_revenue * 1.2
    if min_net_income < 0:
        adjusted_net_income_limit = min_net_income * 1.3
        adjusted_revenue_limit = 0 - (max_revenue * 0.25)
        lower_limit = min(adjusted_net_income_limit, adjusted_revenue_limit)
    else:
        lower_limit = 0


    # Set the new y-axis limits
    ax.set_ylim(lower_limit, upper_limit)

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




#end of chart_generator.py
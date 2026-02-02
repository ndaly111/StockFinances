import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, AutoMinorLocator
import numpy as np
import os
import yfinance as yf
import shutil
from functools import lru_cache

# Cache for yfinance data to avoid repeated API calls
_yf_cache = {}

def prefetch_yfinance_data(tickers: list) -> dict:
    """Batch fetch yfinance data for multiple tickers at once.

    This is much faster than calling yf.Ticker().info individually.
    Call this once before processing tickers to warm the cache.
    """
    global _yf_cache
    # Filter out tickers we already have cached
    missing = [t for t in tickers if t not in _yf_cache]
    if not missing:
        return _yf_cache

    try:
        # Batch fetch using yf.Tickers (much faster than individual calls)
        batch = yf.Tickers(" ".join(missing))
        for ticker in missing:
            try:
                info = batch.tickers[ticker].info
                _yf_cache[ticker] = info if info else {}
            except Exception:
                _yf_cache[ticker] = {}
    except Exception as e:
        print(f"[yfinance batch] Error fetching batch data: {e}")
        # Fallback: mark as empty so we don't retry
        for ticker in missing:
            _yf_cache[ticker] = {}

    return _yf_cache

def get_cached_yf_info(ticker: str) -> dict:
    """Get yfinance info from cache, or fetch if not cached."""
    if ticker in _yf_cache:
        return _yf_cache[ticker]

    # Single ticker fetch as fallback
    try:
        info = yf.Ticker(ticker).info
        _yf_cache[ticker] = info if info else {}
    except Exception:
        _yf_cache[ticker] = {}

    return _yf_cache[ticker]

def millions_formatter(x, pos):
    return f'${int(x / 1e6)}M'

def format_axis(ax, max_value):
    buffer = max_value * 0.1
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

    ax.set_ylim(top=max_lim)

def fetch_financial_data(ticker, db_path):
    """Fetch all financial data in optimized queries."""
    historical_table = 'Annual_Data'
    forecast_table = 'ForwardFinancialData'

    with sqlite3.connect(db_path) as conn:
        # Combined query for forecast data + analyst counts (avoid 2 separate queries)
        forecast_query = f"""
        SELECT Date, ForwardRevenue AS Revenue, ForwardEPS AS EPS,
               ForwardEPSAnalysts, ForwardRevenueAnalysts
        FROM {forecast_table}
        WHERE Ticker = ?
        ORDER BY Date;
        """
        forecast_full = pd.read_sql_query(forecast_query, conn, params=(ticker,))

        # Split into forecast_data and analyst_counts
        forecast_data = forecast_full[['Date', 'Revenue', 'EPS']].copy()
        analyst_counts = forecast_full[['Date', 'ForwardEPSAnalysts', 'ForwardRevenueAnalysts']].copy()

        # Historical data query
        historical_query = f"""
        SELECT Date, Revenue, Net_Income, EPS
        FROM {historical_table}
        WHERE Symbol = ?
        ORDER BY Date;
        """
        historical_data = pd.read_sql_query(historical_query, conn, params=(ticker,))

        # Shares outstanding query
        shares_outstanding_query = """
        SELECT Shares_Outstanding FROM TTM_Data WHERE Symbol = ?
        ORDER BY Last_Updated DESC LIMIT 1;
        """
        shares_outstanding_result = pd.read_sql_query(shares_outstanding_query, conn, params=(ticker,))

        shares_outstanding = shares_outstanding_result.iloc[0]['Shares_Outstanding'] if not shares_outstanding_result.empty else None

        if shares_outstanding is not None:
            forecast_data['EPS'] = pd.to_numeric(forecast_data['EPS'], errors='coerce')
            forecast_data['Net_Income'] = forecast_data['EPS'] * shares_outstanding
        else:
            forecast_data['Net_Income'] = pd.NA
        return historical_data, forecast_data, analyst_counts, shares_outstanding

def prepare_data_for_plotting(historical_data, forecast_data, shares_outstanding, ticker):
    """
    Updated: Uses cached yfinance data for better performance.
    """
    # Use cached yfinance data (much faster than individual API calls)
    info = get_cached_yf_info(ticker)

    # Now fetch current_price and market_cap from the safe 'info' dict
    current_price = info.get('regularMarketPrice', None)
    if not current_price:
        current_price = info.get('previousClose', None)
    if not current_price:
        bid = info.get('bid', None)
        ask = info.get('ask', None)
        if bid and ask:
            current_price = (bid + ask) / 2
        else:
            current_price = None

    market_cap = info.get('marketCap', None)

    # Mark data as historical or forecast
    historical_data['Type'] = 'Historical'
    forecast_data['Type'] = 'Forecast'

    forecast_data['EPS'] = pd.to_numeric(forecast_data['EPS'], errors='coerce')

    # If we have a current_price and market_cap, recalc Net_Income in forecast
    if current_price and market_cap:
        forecast_data['Net_Income'] = (forecast_data['EPS'] / current_price) * market_cap
    else:
        forecast_data['Net_Income'] = pd.NA

    combined_data = pd.concat([historical_data, forecast_data])
    combined_data.sort_values(by=['Date', 'Type'], inplace=True)

    combined_data['Revenue'] = pd.to_numeric(combined_data['Revenue'], errors='coerce')
    combined_data['Net_Income'] = pd.to_numeric(combined_data['Net_Income'], errors='coerce')
    combined_data['EPS'] = pd.to_numeric(combined_data['EPS'], errors='coerce')

    combined_data.fillna(0, inplace=True)

    return combined_data

def plot_bars(ax, combined_data, bar_width, analyst_counts):
    max_revenue = combined_data['Revenue'].max()
    min_revenue = combined_data['Revenue'].min()
    max_net_income = combined_data['Net_Income'].max()
    min_net_income = combined_data['Net_Income'].min()

    max_abs_value = max(abs(max_revenue), abs(min_revenue), abs(max_net_income), abs(min_net_income))
    padding = max_abs_value * 0.2

    scale = 1e9 if max_abs_value >= 1e9 else 1e6
    unit = 'B' if scale == 1e9 else 'M'

    y_lower_limit = min(min_revenue, min_net_income) - padding
    y_upper_limit = max(max_revenue, max_net_income) + padding

    ax.set_ylim(y_lower_limit, y_upper_limit)

    unique_dates = combined_data['Date'].unique()
    n_dates = len(unique_dates)
    positions = np.arange(n_dates) * bar_width * 3

    bar_settings = {
        'width': bar_width,
        'align': 'center'
    }

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
    for rect in ax.patches:
        height = rect.get_height()
        x = rect.get_x() + rect.get_width() / 2
        offset = 0.05 * max(ax.get_ylim())
        y = height - offset if height < 0 else height + offset
        label_text = f'{height / 1e6:.1f}M' if abs(height) < 1e9 else f'{height / 1e9:.1f}B'
        va = 'top' if height < 0 else 'bottom'
        ax.text(x, y, label_text, ha='center', va=va, color='black')

def format_chart(ax, combined_data, output_path, ticker):
    ax.set_title(f'{ticker} Revenue and Net Income (Historical & Forecasted)')
    ax.set_xlabel('Date')
    ax.set_ylabel('USD (Millions or Billions)')

    max_value = combined_data[['Revenue', 'Net_Income']].max().max()

    if max_value >= 1e9:
        formatter = FuncFormatter(lambda x, p: f'${int(x / 1e9)}B')
        ax.set_ylabel('USD (Billions)')
    else:
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
    historical_eps_color = '#2c3e50'
    forecast_eps_color = '#74a9cf'

    max_eps = combined_data['EPS'].max()
    min_eps = combined_data['EPS'].min() if combined_data['EPS'].min() < 0 else 0
    padding = max(abs(max_eps), abs(min_eps)) * 0.2
    ax.set_ylim(min_eps - padding, max_eps + padding)

    unique_dates = combined_data['Date'].unique()
    positions = np.arange(len(unique_dates)) * (bar_width * 3)

    for date in unique_dates:
        date_data = combined_data[combined_data['Date'] == date]
        group_offset = positions[list(unique_dates).index(date)] - bar_width / 2

        if 'Historical' in date_data['Type'].values:
            historical_eps = date_data[date_data['Type'] == 'Historical']
            ax.bar(group_offset, historical_eps['EPS'], width=bar_width, color=historical_eps_color,
                   label='Historical EPS', align='center')

        if 'Forecast' in date_data['Type'].values:
            forecast_eps = date_data[date_data['Type'] == 'Forecast']
            ax.bar(group_offset + bar_width, forecast_eps['EPS'], width=bar_width, color=forecast_eps_color,
                   label='Forecast EPS', align='center')

    for rect in ax.patches:
        height = rect.get_height()
        x = rect.get_x() + rect.get_width() / 2
        y = height
        label_text = f'{height:.2f}'
        va = 'top' if height < 0 else 'bottom'
        ax.text(x, y, label_text, ha='center', va=va)

    ax.set_xlabel('Date')
    ax.set_ylabel('Earnings Per Share (EPS)')
    ax.set_title(f"{ticker} EPS (Historical & Forecasted)")
    ax.axhline(y=0, color='black', linewidth=1)

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

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best')

    return ax

def generate_financial_forecast_chart(ticker, combined_data, charts_output_dir, db_path, historical_data, forecast_data, analyst_counts):
    max_revenue = combined_data['Revenue'].max()
    max_net_income = combined_data['Net_Income'].max()
    max_value = max(max_revenue, max_net_income)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    bar_width = 0.3
    plot_bars(ax1, combined_data, bar_width, analyst_counts)

    format_axis(ax1, max_value)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    bar_width = 0.3
    plot_bars(ax1, combined_data, bar_width, analyst_counts)
    add_value_labels(ax1)
    format_chart(ax1, combined_data, charts_output_dir, ticker)

    fig, ax2 = plt.subplots(figsize=(10, 6))
    plot_eps(ticker, ax2, combined_data, analyst_counts, bar_width)
    plt.tight_layout()
    plt.savefig(f"{charts_output_dir}{ticker}_EPS_Forecast.png")
    plt.close(fig)

def calculate_yoy_growth(combined_data, analyst_counts):
    combined_data['Year'] = pd.to_datetime(combined_data['Date']).dt.year
    combined_data.sort_values(by='Date', inplace=True)

    combined_data['Revenue_YoY'] = combined_data['Revenue'].pct_change() * 100
    combined_data['Net_Income_YoY'] = combined_data['Net_Income'].pct_change() * 100

    yoy_table = combined_data.groupby('Year').tail(1).set_index('Year')
    yoy_table = yoy_table[['Revenue_YoY', 'Net_Income_YoY']]

    analyst_counts['Year'] = pd.to_datetime(analyst_counts['Date']).dt.year
    analyst_counts_grouped = analyst_counts.groupby('Year').tail(1).set_index('Year')[
        ['ForwardRevenueAnalysts', 'ForwardEPSAnalysts']]
    yoy_table = yoy_table.join(analyst_counts_grouped, how='left')

    yoy_table['Revenue_YoY'] = yoy_table['Revenue_YoY'].map(lambda x: f'{x:.1f}%' if not pd.isnull(x) else '')
    yoy_table['Net_Income_YoY'] = yoy_table['Net_Income_YoY'].map(lambda x: f'{x:.1f}%' if not pd.isnull(x) else '')

    yoy_table['ForwardRevenueAnalysts'] = yoy_table['ForwardRevenueAnalysts'].fillna(0).astype(int)
    yoy_table['ForwardEPSAnalysts'] = yoy_table['ForwardEPSAnalysts'].fillna(0).astype(int)

    yoy_table = yoy_table[['Revenue_YoY', 'ForwardRevenueAnalysts', 'Net_Income_YoY', 'ForwardEPSAnalysts']]

    avg_revenue_yoy = yoy_table['Revenue_YoY'].replace('', np.nan).dropna().apply(lambda x: float(x.strip('%'))).mean()
    avg_net_income_yoy = yoy_table['Net_Income_YoY'].replace('', np.nan).dropna().apply(
        lambda x: float(x.strip('%'))).mean()

    avg_row = pd.Series({
        'Revenue_YoY': f'{avg_revenue_yoy:.1f}%',
        'ForwardRevenueAnalysts': '',
        'Net_Income_YoY': f'{avg_net_income_yoy:.1f}%',
        'ForwardEPSAnalysts': ''
    }, name='Average')

    yoy_table = pd.concat([yoy_table, avg_row.to_frame().T])

    yoy_table.rename(columns={
        'Revenue_YoY': 'Revenue Growth (%)',
        'ForwardRevenueAnalysts': 'Revenue Analysts (#)',
        'Net_Income_YoY': 'EPS Growth (%)',
        'ForwardEPSAnalysts': 'EPS Analysts (#)'
    }, inplace=True)

    yoy_table_transposed = yoy_table.T

    return yoy_table_transposed

def style_negative(value):
    try:
        value_float = float(str(value).replace('%', '').strip())
        return 'color: red;' if value_float < 0 else ''
    except (ValueError, TypeError):
        return ''

def style_positive(value):
    try:
        value_float = float(str(value).replace('%', '').strip())
        return 'color: green;' if value_float > 0 else ''
    except (ValueError, TypeError):
        return ''

def save_yoy_growth_to_html(yoy_growth_table, charts_output_dir, ticker):
    filename = f"{ticker}_yoy_growth_tbl"
    styled_table = (yoy_growth_table.style
                    .applymap(style_negative)
                    .applymap(style_positive)
                    .set_table_styles({
                        'Revenue Growth (%)': [{'selector': 'td', 'props': [('text-align', 'center')]}],
                        'Revenue Analysts (#)': [{'selector': 'td', 'props': [('text-align', 'center')]}],
                        'EPS Growth (%)': [{'selector': 'td', 'props': [('text-align', 'center')]}],
                        'EPS Analysts (#)': [{'selector': 'td', 'props': [('text-align', 'center')]}]
                    }))
    html_table = styled_table.to_html()

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
    os.makedirs(charts_output_dir, exist_ok=True)
    full_path = os.path.join(charts_output_dir, f"{filename}.html")
    with open(full_path, 'w') as f:
        f.write(html_string)
    print(f"YoY Growth Table saved to {full_path}")

def generate_yoy_line_chart(chart_type, data, title, ylabel, output_path, analyst_counts=None, analyst_column=None):
    fig, ax = plt.subplots(figsize=(8, 5))

    data = data.drop(index='Average', errors='ignore')
    years = pd.to_numeric(data.index, errors='coerce')
    values = pd.to_numeric(data.values, errors='coerce')

    if analyst_counts is not None and analyst_column is not None:
        analyst_counts['Year'] = pd.to_numeric(analyst_counts['Year'], errors='coerce')
        analyst_counts[analyst_column] = pd.to_numeric(analyst_counts[analyst_column], errors='coerce')

    if np.any(np.isnan(values)) or np.isinf(np.max(values)) or np.isinf(np.min(values)):
        values = np.nan_to_num(values, nan=0.0, posinf=np.max(values[np.isfinite(values)]),
                               neginf=np.min(values[np.isfinite(values)]))

    historical_color = 'blue' if chart_type == "revenue" else 'green'
    forecast_color = 'grey'

    forecast_years = analyst_counts['Year'].values if analyst_counts is not None and analyst_column else []

    for i in range(len(years) - 1):
        if years[i] in forecast_years or years[i + 1] in forecast_years:
            color = forecast_color
        else:
            color = historical_color
        ax.plot(years[i:i + 2], values[i:i + 2], marker='o', linestyle='-', color=color)

    buffer = 5
    min_y_value = max(min(values) - buffer, -95)
    max_y_value = min(max(values) + buffer, 95)
    ax.set_ylim(min_y_value, max_y_value)

    for i, (year, value) in enumerate(zip(years, values)):
        label_pos = max(min(value + 2, 90), -90)
        va = 'top' if value < 0 else 'bottom'
        ax.text(year, label_pos, f'{value:.1f}%', ha='center', va=va, fontsize=10)

    x_labels = [
        f"{year}* ({analyst_counts.loc[analyst_counts['Year'] == year, analyst_column].iloc[0]})" 
        if year in forecast_years else str(year)
        for year in years
    ]
    ax.set_xticks(years)
    ax.set_xticklabels(x_labels)

    ax.set_title(title)
    ax.set_xlabel('Year | * = Forecasted Data')
    ax.set_ylabel(ylabel)
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    fig.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Chart saved to {output_path}")

def generate_revenue_yoy_change_chart(yoy_table, ticker, output_dir, analyst_counts, analyst_column):
    analyst_counts['Year'] = pd.to_numeric(analyst_counts['Year'], errors='coerce')
    analyst_counts[analyst_column] = pd.to_numeric(analyst_counts[analyst_column], errors='coerce')

    chart_type = "revenue"
    revenue_changes = pd.to_numeric(yoy_table.loc['Revenue Growth (%)'].replace('%', '', regex=True),
                                    errors='coerce').dropna()
    output_path = f"{output_dir}{ticker}_revenue_yoy_change.png"
    generate_yoy_line_chart(chart_type, revenue_changes, f"{ticker} Revenue Year-over-Year Change", "Revenue YoY (%)",
                            output_path, analyst_counts, analyst_column)

def generate_eps_yoy_change_chart(yoy_table, ticker, output_dir, analyst_counts, analyst_column):
    analyst_counts['Year'] = pd.to_numeric(analyst_counts['Year'], errors='coerce')
    analyst_counts[analyst_column] = pd.to_numeric(analyst_counts[analyst_column], errors='coerce')

    chart_type = "eps"
    eps_changes = pd.to_numeric(yoy_table.loc['EPS Growth (%)'].replace('%', '', regex=True), errors='coerce').dropna()
    output_path = f"{output_dir}{ticker}_eps_yoy_change.png"
    generate_yoy_line_chart(chart_type, eps_changes, f"{ticker} EPS Year-over-Year Change", "EPS YoY (%)", output_path,
                            analyst_counts, analyst_column)

def generate_forecast_charts_and_tables(ticker, db_path, charts_output_dir):
    historical_data, forecast_data, analyst_counts, shares_outstanding = fetch_financial_data(ticker, db_path)
    combined_data = prepare_data_for_plotting(historical_data, forecast_data, shares_outstanding, ticker)

    placeholder_image_path = os.path.join(charts_output_dir, 'No_forecast_data.png')
    revenue_forecast_path = os.path.join(charts_output_dir, f"{ticker}_Revenue_Net_Income_Forecast.png")
    eps_forecast_path = os.path.join(charts_output_dir, f"{ticker}_EPS_Forecast.png")
    revenue_yoy_path = os.path.join(charts_output_dir, f"{ticker}_revenue_yoy_change.png")
    eps_yoy_path = os.path.join(charts_output_dir, f"{ticker}_eps_yoy_change.png")

    if forecast_data.empty:
        shutil.copy(placeholder_image_path, revenue_forecast_path)
        shutil.copy(placeholder_image_path, eps_forecast_path)
        shutil.copy(placeholder_image_path, revenue_yoy_path)
        shutil.copy(placeholder_image_path, eps_yoy_path)
    else:
        generate_financial_forecast_chart(ticker, combined_data, charts_output_dir, db_path, historical_data,
                                          forecast_data, analyst_counts)

        yoy_growth_table = calculate_yoy_growth(combined_data, analyst_counts)
        save_yoy_growth_to_html(yoy_growth_table, charts_output_dir, ticker)

        generate_revenue_yoy_change_chart(yoy_growth_table, ticker, charts_output_dir, analyst_counts,
                                          'ForwardRevenueAnalysts')
        generate_eps_yoy_change_chart(yoy_growth_table, ticker, charts_output_dir, analyst_counts, 'ForwardEPSAnalysts')

    print(f"Completed generating charts and tables for {ticker}.")

# Example usage:
# ticker = 'AAPL'
# db_path = 'Stock Data.db'
# generate_forecast_charts_and_tables(ticker, db_path, 'charts/')

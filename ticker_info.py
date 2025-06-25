import yfinance as yf
import os
import sqlite3
from bs4 import BeautifulSoup
from datetime import datetime

DB_PATH = "Stock Data.db"

def fetch_stock_data(ticker, treasury_yield):
    try:
        stock = yf.Ticker(ticker)
        raw_info = stock.info or {}
    except Exception as e:
        print(f"Error retrieving market data for {ticker}: {e}")
        raw_info = {}

    current_price = raw_info.get('currentPrice') or raw_info.get('regularMarketPrice') \
        or raw_info.get('previousClose') or average_bid_ask(raw_info)

    forward_eps = raw_info.get('forwardEps')
    pe_ratio = raw_info.get('trailingPE')
    price_to_book = raw_info.get('priceToBook')
    marketcap = raw_info.get('marketCap')

    forward_pe_ratio = current_price / forward_eps if current_price and forward_eps else None
    treasury_yield = float(treasury_yield) / 100 if treasury_yield and treasury_yield != '-' else None

    implied_growth = calculate_implied_growth(pe_ratio, treasury_yield) if pe_ratio else '-'
    implied_growth_formatted = f"{implied_growth * 100:.1f}%" if isinstance(implied_growth, (int, float)) else 'N/A'

    implied_forward_growth = calculate_implied_growth(forward_pe_ratio, treasury_yield) if forward_pe_ratio else '-'
    implied_forward_growth_formatted = f"{implied_forward_growth * 100:.1f}%" if isinstance(implied_forward_growth, (int, float)) else '-'

    formatted_close_price = f"${current_price:.2f}" if current_price else '-'

    data = {
        'Close Price': formatted_close_price,
        'Market Cap': marketcap,
        'P/E Ratio': f"{pe_ratio:.1f}" if pe_ratio else '-',
        'Forward P/E Ratio': f"{forward_pe_ratio:.1f}" if forward_pe_ratio else '-',
        'Implied Growth*': implied_growth_formatted,
        'Implied Forward Growth*': implied_forward_growth_formatted,
        'P/B Ratio': f"{price_to_book:.1f}" if price_to_book else '-',
    }

    return data, marketcap, implied_growth, implied_forward_growth


def calculate_implied_growth(pe_ratio, treasury_yield):
    if pe_ratio is None or treasury_yield is None or pe_ratio == 0:
        return '-'
    try:
        result = ((pe_ratio / 10) ** (1 / 10)) + treasury_yield - 1
        return result if isinstance(result, (int, float)) else '-'
    except Exception as e:
        print(f"[calculate_implied_growth] Error computing implied growth: {e}")
        return '-'


def average_bid_ask(info):
    bid = info.get('bid')
    ask = info.get('ask')
    if bid and ask:
        return (bid + ask) / 2
    return None


def format_number(value):
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
    fetched_data, marketcap, ttm_growth, forward_growth = fetch_stock_data(ticker, treasury_yield)

    # Record implied growths
    today_str = datetime.today().strftime("%Y-%m-%d")
    record_implied_growth_history(ticker, today_str, ttm_growth, forward_growth)

    return fetched_data, marketcap


def generate_html_table(data, ticker):
    html_content = """
    <style>
        table {
            width: 80%;
            margin-left: auto;
            margin-right: auto;
            border-collapse: collapse;
            text-align: center;
            font-family: 'Arial', sans-serif;
        }
        th, td {
            padding: 8px 12px;
        }
    </style>
    <table>
    <tr>"""
    for key in data:
        html_content += f"<th>{key}</th>"
    html_content += "</tr><tr>"
    for key, value in data.items():
        if key == 'Market Cap' and isinstance(value, (int, float)):
            formatted_value = format_number(value)
        else:
            formatted_value = value
        html_content += f"<td>{formatted_value}</td>"
    html_content += "</tr></table>"

    file_path = f"charts/{ticker}_ticker_info.html"
    with open(file_path, 'w') as file:
        file.write(html_content)

    return file_path


def record_implied_growth_history(ticker, date_str, ttm_growth, forward_growth):
    os.makedirs("charts", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Implied_Growth_History (
            ticker TEXT,
            growth_type TEXT CHECK(growth_type IN ('TTM', 'Forward')),
            growth_value REAL,
            date_recorded TEXT,
            UNIQUE(ticker, growth_type, date_recorded)
        )
    ''')

    def try_insert(tkr, typ, val, date):
        if val in ('-', None) or not isinstance(val, (int, float)) or isinstance(val, complex):
            print(f"[try_insert] Skipped insert for {tkr} ({typ}) — value: {val}")
            return
        cursor.execute('''
            INSERT OR IGNORE INTO Implied_Growth_History (ticker, growth_type, growth_value, date_recorded)
            VALUES (?, ?, ?, ?)
        ''', (tkr, typ, round(val, 6), date))

    try_insert(ticker, 'TTM', ttm_growth, date_str)
    try_insert(ticker, 'Forward', forward_growth, date_str)

    conn.commit()
    conn.close()


if __name__ == "__main__":
    ticker = 'AAPL'
    treasury_yield = '3.5'
    prepared_data, marketcap = prepare_data_for_display(ticker, treasury_yield)
    html_file_path = generate_html_table(prepared_data, ticker)
    print(f"HTML content has been written to {html_file_path}")

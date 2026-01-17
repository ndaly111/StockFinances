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
    # Retrieve the annual dividend. ``dividendRate`` is preferred but some
    # tickers only populate ``trailingAnnualDividendRate``.  We avoid using the
    # vendor-provided yield values because they may be stale or expressed as a
    # percentage already.  Instead, always compute the yield from the annual
    # dividend divided by the current share price.
    dividend_rate = raw_info.get('dividendRate')
    if dividend_rate is None:
        dividend_rate = raw_info.get('trailingAnnualDividendRate')

    dividend_yield = None
    if dividend_rate is not None and current_price:
        try:
            dividend_yield = dividend_rate / current_price
        except Exception:
            dividend_yield = None

    forward_pe_ratio = current_price / forward_eps if current_price and forward_eps else None
    treasury_yield = float(treasury_yield) / 100 if treasury_yield and treasury_yield != '-' else None

    implied_growth = calculate_implied_growth(pe_ratio, treasury_yield) if pe_ratio else '-'
    implied_growth_formatted = f"{implied_growth * 100:.1f}%" if isinstance(implied_growth, (int, float)) else 'N/A'

    implied_forward_growth = calculate_implied_growth(forward_pe_ratio, treasury_yield) if forward_pe_ratio else '-'
    implied_forward_growth_formatted = f"{implied_forward_growth * 100:.1f}%" if isinstance(implied_forward_growth, (int, float)) else '-'

    formatted_close_price = f"${current_price:.2f}" if current_price else '-'
    dividend_amount_str = f"${dividend_rate:.2f}" if dividend_rate else '-'
    dividend_yield_str = f"{dividend_yield * 100:.1f}%" if dividend_yield else None
    if dividend_amount_str != '-' and dividend_yield_str:
        formatted_dividend = f"{dividend_amount_str}<br>{dividend_yield_str}"
    else:
        formatted_dividend = dividend_amount_str

    data = {
        'Close Price': formatted_close_price,
        'Market Cap': marketcap,
        'P/E Ratio': f"{pe_ratio:.1f}" if pe_ratio else '-',
        'Forward P/E Ratio': f"{forward_pe_ratio:.1f}" if forward_pe_ratio else '-',
        'Implied Growth*': implied_growth_formatted,
        'Implied Forward Growth*': implied_forward_growth_formatted,
        'Dividend': formatted_dividend,
        'P/B Ratio': f"{price_to_book:.1f}" if price_to_book else '-',
    }

    return data, marketcap, implied_growth, implied_forward_growth, forward_eps


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


def ensure_history_schema(cursor):
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Implied_Growth_History (
            ticker TEXT,
            growth_type TEXT CHECK(growth_type IN ('TTM', 'Forward')),
            growth_value REAL,
            date_recorded TEXT,
            UNIQUE(ticker, growth_type, date_recorded)
        )
    ''')
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_igh_ticker_date "
        "ON Implied_Growth_History (ticker, date_recorded)"
    )
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Forward_EPS_History (
            date_recorded TEXT,
            ticker TEXT,
            forward_eps REAL,
            source TEXT,
            PRIMARY KEY (date_recorded, ticker)
        )
    ''')
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_forward_eps_hist_ticker_date "
        "ON Forward_EPS_History (ticker, date_recorded)"
    )


def prepare_data_for_display(ticker, treasury_yield, conn=None, cursor=None, commit=None):
    fetched_data, marketcap, ttm_growth, forward_growth, forward_eps = fetch_stock_data(ticker, treasury_yield)

    # Record implied growths
    today_str = datetime.today().strftime("%Y-%m-%d")
    if commit is None:
        commit = False if (conn is not None or cursor is not None) else True
    record_implied_growth_history(
        ticker, today_str, ttm_growth, forward_growth, conn=conn, cursor=cursor, commit=commit
    )

    # Record forward EPS so revisions can be tracked over time.
    record_forward_eps_history(
        ticker, today_str, forward_eps, conn=conn, cursor=cursor, commit=commit
    )

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


def record_implied_growth_history(ticker, date_str, ttm_growth, forward_growth, conn=None, cursor=None, commit=True):
    os.makedirs("charts", exist_ok=True)
    own_conn = False
    if cursor is None:
        if conn is None:
            conn = sqlite3.connect(DB_PATH, timeout=30)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA busy_timeout=30000;")
            own_conn = True
        cursor = conn.cursor()

    if own_conn:
        ensure_history_schema(cursor)

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

    if commit or own_conn:
        cursor.connection.commit()
    if own_conn:
        conn.close()


def record_forward_eps_history(ticker, date_str, forward_eps, conn=None, cursor=None, commit=True):
    os.makedirs("charts", exist_ok=True)
    own_conn = False
    if cursor is None:
        if conn is None:
            conn = sqlite3.connect(DB_PATH, timeout=30)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA busy_timeout=30000;")
            own_conn = True
        cursor = conn.cursor()

    if own_conn:
        ensure_history_schema(cursor)

    try:
        if forward_eps is None:
            if own_conn:
                conn.close()
            return
        forward_eps = float(forward_eps)
    except Exception:
        if own_conn:
            conn.close()
        return

    if forward_eps != forward_eps:
        if own_conn:
            conn.close()
        return

    cursor.execute(
        '''
        INSERT OR REPLACE INTO Forward_EPS_History (date_recorded, ticker, forward_eps, source)
        VALUES (?, ?, ?, ?)
        ''',
        (date_str, ticker, forward_eps, "yfinance.info.forwardEps"),
    )

    if commit or own_conn:
        cursor.connection.commit()
    if own_conn:
        conn.close()


if __name__ == "__main__":
    ticker = 'AAPL'
    treasury_yield = '3.5'
    prepared_data, marketcap = prepare_data_for_display(ticker, treasury_yield)
    html_file_path = generate_html_table(prepared_data, ticker)
    print(f"HTML content has been written to {html_file_path}")

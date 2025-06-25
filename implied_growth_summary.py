import yfinance as yf
import os
import sqlite3
from datetime import datetime

DB_PATH = "Stock Data.db"

def fetch_stock_data(ticker, treasury_yield):
    """
    Safely fetches stock info using yfinance, avoiding NoneType errors
    and handling fallback logic for currentPrice.
    Returns a tuple: (display_data_dict, marketcap, implied_growth, implied_forward_growth)
    """
    # --- Fetch raw info ---
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}
    except Exception as e:
        print(f"Error retrieving market data for {ticker}: {e}")
        info = {}

    # --- Extract prices & ratios ---
    current_price = (
        info.get('currentPrice')
        or info.get('regularMarketPrice')
        or info.get('previousClose')
        or average_bid_ask(info)
    )
    forward_eps    = info.get('forwardEps')
    pe_ratio       = info.get('trailingPE')
    price_to_book  = info.get('priceToBook')
    marketcap      = info.get('marketCap')

    forward_pe_ratio = None
    if current_price is not None and forward_eps:
        forward_pe_ratio = current_price / forward_eps

    # --- Normalize treasury_yield ---
    try:
        treasury_yield = float(treasury_yield) / 100
    except:
        treasury_yield = None

    # --- Calculate implied growths as floats or None ---
    implied_growth         = calculate_implied_growth(pe_ratio, treasury_yield)
    implied_forward_growth = calculate_implied_growth(forward_pe_ratio, treasury_yield)

    # --- Format for display ---
    implied_growth_str         = f"{implied_growth * 100:.1f}%" if implied_growth is not None else 'N/A'
    implied_forward_growth_str = f"{implied_forward_growth * 100:.1f}%" if implied_forward_growth is not None else 'N/A'
    formatted_close_price      = f"${current_price:.2f}" if current_price is not None else '-'

    display_data = {
        'Close Price': formatted_close_price,
        'Market Cap':  marketcap,
        'P/E Ratio':  f"{pe_ratio:.1f}" if pe_ratio is not None else '-',
        'Forward P/E Ratio': f"{forward_pe_ratio:.1f}" if forward_pe_ratio is not None else '-',
        'Implied Growth*':         implied_growth_str,
        'Implied Forward Growth*': implied_forward_growth_str,
        'P/B Ratio':  f"{price_to_book:.1f}" if price_to_book is not None else '-',
    }

    return display_data, marketcap, implied_growth, implied_forward_growth

def calculate_implied_growth(pe_ratio, treasury_yield):
    """
    Reverse‐compound model: implied_growth = ((PE/10)^(1/10)) + r − 1
    Returns a float or None if inputs are invalid.
    """
    if pe_ratio is None or treasury_yield is None or pe_ratio <= 0:
        return None
    try:
        return ((pe_ratio / 10) ** (1 / 10)) + treasury_yield - 1
    except Exception:
        return None

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

    # Record implied growths in the DB
    today_str = datetime.today().strftime("%Y-%m-%d")
    record_implied_growth_history(ticker, today_str, ttm_growth, forward_growth)

    return fetched_data, marketcap

def generate_html_table(data, ticker):
    # (unchanged)
    html_content = """
    <style>
        table { width: 80%; margin: auto; border-collapse: collapse; text-align: center; font-family: 'Arial', sans-serif;}
        th, td { padding: 8px 12px; }
    </style>
    <table><tr>"""
    for key in data:
        html_content += f"<th>{key}</th>"
    html_content += "</tr><tr>"
    for key, value in data.items():
        if key == 'Market Cap' and isinstance(value, (int, float)):
            formatted = format_number(value)
        else:
            formatted = value
        html_content += f"<td>{formatted}</td>"
    html_content += "</tr></table>"""

    path = f"charts/{ticker}_ticker_info.html"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(html_content)
    return path

def record_implied_growth_history(ticker, date_str, ttm_growth, forward_growth):
    """
    Creates table if needed and INSERTs TTM/Forward growth for today,
    skipping duplicates and invalid values.
    """
    os.makedirs("charts", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS Implied_Growth_History (
            ticker TEXT,
            growth_type TEXT CHECK(growth_type IN ('TTM','Forward')),
            growth_value REAL,
            date_recorded TEXT,
            UNIQUE(ticker, growth_type, date_recorded)
        )
    """)

    def try_insert(tkr, typ, val, date):
        # Skip if no valid float
        if val is None or not isinstance(val, (int, float)):
            return
        cur.execute("""
            INSERT OR IGNORE INTO Implied_Growth_History 
            (ticker, growth_type, growth_value, date_recorded)
            VALUES (?, ?, ?, ?)
        """, (tkr, typ, round(val, 6), date))

    try_insert(ticker, 'TTM',     ttm_growth,     date_str)
    try_insert(ticker, 'Forward', forward_growth, date_str)

    conn.commit()
    conn.close()

if __name__ == "__main__":
    # Quick test
    ticker = 'AAPL'
    treasury_yield = '3.5'  # percent
    data, mcap = prepare_data_for_display(ticker, treasury_yield)
    generate_html_table(data, ticker)

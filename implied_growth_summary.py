# Writing the fully corrected ticker_info.py with the proper calculate_implied_growth guard

from pathlib import Path

ticker_info_code = '''\
import yfinance as yf
import os
import sqlite3
from datetime import datetime

DB_PATH = "Stock Data.db"

def fetch_stock_data(ticker, treasury_yield):
    """
    Safely fetches stock info using yfinance and computes implied growth values.
    Returns (display_data_dict, marketcap, implied_growth, implied_forward_growth).
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}
    except Exception as e:
        print(f"Error retrieving market data for {ticker}: {e}")
        info = {}

    # --- Extract prices & ratios safely ---
    current_price = (
        info.get("currentPrice")
        or info.get("regularMarketPrice")
        or info.get("previousClose")
        or average_bid_ask(info)
    )
    forward_eps    = info.get("forwardEps")
    pe_ratio       = info.get("trailingPE")
    price_to_book  = info.get("priceToBook")
    marketcap      = info.get("marketCap")

    # Calculate forward P/E
    forward_pe_ratio = None
    if current_price is not None and forward_eps:
        try:
            forward_pe_ratio = current_price / forward_eps
        except Exception:
            forward_pe_ratio = None

    # Normalize treasury_yield to a decimal
    try:
        treasury_yield = float(treasury_yield) / 100
    except Exception:
        treasury_yield = None

    # Compute implied growth, guarding against invalid inputs
    implied_growth         = calculate_implied_growth(pe_ratio, treasury_yield)
    implied_forward_growth = calculate_implied_growth(forward_pe_ratio, treasury_yield)

    # Format strings for display
    def fmt(val):
        return f"{val * 100:.1f}%" if isinstance(val, (int, float)) else "N/A"

    display_data = {
        "Close Price":            f"${current_price:.2f}" if isinstance(current_price, (int, float)) else "-",
        "Market Cap":             marketcap,
        "P/E Ratio":              f"{pe_ratio:.1f}" if isinstance(pe_ratio, (int, float)) else "-",
        "Forward P/E Ratio":      f"{forward_pe_ratio:.1f}" if isinstance(forward_pe_ratio, (int, float)) else "-",
        "Implied Growth*":        fmt(implied_growth),
        "Implied Forward Growth*":fmt(implied_forward_growth),
        "P/B Ratio":              f"{price_to_book:.1f}" if isinstance(price_to_book, (int, float)) else "-",
    }

    return display_data, marketcap, implied_growth, implied_forward_growth

def calculate_implied_growth(pe_ratio, treasury_yield):
    """
    Computes implied growth = ((PE/10)^(1/10)) + r - 1.
    Returns a float, or None if inputs are invalid or would produce a complex result.
    """
    if pe_ratio is None or treasury_yield is None:
        return None
    try:
        base = pe_ratio / 10
        # Guard negative or zero base to avoid complex results
        if base <= 0:
            return None
        return (base ** (1/10)) + treasury_yield - 1
    except Exception:
        return None

def average_bid_ask(info):
    bid = info.get("bid")
    ask = info.get("ask")
    if isinstance(bid, (int, float)) and isinstance(ask, (int, float)):
        return (bid + ask) / 2
    return None

def format_number(value):
    if value is None:
        return "N/A"
    if abs(value) >= 1e12:
        return f"${value/1e12:.2f}T"
    if abs(value) >= 1e9:
        return f"${value/1e9:.2f}B"
    if abs(value) >= 1e6:
        return f"${value/1e6:.2f}M"
    return f"${value:.2f}"

def record_implied_growth_history(ticker, date_str, ttm_growth, fwd_growth):
    """
    Inserts one TTM and one Forward implied growth per ticker per date,
    skipping any values that are None or not real floats.
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

    def try_insert(growth_type, value):
        if not isinstance(value, (int, float)):
            return
        try:
            rv = round(value, 6)
        except Exception:
            return
        cur.execute("""
            INSERT OR IGNORE INTO Implied_Growth_History
            (ticker, growth_type, growth_value, date_recorded)
            VALUES (?, ?, ?, ?)
        """, (ticker, growth_type, rv, date_str))

    try_insert("TTM", ttm_growth)
    try_insert("Forward", fwd_growth)

    conn.commit()
    conn.close()

def prepare_data_for_display(ticker, treasury_yield):
    """
    Fetches display data and records implied growth history.
    Returns (display_data_dict, marketcap).
    """
    data, marketcap, ttm, fwd = fetch_stock_data(ticker, treasury_yield)
    today_str = datetime.today().strftime("%Y-%m-%d")
    record_implied_growth_history(ticker, today_str, ttm, fwd)
    return data, marketcap

def generate_html_table(data, ticker):
    """
    Creates the HTML table for ticker info.
    """
    html = """
    <style>
      table {width:80%;margin:auto;border-collapse:collapse;text-align:center;font-family:Arial,sans-serif;}
      th, td {padding:8px 12px;}
    </style>
    <table><tr>"""
    for key in data:
        html += f"<th>{key}</th>"
    html += "</tr><tr>"
    for val in data.values():
        html += f"<td>{val}</td>"
    html += "</tr></table>"
    path = f"charts/{ticker}_ticker_info.html"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(html)
    return path

if __name__ == "__main__":
    # Quick sanity test
    d, m, t, f = fetch_stock_data("AAPL", "3.5")
    print(d)
    prepare_data_for_display("AAPL", "3.5")
'''

# Write to file
output = Path("ticker_info.py")
output.write_text(ticker_info_code)
output

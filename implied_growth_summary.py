import os
import sqlite3
import math
import numbers
from datetime import datetime
import yfinance as yf

# ───────────────────────────────────────────────────────────
# Configuration
# ───────────────────────────────────────────────────────────
DB_PATH    = 'Stock Data.db'
CHARTS_DIR = 'charts'
os.makedirs(CHARTS_DIR, exist_ok=True)

# ───────────────────────────────────────────────────────────
# Fetching & Growth Calculation
# ───────────────────────────────────────────────────────────
def fetch_stock_data(ticker, treasury_yield):
    """
    Fetches price & P/E info via yfinance, safely computes implied growth.
    """
    try:
        stock = yf.Ticker(ticker)
        info  = stock.info or {}
    except Exception as e:
        print(f"[fetch_stock_data] ERROR retrieving {ticker}: {e}")
        info = {}

    # Current price fallback chain
    current_price = (
        info.get('currentPrice')
        or info.get('regularMarketPrice')
        or info.get('previousClose')
        or average_bid_ask(info)
    )

    forward_eps      = info.get('forwardEps')
    pe_ratio         = info.get('trailingPE')
    price_to_book    = info.get('priceToBook')
    marketcap        = info.get('marketCap')

    # Forward P/E fallback
    forward_pe_ratio = None
    if isinstance(current_price, (int, float)) and isinstance(forward_eps, (int, float)) and forward_eps:
        forward_pe_ratio = current_price / forward_eps

    # Treasury yield to decimal
    try:
        treasury_yield = float(treasury_yield) / 100
    except Exception:
        treasury_yield = None

    # Compute implied growths
    implied_growth         = calculate_implied_growth(pe_ratio, treasury_yield)
    implied_forward_growth = calculate_implied_growth(forward_pe_ratio, treasury_yield)

    print(f"[fetch_stock_data] {ticker} → PE={pe_ratio}, FwdPE={forward_pe_ratio}, "
          f"ImplG={implied_growth}, ImplFwdG={implied_forward_growth}")

    # Format for display
    def pct(x):
        return f"{x*100:.1f}%" if isinstance(x, (int, float)) else "N/A"

    data = {
        'Close Price':            f"${current_price:.2f}" if isinstance(current_price, (int, float)) else '-',
        'Market Cap':             marketcap,
        'P/E Ratio':              f"{pe_ratio:.1f}" if isinstance(pe_ratio, (int, float)) else '-',
        'Forward P/E Ratio':      f"{forward_pe_ratio:.1f}" if isinstance(forward_pe_ratio, (int, float)) else '-',
        'Implied Growth*':        pct(implied_growth),
        'Implied Forward Growth*':pct(implied_forward_growth),
        'P/B Ratio':              f"{price_to_book:.1f}" if isinstance(price_to_book, (int, float)) else '-',
    }

    return data, marketcap, implied_growth, implied_forward_growth

def calculate_implied_growth(pe_ratio, treasury_yield):
    """
    Reverse‐compound formula: ((PE/10)**(1/10)) + r - 1.
    Returns None if inputs invalid or would produce a complex result.
    """
    if pe_ratio is None or treasury_yield is None:
        return None
    base = pe_ratio / 10
    # Avoid complex roots: require positive real base
    if not isinstance(base, (int, float)) or base <= 0:
        return None
    try:
        return (base ** (1/10)) + treasury_yield - 1
    except Exception as e:
        print(f"[calculate_implied_growth] SKIP complex outcome for PE={pe_ratio}: {e}")
        return None

def average_bid_ask(info):
    bid = info.get('bid')
    ask = info.get('ask')
    if isinstance(bid, (int, float)) and isinstance(ask, (int, float)):
        return (bid + ask) / 2
    return None

# ───────────────────────────────────────────────────────────
# Recording History
# ───────────────────────────────────────────────────────────
def record_implied_growth_history(ticker, date_str, ttm_growth, fwd_growth):
    """
    Creates history table and inserts one TTM + one Forward per ticker/day,
    skipping invalid values before rounding.
    """
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
        print(f"[record] {ticker} {growth_type}: raw={value!r} ({type(value).__name__})")
        # Skip None or non-real
        if not isinstance(value, (int, float)):
            print("  → SKIP: not a real float")
            return
        # Skip NaN/inf
        if not math.isfinite(value):
            print("  → SKIP: non-finite (NaN/inf)")
            return
        # Safe to round & insert
        rounded = round(value, 6)
        cur.execute("""
            INSERT OR IGNORE INTO Implied_Growth_History
            (ticker, growth_type, growth_value, date_recorded)
            VALUES (?, ?, ?, ?)
        """, (ticker, growth_type, rounded, date_str))
        print(f"  → INSERTED: {rounded}")

    try_insert('TTM',     ttm_growth)
    try_insert('Forward', fwd_growth)

    conn.commit()
    conn.close()

# ───────────────────────────────────────────────────────────
# Public Helpers
# ───────────────────────────────────────────────────────────
def prepare_data_for_display(ticker, treasury_yield):
    """
    Fetches display-ready data and records history.
    """
    data, marketcap, ttm, fwd = fetch_stock_data(ticker, treasury_yield)
    today_str = datetime.today().strftime('%Y-%m-%d')
    record_implied_growth_history(ticker, today_str, ttm, fwd)
    return data, marketcap

def generate_html_table(data, ticker):
    """
    Writes the ticker_info HTML table for the UI.
    """
    style = """
    <style>
        table {width:80%;margin:auto;border-collapse:collapse;text-align:center;font-family:Arial,sans-serif;}
        th, td {padding:8px 12px;}
    </style>
    """
    html = style + "<table><tr>" + "".join(f"<th>{k}</th>" for k in data) + "</tr><tr>"
    html += "".join(f"<td>{v}</td>" for v in data.values()) + "</tr></table>"

    path = os.path.join(CHARTS_DIR, f"{ticker}_ticker_info.html")
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    return path

# ───────────────────────────────────────────────────────────
# Quick Local Test
# ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    d, m, t, f = fetch_stock_data("AAPL", 0.035)
    print("[TEST] display data:", d)
    prepare_data_for_display("AAPL", 0.035)

import yfinance as yf
import os
import sqlite3
import math
import numbers
from datetime import datetime

DB_PATH = "Stock Data.db"

# ────────────────────────────────────────────────────────────
# Fetching & growth calculations
# ────────────────────────────────────────────────────────────
def fetch_stock_data(ticker, treasury_yield):
    """Safely fetch stock info and compute implied growth values."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}
    except Exception as e:
        print(f"[fetch_stock_data] ERROR retrieving {ticker}: {e}")
        info = {}

    current_price = (
        info.get("currentPrice")
        or info.get("regularMarketPrice")
        or info.get("previousClose")
        or average_bid_ask(info)
    )
    forward_eps   = info.get("forwardEps")
    pe_ratio      = info.get("trailingPE")
    price_to_book = info.get("priceToBook")
    marketcap     = info.get("marketCap")

    forward_pe_ratio = None
    if isinstance(current_price, (int, float)) and isinstance(forward_eps, (int, float)) and forward_eps:
        forward_pe_ratio = current_price / forward_eps

    try:
        treasury_yield = float(treasury_yield) / 100
    except Exception:
        treasury_yield = None

    implied_growth         = calculate_implied_growth(pe_ratio, treasury_yield)
    implied_forward_growth = calculate_implied_growth(forward_pe_ratio, treasury_yield)

    print(f"[fetch_stock_data] {ticker} → pe={pe_ratio}, fwd_pe={forward_pe_ratio}, "
          f"impl={implied_growth}, impl_fwd={implied_forward_growth}")

    def pct(val):
        return f"{val*100:.1f}%" if isinstance(val, (int, float)) else "N/A"

    display = {
        "Close Price":            f"${current_price:.2f}" if isinstance(current_price, (int, float)) else "-",
        "Market Cap":             marketcap,
        "P/E Ratio":              f"{pe_ratio:.1f}" if isinstance(pe_ratio, (int, float)) else "-",
        "Forward P/E Ratio":      f"{forward_pe_ratio:.1f}" if isinstance(forward_pe_ratio, (int, float)) else "-",
        "Implied Growth*":        pct(implied_growth),
        "Implied Forward Growth*":pct(implied_forward_growth),
        "P/B Ratio":              f"{price_to_book:.1f}" if isinstance(price_to_book, (int, float)) else "-",
    }

    return display, marketcap, implied_growth, implied_forward_growth

def calculate_implied_growth(pe_ratio, treasury_yield):
    """Reverse-compound model, returns None if inputs invalid."""
    if pe_ratio is None or treasury_yield is None:
        return None
    base = pe_ratio / 10
    if not isinstance(base, (int, float)) or base <= 0:
        return None
    try:
        return (base ** (1/10)) + treasury_yield - 1
    except Exception as e:
        print(f"[calc_implied_growth] ERROR: {e}")
        return None

def average_bid_ask(info):
    bid = info.get("bid"); ask = info.get("ask")
    if isinstance(bid, (int, float)) and isinstance(ask, (int, float)):
        return (bid + ask) / 2
    return None

# ────────────────────────────────────────────────────────────
# Recording history
# ────────────────────────────────────────────────────────────
def record_implied_growth_history(ticker, date_str, ttm_growth, fwd_growth):
    """Insert growth values if they are finite real numbers; skip otherwise."""
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS Implied_Growth_History (
            ticker TEXT,
            growth_type TEXT CHECK (growth_type IN ('TTM','Forward')),
            growth_value REAL,
            date_recorded TEXT,
            UNIQUE(ticker, growth_type, date_recorded)
        )
    """)

    def try_insert(growth_type, value):
        print(f"[record] {ticker} {growth_type} raw={value!r} ({type(value).__name__})")
        # must be a finite real float/int (bool is subclass of int—exclude)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            print("  → SKIP: not a real float")
            return
        if not math.isfinite(value):
            print("  → SKIP: non-finite")
            return
        rounded = round(value, 6)
        cur.execute(
            "INSERT OR IGNORE INTO Implied_Growth_History "
            "(ticker, growth_type, growth_value, date_recorded) "
            "VALUES (?, ?, ?, ?)",
            (ticker, growth_type, rounded, date_str)
        )
        print(f"  → INSERTED {rounded}")

    try_insert("TTM",     ttm_growth)
    try_insert("Forward", fwd_growth)

    conn.commit(); conn.close()

# ────────────────────────────────────────────────────────────
# Public helpers used by main_remote.py
# ────────────────────────────────────────────────────────────
def prepare_data_for_display(ticker, treasury_yield):
    data, marketcap, ttm, fwd = fetch_stock_data(ticker, treasury_yield)
    today_str = datetime.today().strftime("%Y-%m-%d")
    record_implied_growth_history(ticker, today_str, ttm, fwd)
    return data, marketcap

def generate_html_table(data, ticker):
    style = """
    <style>
      table {width:80%;margin:auto;border-collapse:collapse;text-align:center;font-family:Arial,sans-serif;}
      th, td {padding:8px 12px;}
    </style>"""
    html  = style + "<table><tr>" + "".join(f"<th>{h}</th>" for h in data) + "</tr><tr>"
    html += "".join(f"<td>{v}</td>" for v in data.values()) + "</tr></table>"
    path  = f"charts/{ticker}_ticker_info.html"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(html)
    return path

# ────────────────────────────────────────────────────────────
# Local manual test
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _d, _m, _t, _f = fetch_stock_data("AAPL", "3.5")
    prepare_data_for_display("AAPL", "3.5")

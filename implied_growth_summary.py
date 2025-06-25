import os
import sqlite3
from datetime import datetime

import yfinance as yf
import pandas as pd

DB_PATH = "Stock Data.db"

def fetch_market_cap(ticker: str) -> float | None:
    try:
        stock = yf.Ticker(ticker)
        return stock.info.get("marketCap", None)
    except Exception as e:
        print(f"Error fetching market cap for {ticker}: {e}")
        return None

def calculate_implied_growth(pe_ratio, treasury_yield):
    try:
        growth_rate = pe_ratio * treasury_yield
        if isinstance(growth_rate, complex) or growth_rate is None or abs(growth_rate) > 1e6:
            print(f"Skipping invalid growth rate: {growth_rate}")
            return None
        return round(growth_rate, 4)
    except Exception as e:
        print(f"Error calculating implied growth: {e}")
        return None

def try_insert(tkr, typ, val, date):
    if val is None:
        print(f"⚠️ Skipping {tkr} {typ}: value is None")
        return
    if isinstance(val, complex):
        print(f"⚠️ Skipping {tkr} {typ}: value is complex: {val}")
        return
    if abs(val) > 1e6:
        print(f"⚠️ Skipping {tkr} {typ}: value is unreasonably large: {val}")
        return

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO implied_growth (ticker, type, growth_rate, date)
            VALUES (?, ?, ?, ?)
        ''', (tkr, typ, round(val, 6), date))
        conn.commit()
        conn.close()
        print(f"✅ Inserted implied growth for {tkr} ({typ}): {val}")
    except Exception as e:
        print(f"❌ Failed to insert {tkr} ({typ}) due to error: {e}")

def record_implied_growth_history(ticker, date_str, ttm_growth, forward_growth):
    try_insert(ticker, 'TTM', ttm_growth, date_str)
    try_insert(ticker, 'Forward', forward_growth, date_str)

def prepare_data_for_display(ticker: str, treasury_yield: float):
    print(f"Preparing implied growth data for {ticker}...")

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT trailing_pe, forward_pe FROM Valuation WHERE ticker = ?", (ticker,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            print(f"No valuation data found for {ticker}")
            return None, None

        trailing_pe, forward_pe = row
        ttm_growth = calculate_implied_growth(trailing_pe, treasury_yield)
        forward_growth = calculate_implied_growth(forward_pe, treasury_yield)

        today_str = datetime.today().strftime("%Y-%m-%d")
        record_implied_growth_history(ticker, today_str, ttm_growth, forward_growth)

        data = {
            "Ticker": ticker,
            "Trailing P/E": trailing_pe,
            "Forward P/E": forward_pe,
            "TTM Implied Growth": ttm_growth,
            "Forward Implied Growth": forward_growth,
        }
        return data, fetch_market_cap(ticker)

    except Exception as e:
        print(f"Error preparing data for {ticker}: {e}")
        return None, None

# 🚨 DO NOT CHANGE THIS FUNCTION NAME
def generate_all_summaries():
    print("⏳ Starting implied growth data preparation...")

    from ticker_manager import read_tickers
    tickers = read_tickers("tickers.csv")
    print(f"Found {len(tickers)} tickers")

    results = []
    for ticker in tickers:
        data, marketcap = prepare_data_for_display(ticker, treasury_yield=0.045)
        if data:
            data["Market Cap"] = marketcap
            results.append(data)

    df = pd.DataFrame(results)
    os.makedirs("charts", exist_ok=True)
    df.to_html("charts/implied_growth_summary.html", index=False)
    print("✅ All summaries generated and saved to charts/implied_growth_summary.html")

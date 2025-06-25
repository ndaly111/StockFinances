import os
import sqlite3
from datetime import datetime
import yfinance as yf
from implied_growth import calculate_implied_growth
from utils import fetch_balance_sheet_data

DB_PATH = "Stock Data.db"
OUTPUT_DIR = "charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def record_implied_growth_history(tkr, date_str, ttm_val, forward_val):
    print(f"Recording implied growth for {tkr} on {date_str}")
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS Implied_Growth (
                    ticker TEXT,
                    type TEXT,
                    value REAL,
                    date TEXT,
                    PRIMARY KEY (ticker, type, date)
                )
            ''')
            def try_insert(tkr, typ, val, date):
                if isinstance(val, complex):
                    print(f"Warning: Skipping complex number for {tkr} {typ}: {val}")
                    return
                if not isinstance(val, (int, float)):
                    print(f"Warning: Skipping non-numeric value for {tkr} {typ}: {val}")
                    return
                try:
                    cursor.execute('''
                        INSERT OR REPLACE INTO Implied_Growth (ticker, type, value, date)
                        VALUES (?, ?, ?, ?)
                    ''', (tkr, typ, round(val, 6), date))
                    print(f"Inserted {typ} for {tkr}: {val}")
                except Exception as e:
                    print(f"Failed to insert {typ} for {tkr}: {e}")
            try_insert(tkr, 'TTM', ttm_val, date_str)
            try_insert(tkr, 'Forward', forward_val, date_str)
            conn.commit()
    except Exception as e:
        print(f"Database error while recording implied growth: {e}")

def prepare_data_for_display(ticker, treasury_yield):
    print(f"Preparing data for {ticker}")
    try:
        ttm_growth, forward_growth = calculate_implied_growth(ticker, treasury_yield)
        print(f"Implied TTM growth: {ttm_growth}, Forward growth: {forward_growth}")
    except Exception as e:
        print(f"Error calculating implied growth for {ticker}: {e}")
        return {}, None
    try:
        today_str = datetime.today().strftime('%Y-%m-%d')
        record_implied_growth_history(ticker, today_str, ttm_growth, forward_growth)
    except Exception as e:
        print(f"Error recording growth history for {ticker}: {e}")
    try:
        marketcap = yf.Ticker(ticker).info.get("marketCap")
        return {
            "TTM_Growth": ttm_growth,
            "Forward_Growth": forward_growth,
        }, marketcap
    except Exception as e:
        print(f"Error fetching market cap for {ticker}: {e}")
        return {
            "TTM_Growth": ttm_growth,
            "Forward_Growth": forward_growth,
        }, None

# DO NOT CHANGE THIS NAME — MINI MAIN
def generate_all_summaries():
    from ticker_manager import read_tickers
    tickers = read_tickers("tickers.csv")
    for ticker in tickers:
        prepare_data_for_display(ticker, treasury_yield=0.045)

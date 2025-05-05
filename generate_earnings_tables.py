# generate_earnings_tables_to_db.py

import os
import sqlite3
import logging
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from ticker_manager import read_tickers, modify_tickers

# Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
OUTPUT_DIR = 'charts'
os.makedirs(OUTPUT_DIR, exist_ok=True)
yf.set_tz_cache_location(os.path.join(OUTPUT_DIR, 'tz_cache'))

DB_PATH = os.path.join(OUTPUT_DIR, 'earnings.db')
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Create tables if they don't exist
cursor.execute('''
CREATE TABLE IF NOT EXISTS earnings_past (
    ticker TEXT,
    earnings_date TEXT,
    eps_estimate TEXT,
    reported_eps TEXT,
    surprise_percent REAL,
    timestamp TEXT,
    PRIMARY KEY (ticker, earnings_date)
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS earnings_upcoming (
    ticker TEXT,
    earnings_date TEXT,
    timestamp TEXT,
    PRIMARY KEY (ticker, earnings_date)
)
''')

# Time references
today = pd.to_datetime(datetime.now().date())
seven_days_ago = today - pd.Timedelta(days=7)
tickers = modify_tickers(read_tickers('tickers.csv'), is_remote=True)

# Collect data
for ticker in tickers:
    logging.info(f"Processing {ticker}")
    try:
        stock = yf.Ticker(ticker)
        df = stock.get_earnings_dates(limit=30)
        if df is None or df.empty:
            continue
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()

        # Past earnings
        recent = df.loc[(df.index >= seven_days_ago) & (df.index <= today)]
        for edate, row in recent.iterrows():
            surprise = pd.to_numeric(row.get('Surprise(%)'), errors='coerce')
            eps_est = row.get('EPS Estimate')
            rpt_eps = row.get('Reported EPS')
            cursor.execute('''
                INSERT OR REPLACE INTO earnings_past (ticker, earnings_date, eps_estimate, reported_eps, surprise_percent, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                ticker,
                edate.date().isoformat(),
                f"{eps_est:.2f}" if pd.notna(eps_est) else None,
                f"{rpt_eps:.2f}" if pd.notna(rpt_eps) else None,
                surprise if pd.notna(surprise) else None,
                datetime.utcnow().isoformat()
            ))

        # Upcoming earnings
        future = df.loc[df.index > today]
        for fdate in future.index:
            cursor.execute('''
                INSERT OR REPLACE INTO earnings_upcoming (ticker, earnings_date, timestamp)
                VALUES (?, ?, ?)
            ''', (
                ticker,
                fdate.date().isoformat(),
                datetime.utcnow().isoformat()
            ))

    except Exception as e:
        logging.error(f"Error processing {ticker}: {e}")

# Commit all changes
conn.commit()
conn.close()
logging.info("Data collection complete and saved to database.")

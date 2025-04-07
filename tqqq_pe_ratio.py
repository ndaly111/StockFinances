#!/usr/bin/env python3
"""
Retrieves TQQQ's trailing P/E ratio using yfinance, printing both
the ratio (if available) and current UTC time.
"""

import yfinance as yf
from datetime import datetime

def get_pe_ratio(ticker_symbol):
    ticker_data = yf.Ticker(ticker_symbol)
    info = ticker_data.info
    pe_ratio = info.get('trailingPE', None)
    return pe_ratio

if __name__ == "__main__":
    symbol = "TQQQ"
    pe = get_pe_ratio(symbol)
    now_utc = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    if pe is None:
        print(f"{now_utc}\nNo trailing P/E ratio data available for {symbol}.")
    else:
        print(f"{now_utc}\nThe trailing P/E ratio for {symbol} is: {pe}")
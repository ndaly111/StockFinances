import requests
from bs4 import BeautifulSoup
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import os
import yfinance as yf
import csv
import numpy as np   # <-- already imported earlier but kept here for completeness


# --------------------------------------------------------------------------- #
#  NEW: safe price helper                                                     #
# --------------------------------------------------------------------------- #
def get_current_price(ticker_obj: yf.Ticker):
    """
    Safely obtain a current share price.
    Order of attempts:
        1) ticker.info['currentPrice']
        2) ticker.fast_info['lastPrice']
        3) last daily close from ticker.history('1d')
    Returns float or None.
    """
    price = ticker_obj.info.get("currentPrice")
    if price is None:
        # fast_info is much faster and often populated
        try:
            price = ticker_obj.fast_info.get("lastPrice")
        except Exception:
            price = None
    if price is None:
        try:
            hist = ticker_obj.history(period="1d")
            if not hist.empty and "Close" in hist.columns:
                price = float(hist["Close"].iloc[-1])
        except Exception:
            price = None
    return price


# --------------------------------------------------------------------------- #
#  (unchanged) log_valuation_data …                                           #
# --------------------------------------------------------------------------- #
# … keep the entire definition exactly as in your current script …


def finviz_five_yr(ticker, cursor):
    # … unchanged …
    pass  # ⇠ just a marker; keep your real function body here


# --------------------------------------------------------------------------- #
#  fetch_financial_valuation_data – NOW uses safe price helper               #
# --------------------------------------------------------------------------- #
def fetch_financial_valuation_data(ticker, db_path):
    print(f"Fetching financial valuation data for: {ticker}")

    stock = yf.Ticker(ticker)
    current_price = get_current_price(stock)   # <-- single-line change

    with sqlite3.connect(db_path) as conn:
        # (all SQL exactly as before) …
        # …
        combined_data = pd.concat([ttm_data, forecast_data]).reset_index(drop=True)
        return combined_data, growth_value, current_price, forecast_data


# --------------------------------------------------------------------------- #
#  calculate_valuations – guard if price is None                             #
# --------------------------------------------------------------------------- #
def calculate_valuations(combined_data, growth_values, treasury_yield,
                         current_price, marketcap):

    treasury_yield = float(treasury_yield) / 100
    # … growth-rate math unchanged …

    # Only compute RevPS when we actually have a price
    if current_price is not None:
        combined_data["Revenue_Per_Share"] = (
            combined_data["Revenue"] / marketcap
        ) * current_price
    else:
        combined_data["Revenue_Per_Share"] = np.nan   # safe placeholder

    # … rest of the function unchanged …


# --------------------------------------------------------------------------- #
#  fetch_stock_data – NOW uses safe price helper                             #
# --------------------------------------------------------------------------- #
def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)
    current_price = get_current_price(stock)         # <-- change
    forward_eps = stock.info.get('forwardEps')
    pe_ratio = stock.info.get('trailingPE', None)
    price_to_sales = stock.info.get('priceToSalesTrailing12Months', None)
    forward_pe_ratio = current_price / forward_eps if (forward_eps and current_price) else None
    return current_price, pe_ratio, price_to_sales, forward_pe_ratio


# --------------------------------------------------------------------------- #
#  generate_valuation_tables – unchanged                                      #
# --------------------------------------------------------------------------- #
# … keep as is …


# --------------------------------------------------------------------------- #
#  process_update_growth_csv – unchanged                                      #
# --------------------------------------------------------------------------- #
# … keep as is …


# --------------------------------------------------------------------------- #
#  valuation_update – extra guard before heavy work                          #
# --------------------------------------------------------------------------- #
def valuation_update(ticker, cursor, treasury_yield, marketcap, dashboard_data):
    db_path = "Stock Data.db"
    finviz_five_yr(ticker, cursor)

    combined_data, growth_values, current_price, forecast_data = \
        fetch_financial_valuation_data(ticker, db_path)

    if current_price is None:
        print(f"[{ticker}] Current price unavailable – skipping valuation.")
        return

    # rest of the function body is identical to your current version
    # -------------------------------------------------------------------- #
    # copy everything below unchanged, starting with the forecast_data.empty
    # check and down to the dashboard_data.append(...)
    # -------------------------------------------------------------------- #

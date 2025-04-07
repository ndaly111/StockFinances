#!/usr/bin/env python3
"""
Module: index_growth_table.py

This module fetches the trailing P/E ratio for SPY and QQQ,
automatically retrieves the current 10-year treasury yield,
calculates their implied growth using a custom formula,
and generates an HTML table displaying these metrics.

Functions:
    - fetch_trailing_pe(ticker): Returns the trailing P/E ratio for a given ticker.
    - fetch_treasury_yield(): Retrieves the current 10-year treasury yield from Yahoo Finance.
    - compute_implied_growth(pe_ratio, treasury_yield): Calculates the implied growth.
    - create_table_row(ticker, pe_ratio, implied_growth): Returns an HTML row for the ticker.
    - index_growth(treasury_yield=None): Returns the full HTML table for SPY and QQQ.
"""

import yfinance as yf

def fetch_trailing_pe(ticker):
    """
    Fetch the trailing P/E ratio for a given ticker using yfinance.
    
    Returns:
        The trailing P/E ratio as a float, or None if unavailable.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}
        pe_ratio = info.get('trailingPE', None)
        return pe_ratio
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

def fetch_treasury_yield():
    """
    Fetch the current 10-year treasury yield using yfinance.
    Uses the ticker '^TNX', which is quoted as a percentage.
    
    Returns:
        The treasury yield as a float (e.g. 3.5 for 3.5%), or None if unavailable.
    """
    try:
        treasury_ticker = yf.Ticker("^TNX")
        info = treasury_ticker.info or {}
        # Yahoo Finance quotes ^TNX as the yield percentage.
        yield_value = info.get("regularMarketPrice", None)
        if yield_value is None:
            print("Warning: Treasury yield not available; using fallback value.")
        return yield_value
    except Exception as e:
        print(f"Error fetching treasury yield: {e}")
        return None

def compute_implied_growth(pe_ratio, treasury_yield):
    """
    Calculate the implied growth based on the trailing P/E ratio and treasury yield.
    
    The treasury_yield is expected as a percentage (e.g., 3.5 for 3.5%).
    The formula used is:
    
        implied_growth = ((pe_ratio / 10) ** (1/10)) + (treasury_yield/100) - 1
    
    Returns:
        The implied growth as a float, or None if inputs are invalid.
    """
    if pe_ratio is None or treasury_yield is None:
        return None
    try:
        treasury_yield_decimal = float(treasury_yield) / 100
        implied_growth = ((pe_ratio / 10) ** (1/10)) + treasury_yield_decimal - 1
        return implied_growth
    except Exception as e:
        print(f"Error calculating implied growth for pe_ratio {pe_ratio} and treasury_yield {treasury_yield}: {e}")
        return None

def create_table_row(ticker, pe_ratio, implied_growth):
    """
    Generate an HTML table row for a given ticker, its P/E ratio, and its implied growth.
    
    Returns:
        A string containing the HTML table row.
    """
    pe_str = f"{pe_ratio:.1f}" if pe_ratio is not None else "N/A"
    growth_str = f"{implied_growth * 100:.1f}%" if implied_growth is not None else "N/A"
    row = f"<tr><td>{ticker}</td><td>{pe_str}</td><td>{growth_str}</td></tr>"
    return row

def index_growth(treasury_yield=None):
    """
    Main function that generates an HTML table containing the trailing P/E ratio
    and implied growth for SPY and QQQ.
    
    If no treasury yield is provided, it fetches the current 10-year treasury yield.
    
    Returns:
        A string containing the full HTML table.
    """
    # If treasury_yield is not provided, fetch it.
    if treasury_yield is None:
        fetched_yield = fetch_treasury_yield()
        # If fetching failed, you can choose a fallback default value.
        treasury_yield = str(fetched_yield) if fetched_yield is not None else "3.5"
    else:
        treasury_yield = str(treasury_yield)
    
    tickers = ["SPY", "QQQ"]
    rows = []
    # Add header row
    header = "<tr><th>Ticker</th><th>P/E Ratio</th><th>Implied Growth</th></tr>"
    rows.append(header)
    
    for ticker in tickers:
        pe = fetch_trailing_pe(ticker)
        growth = compute_implied_growth(pe, treasury_yield)
        row = create_table_row(ticker, pe, growth)
        rows.append(row)
    
    table_html = (
        "<table border='1' cellspacing='0' cellpadding='4'>"
        + "".join(rows) +
        "</table>"
    )
    return table_html

if __name__ == "__main__":
    # Example usage: generate the table and write it to a file.
    table = index_growth()
    output_file = "charts/spy_qqq_growth.html"
    with open(output_file, "w") as f:
        f.write(table)
    print(f"HTML table for SPY & QQQ growth metrics written to {output_file}")

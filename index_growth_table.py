#!/usr/bin/env python3
"""
Module: index_growth_table.py

This module fetches the trailing P/E ratio for SPY and QQQ,
automatically retrieves the current 10-year treasury yield if none is provided,
calculates their implied growth using a custom formula, and
generates an HTML table displaying these metrics.

Functions:
    - fetch_trailing_pe(ticker): Returns the trailing P/E ratio for a given ticker.
    - fetch_treasury_yield(): Retrieves the current 10-year treasury yield from Yahoo Finance.
    - compute_implied_growth(pe_ratio, treasury_yield): Calculates the implied growth.
    - create_table_row(ticker, pe_ratio, implied_growth): Returns an HTML row for the ticker.
    - index_growth(treasury_yield=None): Returns the full HTML table for SPY and QQQ.
"""

import yfinance as yf


def fetch_trailing_pe(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}
        pe_ratio = info.get('trailingPE', None)
        return pe_ratio
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None


def fetch_treasury_yield():
    try:
        treasury_ticker = yf.Ticker("^TNX")
        info = treasury_ticker.info or {}
        # Yahoo Finance quotes ^TNX as the yield percentage.
        yield_value = info.get("regularMarketPrice", None)
        return yield_value
    except Exception as e:
        print(f"Error fetching treasury yield: {e}")
        return None


def compute_implied_growth(pe_ratio, treasury_yield):
    if pe_ratio is None or treasury_yield is None:
        return None
    try:
        # treasury_yield is expected as a decimal (e.g. 0.0388 for 3.88%)
        implied_growth = ((pe_ratio / 10) ** (1 / 10)) + treasury_yield - 1
        return implied_growth
    except Exception as e:
        print(f"Error calculating implied growth: {e}")
        return None


def create_table_row(ticker, pe_ratio, implied_growth):
    pe_str = f"{pe_ratio:.1f}" if pe_ratio is not None else "N/A"
    growth_str = f"{implied_growth * 100:.1f}%" if implied_growth is not None else "N/A"
    return f"<tr><td>{ticker}</td><td>{pe_str}</td><td>{growth_str}</td></tr>"


def index_growth(treasury_yield=None):
    """
    If a treasury yield is provided, trust and clean it.
    Otherwise, fetch the treasury yield and fallback to 3.5% if necessary.
    The final treasury yield is converted to a decimal (e.g., 3.88% -> 0.0388).
    """
    if treasury_yield is not None and treasury_yield not in ["N/A", "-", ""]:
        treasury_yield = float(str(treasury_yield).replace('%', '').strip()) / 100
    else:
        treasury_yield_raw = fetch_treasury_yield()
        if treasury_yield_raw and str(treasury_yield_raw).strip() not in ["N/A", "-", ""]:
            treasury_yield = float(str(treasury_yield_raw).replace('%', '').strip()) / 100
        else:
            treasury_yield = 0.035  # default fallback for 3.5%
    
    tickers = ["SPY", "QQQ"]
    rows = ["<tr><th>Ticker</th><th>P/E Ratio</th><th>Implied Growth</th></tr>"]

    for ticker in tickers:
        pe = fetch_trailing_pe(ticker)
        growth = compute_implied_growth(pe, treasury_yield)
        rows.append(create_table_row(ticker, pe, growth))

    table_html = "<table border='1' cellspacing='0' cellpadding='4'>" + "".join(rows) + "</table>"
    return table_html


if __name__ == "__main__":
    table = index_growth()
    import os
    os.makedirs("charts", exist_ok=True)
    output_file = os.path.join("charts", "spy_qqq_growth.html")
    with open(output_file, "w") as f:
        f.write(table)
    print(f"HTML table for SPY & QQQ growth metrics written to {output_file}")

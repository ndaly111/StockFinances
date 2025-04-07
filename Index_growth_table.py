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
        yield_value = info.get("regularMarketPrice", None)
        return yield_value
    except Exception as e:
        print(f"Error fetching treasury yield: {e}")
        return None


def compute_implied_growth(pe_ratio, treasury_yield):
    if pe_ratio is None or treasury_yield is None:
        return None
    try:
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
    if treasury_yield is None or treasury_yield in ["N/A", "-"]:
        treasury_yield_raw = fetch_treasury_yield()
        treasury_yield = float(treasury_yield_raw) / 100 if treasury_yield_raw else 0.035
    else:
        treasury_yield = float(str(treasury_yield).replace('%', '')) / 100

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
    output_dir = "charts"
    import os
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "spy_qqq_growth.html")
    with open(output_file, "w") as f:
        f.write(table)
    print(f"HTML table for SPY & QQQ growth metrics written to {output_file}")

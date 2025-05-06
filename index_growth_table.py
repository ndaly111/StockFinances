#!/usr/bin/env python3
"""
Module: index_growth_table.py

Fetches trailing P/E ratios for SPY and QQQ, caches them in stock data.db,
retrieves the 10‑year Treasury yield, computes implied growth,
and writes an HTML table to charts/spy_qqq_growth.html.
"""

import os
import time
import sqlite3
import yfinance as yf
from datetime import datetime

# Ensure output folder exists
os.makedirs("charts", exist_ok=True)

# Path to your database in the project root
DB_PATH = "stock data.db"

# Create the cache table if missing
with sqlite3.connect(DB_PATH) as conn:
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS pe_cache (
            ticker    TEXT PRIMARY KEY,
            pe        REAL,
            timestamp TEXT
        )
    ''')
    conn.commit()


def fetch_trailing_pe(ticker, retries=3, delay=1):
    now = datetime.utcnow()
    # 1) Check cache
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT pe, timestamp FROM pe_cache WHERE ticker = ?", (ticker,))
        row = c.fetchone()
        if row:
            pe_cached, ts = row
            try:
                if datetime.fromisoformat(ts).date() == now.date():
                    print(f"[Cache] {ticker} P/E = {pe_cached:.2f}")
                    return pe_cached
            except Exception:
                pass  # fall through to re-fetch

    # 2) Fetch from yfinance
    pe = None
    for attempt in range(1, retries + 1):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info or {}
            pe = info.get("trailingPE")
            if pe:
                break

            # Fallback: price / trailingEps
            price = info.get("previousClose") or info.get("regularMarketPrice")
            eps   = info.get("trailingEps")
            if price is None:
                price = getattr(stock, "fast_info", {}).get("lastPrice")
            if eps is None:
                try:
                    eps = stock.quarterly_earnings["Earnings"].iloc[-1]
                except Exception:
                    eps = None

            if price and eps:
                pe = price / eps
                print(f"[Fallback] {ticker} P/E = {price:.2f}/{eps:.2f} = {pe:.2f}")
                break

            print(f"[Attempt {attempt}] No P/E for {ticker}; keys: {list(info.keys())}")
        except Exception as e:
            print(f"[Error][Attempt {attempt}] fetching P/E for {ticker}: {e}")

        time.sleep(delay)

    if pe is None:
        print(f"[Failed] Could not find P/E for {ticker} after {retries} attempts.")
        return None

    # 3) Cache the fresh value
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute('''
            INSERT OR REPLACE INTO pe_cache (ticker, pe, timestamp)
            VALUES (?, ?, ?)
        ''', (ticker, float(pe), now.isoformat()))
        conn.commit()
        print(f"[Cached] {ticker} P/E = {pe:.2f} into {DB_PATH}")

    return pe


def fetch_treasury_yield():
    try:
        info = yf.Ticker("^TNX").info or {}
        y = info.get("regularMarketPrice")
        return float(y) / 100 if y is not None else None
    except Exception as e:
        print(f"[Error] fetching treasury yield: {e}")
        return None


def compute_implied_growth(pe_ratio, treasury_yield):
    if pe_ratio is None or treasury_yield is None:
        return None
    try:
        return ((pe_ratio / 10) ** (1 / 10)) + treasury_yield - 1
    except Exception as e:
        print(f"[Error] computing implied growth: {e}")
        return None


def create_table_row(ticker, pe_ratio, implied_growth):
    pe_str     = f"{pe_ratio:.1f}" if pe_ratio is not None else "N/A"
    growth_str = f"{implied_growth * 100:.1f}%" if implied_growth is not None else "N/A"
    return f"<tr><td>{ticker}</td><td>{pe_str}</td><td>{growth_str}</td></tr>"


def index_growth(treasury_yield=None):
    if treasury_yield:
        treasury_yield = float(str(treasury_yield).strip('%')) / 100
    else:
        treasury_yield = fetch_treasury_yield() or 0.035

    tickers = ["SPY", "QQQ"]
    rows    = ["<tr><th>Ticker</th><th>P/E Ratio</th><th>Implied Growth</th></tr>"]

    for tk in tickers:
        pe     = fetch_trailing_pe(tk)
        growth = compute_implied_growth(pe, treasury_yield)
        rows.append(create_table_row(tk, pe, growth))

    return "<table border='1' cellspacing='0' cellpadding='4'>" + "".join(rows) + "</table>"


if __name__ == "__main__":
    html = index_growth()
    path = os.path.join("charts", "spy_qqq_growth.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote SPY/QQQ growth table to {path}")
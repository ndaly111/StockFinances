#!/usr/bin/env python3
"""Backfill SPY P/E and implied growth from Multpl.com.

This script scrapes monthly S&P 500 P/E ratio data from Multpl.com,
which is regularly updated (typically through the current month).

Data source: https://www.multpl.com/s-p-500-pe-ratio/table/by-month

Run this script to get recent monthly P/E data:
    python scripts/backfill_spy_from_multpl.py
"""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

# Configuration
DB_PATH = "Stock Data.db"
MULTPL_PE_URL = "https://www.multpl.com/s-p-500-pe-ratio/table/by-month"
FALLBACK_YIELD = 0.045


def scrape_multpl_pe() -> pd.DataFrame:
    """Scrape P/E ratio data from Multpl.com."""
    print(f"Scraping data from {MULTPL_PE_URL}...")

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }

    resp = requests.get(MULTPL_PE_URL, headers=headers, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.content, 'html.parser')

    # Find the data table
    table = soup.find('table', {'id': 'datatable'})
    if not table:
        # Try alternative selector
        table = soup.find('table')

    if not table:
        raise ValueError("Could not find data table on page")

    rows = []
    for tr in table.find_all('tr')[1:]:  # Skip header row
        cells = tr.find_all('td')
        if len(cells) >= 2:
            date_text = cells[0].get_text(strip=True)
            pe_text = cells[1].get_text(strip=True)

            try:
                # Parse date (format: "Jan 1, 2025" or similar)
                date = pd.to_datetime(date_text)

                # Parse P/E (remove any non-numeric chars except decimal)
                pe_clean = re.sub(r'[^\d.]', '', pe_text)
                pe = float(pe_clean) if pe_clean else None

                if pe and pe > 0 and pe < 200:
                    rows.append({
                        'Date': date,
                        'PE': pe
                    })
            except (ValueError, TypeError):
                continue

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError("No valid data scraped from page")

    df = df.sort_values('Date').drop_duplicates(subset=['Date'], keep='last')
    print(f"Scraped {len(df)} months of P/E data: {df['Date'].min()} to {df['Date'].max()}")

    return df


def get_treasury_yields(conn: sqlite3.Connection) -> pd.DataFrame:
    """Get treasury yield data from the database."""
    try:
        df = pd.read_sql_query(
            """
            SELECT date AS Date, value / 100.0 AS Yield
            FROM economic_data
            WHERE indicator = 'DGS10' AND value IS NOT NULL
            ORDER BY date
            """,
            conn
        )
        df['Date'] = pd.to_datetime(df['Date'])
        return df.set_index('Date')['Yield']
    except Exception as e:
        print(f"Warning: Could not load treasury yields: {e}")
        return pd.Series(dtype=float)


def calculate_implied_growth(pe: float, treasury_yield: float) -> float | None:
    """Calculate implied growth from P/E and treasury yield."""
    if pe is None or pe <= 0 or treasury_yield is None:
        return None
    try:
        return (pe / 10.0) ** 0.1 + treasury_yield - 1.0
    except (OverflowError, ValueError):
        return None


def backfill_to_database(data: pd.DataFrame, yields: pd.Series, conn: sqlite3.Connection):
    """Write the backfill data to the database."""

    cur = conn.cursor()

    # Ensure tables exist
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS Index_PE_History (
            Date TEXT, Ticker TEXT, PE_Type TEXT, PE_Ratio REAL,
            PRIMARY KEY (Date, Ticker, PE_Type)
        );
        CREATE TABLE IF NOT EXISTS Index_Growth_History (
            Date TEXT, Ticker TEXT, Growth_Type TEXT, Implied_Growth REAL,
            PRIMARY KEY (Date, Ticker, Growth_Type)
        );
    """)

    pe_rows = []
    growth_rows = []

    for _, row in data.iterrows():
        date = row['Date']
        date_str = date.strftime('%Y-%m-%d')
        pe = row['PE']

        # Get closest yield
        if not yields.empty:
            # Find yield on or before this date
            prior_yields = yields[yields.index <= date]
            if not prior_yields.empty:
                treasury_yield = prior_yields.iloc[-1]
            else:
                treasury_yield = FALLBACK_YIELD
        else:
            treasury_yield = FALLBACK_YIELD

        implied_growth = calculate_implied_growth(pe, treasury_yield)

        pe_rows.append((date_str, 'SPY', 'TTM', float(pe)))

        if implied_growth is not None:
            growth_rows.append((date_str, 'SPY', 'TTM', float(implied_growth)))

    # Insert P/E data (only insert if we don't have daily data for that month)
    # Use INSERT OR IGNORE to not overwrite existing daily data
    cur.executemany(
        """INSERT OR IGNORE INTO Index_PE_History (Date, Ticker, PE_Type, PE_Ratio)
           VALUES (?, ?, ?, ?)""",
        pe_rows
    )

    cur.executemany(
        """INSERT OR IGNORE INTO Index_Growth_History (Date, Ticker, Growth_Type, Implied_Growth)
           VALUES (?, ?, ?, ?)""",
        growth_rows
    )

    conn.commit()
    print(f"Inserted up to {len(pe_rows)} P/E rows and {len(growth_rows)} growth rows (ignoring duplicates)")


def main():
    """Main entry point."""
    print("=" * 60)
    print("SPY Monthly P/E Backfill from Multpl.com")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    db_path = Path(DB_PATH)
    if not db_path.exists():
        print(f"Error: Database not found at {DB_PATH}")
        return 1

    try:
        # Scrape data
        data = scrape_multpl_pe()

        if data.empty:
            print("Error: No data scraped")
            return 1

        # Connect to database
        conn = sqlite3.connect(DB_PATH)

        # Get treasury yields
        yields = get_treasury_yields(conn)
        print(f"Loaded {len(yields)} treasury yield data points")

        # Backfill to database
        backfill_to_database(data, yields, conn)

        conn.close()

        print("\n" + "=" * 60)
        print("Backfill complete!")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

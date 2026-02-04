#!/usr/bin/env python3
"""Backfill SPY P/E and implied growth from NYU Stern (Damodaran) data.

This script downloads the S&P 500 earnings spreadsheet from NYU Stern,
which is updated annually and contains historical earnings data going back
to 1960. It calculates P/E ratios and implied growth rates to populate
the Index_PE_History and Index_Growth_History tables.

Data source: https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/spearn.html
Excel file: https://www.stern.nyu.edu/~adamodar/pc/datasets/spearn.xls

Run this script periodically (e.g., quarterly) to refresh historical data:
    python scripts/backfill_spy_from_stern.py
"""

from __future__ import annotations

import io
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

# Configuration
DB_PATH = "Stock Data.db"
STERN_URL = "https://www.stern.nyu.edu/~adamodar/pc/datasets/spearn.xls"
FALLBACK_YIELD = 0.045  # 4.5% default if no yield data available


def download_stern_data() -> pd.DataFrame:
    """Download and parse the NYU Stern S&P earnings spreadsheet."""
    print(f"Downloading data from {STERN_URL}...")

    resp = requests.get(STERN_URL, timeout=60)
    resp.raise_for_status()

    # Read the Excel file - the main data is typically in the first sheet
    # The spreadsheet has earnings by year
    df = pd.read_excel(
        io.BytesIO(resp.content),
        sheet_name=0,
        header=None,
        engine='xlrd'  # For .xls files
    )

    print(f"Downloaded {len(df)} rows")
    return df


def parse_stern_data(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the Stern spreadsheet into a clean DataFrame with Year, Price, Earnings, P/E."""

    # The Damodaran spreadsheet format varies, but typically has:
    # - Year in first column
    # - S&P 500 level/price
    # - Earnings
    # We need to find the header row and data

    # Look for a row containing "Year" to find the header
    header_idx = None
    for idx, row in df.iterrows():
        row_str = ' '.join(str(v).lower() for v in row.values if pd.notna(v))
        if 'year' in row_str and ('earnings' in row_str or 'eps' in row_str):
            header_idx = idx
            break

    if header_idx is None:
        # Try alternative: assume first row with numeric year is start of data
        for idx, row in df.iterrows():
            first_val = row.iloc[0]
            if pd.notna(first_val):
                try:
                    year = int(float(first_val))
                    if 1900 <= year <= 2100:
                        header_idx = idx - 1 if idx > 0 else 0
                        break
                except (ValueError, TypeError):
                    continue

    if header_idx is None:
        raise ValueError("Could not find header row in spreadsheet")

    # Set header and get data
    df.columns = df.iloc[header_idx]
    df = df.iloc[header_idx + 1:].reset_index(drop=True)

    # Normalize column names
    df.columns = [str(c).strip().lower() if pd.notna(c) else f'col_{i}'
                  for i, c in enumerate(df.columns)]

    # Find the relevant columns
    year_col = next((c for c in df.columns if 'year' in c), df.columns[0])

    # Look for earnings column
    earnings_col = None
    for c in df.columns:
        c_lower = str(c).lower()
        if 'earnings' in c_lower or 'eps' in c_lower:
            earnings_col = c
            break

    # Look for price/level column
    price_col = None
    for c in df.columns:
        c_lower = str(c).lower()
        if any(term in c_lower for term in ['price', 'level', 's&p', 'sp500', 'index']):
            if 'earnings' not in c_lower:
                price_col = c
                break

    if earnings_col is None:
        # Try second column as earnings
        earnings_col = df.columns[1] if len(df.columns) > 1 else None

    if price_col is None:
        # Try to find a numeric column that looks like prices (hundreds to thousands)
        for c in df.columns:
            if c == year_col or c == earnings_col:
                continue
            try:
                vals = pd.to_numeric(df[c], errors='coerce').dropna()
                if len(vals) > 0 and 100 < vals.mean() < 10000:
                    price_col = c
                    break
            except:
                continue

    print(f"Using columns: Year={year_col}, Price={price_col}, Earnings={earnings_col}")

    # Build clean dataframe
    result = pd.DataFrame()
    result['Year'] = pd.to_numeric(df[year_col], errors='coerce')

    if price_col:
        result['Price'] = pd.to_numeric(df[price_col], errors='coerce')

    if earnings_col:
        result['Earnings'] = pd.to_numeric(df[earnings_col], errors='coerce')

    # Filter to valid years
    result = result[result['Year'].notna() & (result['Year'] >= 1950) & (result['Year'] <= 2100)]
    result['Year'] = result['Year'].astype(int)

    # Calculate P/E
    if 'Price' in result.columns and 'Earnings' in result.columns:
        result['PE'] = result['Price'] / result['Earnings']
        # Filter out invalid P/E values
        result.loc[result['PE'] <= 0, 'PE'] = None
        result.loc[result['PE'] > 200, 'PE'] = None

    result = result.dropna(subset=['Year'])
    print(f"Parsed {len(result)} years of data: {result['Year'].min()} - {result['Year'].max()}")

    return result


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
        df['Year'] = df['Date'].dt.year
        # Get average yield per year
        yearly = df.groupby('Year')['Yield'].mean().reset_index()
        return yearly
    except Exception as e:
        print(f"Warning: Could not load treasury yields: {e}")
        return pd.DataFrame(columns=['Year', 'Yield'])


def calculate_implied_growth(pe: float, treasury_yield: float) -> float | None:
    """Calculate implied growth from P/E and treasury yield."""
    if pe is None or pe <= 0 or treasury_yield is None:
        return None
    try:
        # Formula: growth = (PE / 10) ** 0.1 + yield - 1
        return (pe / 10.0) ** 0.1 + treasury_yield - 1.0
    except (OverflowError, ValueError):
        return None


def backfill_to_database(data: pd.DataFrame, yields: pd.DataFrame, conn: sqlite3.Connection):
    """Write the backfill data to the database."""

    # Merge yields
    if not yields.empty:
        data = data.merge(yields, on='Year', how='left')

    # Fill missing yields with fallback
    if 'Yield' not in data.columns:
        data['Yield'] = FALLBACK_YIELD
    data['Yield'] = data['Yield'].fillna(FALLBACK_YIELD)

    # Calculate implied growth
    data['Implied_Growth'] = data.apply(
        lambda r: calculate_implied_growth(r.get('PE'), r.get('Yield')),
        axis=1
    )

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

    # Insert data for each year (using July 1 as the representative date)
    pe_rows = []
    growth_rows = []

    for _, row in data.iterrows():
        year = int(row['Year'])
        date_str = f"{year}-07-01"  # Mid-year as representative

        if pd.notna(row.get('PE')):
            pe_rows.append((date_str, 'SPY', 'TTM_ANNUAL', float(row['PE'])))

        if pd.notna(row.get('Implied_Growth')):
            growth_rows.append((date_str, 'SPY', 'TTM_ANNUAL', float(row['Implied_Growth'])))

    # Insert P/E data
    cur.executemany(
        """INSERT OR REPLACE INTO Index_PE_History (Date, Ticker, PE_Type, PE_Ratio)
           VALUES (?, ?, ?, ?)""",
        pe_rows
    )

    # Insert growth data
    cur.executemany(
        """INSERT OR REPLACE INTO Index_Growth_History (Date, Ticker, Growth_Type, Implied_Growth)
           VALUES (?, ?, ?, ?)""",
        growth_rows
    )

    conn.commit()
    print(f"Inserted {len(pe_rows)} P/E rows and {len(growth_rows)} growth rows")


def main():
    """Main entry point."""
    print("=" * 60)
    print("SPY Historical Data Backfill from NYU Stern")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    # Check if database exists
    db_path = Path(DB_PATH)
    if not db_path.exists():
        print(f"Error: Database not found at {DB_PATH}")
        return 1

    try:
        # Download and parse data
        raw_df = download_stern_data()
        parsed_df = parse_stern_data(raw_df)

        if parsed_df.empty:
            print("Error: No data parsed from spreadsheet")
            return 1

        # Connect to database
        conn = sqlite3.connect(DB_PATH)

        # Get treasury yields
        yields = get_treasury_yields(conn)
        print(f"Loaded {len(yields)} years of treasury yield data")

        # Backfill to database
        backfill_to_database(parsed_df, yields, conn)

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

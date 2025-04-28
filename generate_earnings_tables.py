# generate_earnings_tables_debug.py

import os
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from ticker_manager import read_tickers, modify_tickers

# Constants
TICKERS_FILE_PATH    = 'tickers.csv'
OUTPUT_DIR           = 'charts'
PAST_HTML_PATH       = os.path.join(OUTPUT_DIR, 'earnings_past.html')
UPCOMING_HTML_PATH   = os.path.join(OUTPUT_DIR, 'earnings_upcoming.html')

today               = datetime.now().date()
seven_days_ago      = today - timedelta(days=7)
three_days_from_now = today + timedelta(days=3)

# === Prepare output & tz-cache ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create a dummy tz cache file so yfinance doesn't try to stat(None)
dummy_tz_cache = os.path.join(OUTPUT_DIR, 'tz_cache.json')
# Ensure the file exists (empty is fine)
open(dummy_tz_cache, 'a').close()
yf.set_tz_cache_location(dummy_tz_cache)

# Load tickers
tickers = modify_tickers(read_tickers(TICKERS_FILE_PATH), is_remote=True)

past_rows     = []
upcoming_rows = []

print("\n=== Starting earnings data collection ===\n")

for ticker in tickers:
    print(f"\n--- Processing {ticker} ---")
    try:
        # Instantiate ticker and fetch calendar
        stock = yf.Ticker(ticker)
        cal   = stock.calendar

        # --- DEBUG: calendar structure & raw earnings date ---
        print("  calendar type:", type(cal))
        if isinstance(cal, dict):
            print("  calendar keys:", list(cal.keys()))
            print("  raw 'Earnings Date':", cal.get('Earnings Date'))
        else:
            print("  calendar (non-dict):\n", cal)

        # --- Past Earnings (unchanged) ---
        try:
            df = stock.earnings_dates
            if isinstance(df, pd.DataFrame):
                recent = df[(df.index.date >= seven_days_ago) & (df.index.date <= today)]
                if not recent.empty:
                    print("  Past earnings:", list(recent.index.date))
                else:
                    print("  No past earnings in last 7 days.")
                for date, row in recent.iterrows():
                    surprise      = row.get('Surprise(%)', None)
                    surprise_str  = f"{surprise:+.2f}%" if pd.notna(surprise) else "-"
                    css_class     = 'positive' if surprise > 0 else 'negative' if surprise < 0 else ''
                    surprise_html = f'<span class="{css_class}">{surprise_str}</span>' if css_class else surprise_str

                    eps_estimate     = f"{row['EPS Estimate']:.2f}" if pd.notna(row.get('EPS Estimate')) else "-"
                    reported_eps     = f"{row['Reported EPS']:.2f}" if pd.notna(row.get('Reported EPS')) else "-"
                    revenue_est      = row.get('Revenue Estimate')
                    revenue_str      = f"${revenue_est:,.0f}" if pd.notna(revenue_est) else "-"
                    reported_rev     = row.get('Reported Revenue')
                    reported_rev_str = f"${reported_rev:,.0f}" if pd.notna(reported_rev) else "-"

                    past_rows.append([
                        ticker,
                        date.date().isoformat(),
                        eps_estimate,
                        reported_eps,
                        surprise_html,
                        revenue_str,
                        reported_rev_str
                    ])
            else:
                print("  No past-earnings DataFrame available.")
        except Exception as e:
            print(f"  Error in past earnings block: {e}")

        # --- Upcoming Earnings (DICT HANDLING) ---
        try:
            if isinstance(cal, dict) and 'Earnings Date' in cal:
                ed_raw = cal['Earnings Date']
                if isinstance(ed_raw, list) and ed_raw:
                    earnings_date = ed_raw[0]
                else:
                    earnings_date = ed_raw  # single date or None

                # Convert pandas Timestamp if needed
                if isinstance(earnings_date, pd.Timestamp):
                    earnings_date = earnings_date.date()

                print("  Parsed earnings_date:", earnings_date)

                if isinstance(earnings_date, datetime):
                    earnings_date = earnings_date.date()

                if earnings_date and earnings_date >= today:
                    highlight = 'highlight-soon' if earnings_date <= three_days_from_now else ''
                    upcoming_rows.append((
                        earnings_date,
                        f'<tr class="{highlight}"><td>{ticker}</td><td>{earnings_date}</td></tr>'
                    ))
                    print(f"  → Upcoming earnings for {ticker} on {earnings_date}")
                else:
                    print(f"  → No valid future earnings date (got {earnings_date})")
            else:
                print("  No 'Earnings Date' key in calendar")
        except Exception as e:
            print(f"  Error in upcoming earnings block: {e}")

    except Exception as e:
        print(f"Failed to process {ticker}: {e}")

print("\n=== Finished collecting earnings data ===\n")

# --- Save Past Earnings Table ---
if past_rows:
    df_past = pd.DataFrame(past_rows, columns=[
        'Ticker', 'Earnings Date', 'EPS Estimate', 'Reported EPS',
        'Surprise', 'Revenue Estimate', 'Reported Revenue'
    ])
    df_past['Earnings Date'] = pd.to_datetime(df_past['Earnings Date'])
    df_past.sort_values('Earnings Date', ascending=False, inplace=True)

    note = f"<p>Showing earnings from {seven_days_ago} to {today}.</p>"
    html = df_past.to_html(escape=False, index=False, classes='center-table', border=0)
    with open(PAST_HTML_PATH, 'w', encoding='utf-8') as f:
        f.write(note + html)
    print("Past earnings table saved.")
else:
    with open(PAST_HTML_PATH, 'w', encoding='utf-8') as f:
        f.write("<p>No earnings in the past 7 days.</p>")
    print("No past earnings to save.")

# --- Save Upcoming Earnings Table ---
if upcoming_rows:
    upcoming_rows.sort(key=lambda x: x[0])
    tbl = (
        "<table class='center-table'>"
        "<thead><tr><th>Ticker</th><th>Upcoming Earnings Date</th></tr></thead>"
        "<tbody>"
        + "".join(row for _, row in upcoming_rows) +
        "</tbody></table>"
    )
    with open(UPCOMING_HTML_PATH, 'w', encoding='utf-8') as f:
        f.write(tbl)
    print("Upcoming earnings table saved.")
else:
    with open(UPCOMING_HTML_PATH, 'w', encoding='utf-8') as f:
        f.write("<p>No upcoming earnings scheduled.</p>")
    print("No upcoming earnings to save.")

# --- Summary ---
print("\n=== Earnings Summary ===")
print(f"Tickers processed: {len(tickers)}")
print(f"Past events found: {len(past_rows)}")
print(f"Upcoming events found: {len(upcoming_rows)}")
print("=== Done ===\n")
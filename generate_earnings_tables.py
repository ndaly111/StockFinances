# generate_earnings_tables.py

import os
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from ticker_manager import read_tickers, modify_tickers

# Constants
TICKERS_FILE_PATH = 'tickers.csv'
OUTPUT_DIR = 'charts'
PAST_HTML_PATH = os.path.join(OUTPUT_DIR, 'earnings_past.html')
UPCOMING_HTML_PATH = os.path.join(OUTPUT_DIR, 'earnings_upcoming.html')

today = datetime.now().date()
seven_days_ago = today - timedelta(days=7)
three_days_from_now = today + timedelta(days=3)

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load tickers
tickers = modify_tickers(read_tickers(TICKERS_FILE_PATH), is_remote=True)

past_rows = []
upcoming_rows = []

for ticker in tickers:
    try:
        stock = yf.Ticker(ticker)

        # Past Earnings
        try:
            df = stock.earnings_dates
            if isinstance(df, pd.DataFrame):
                recent = df[(df.index.date >= seven_days_ago) & (df.index.date <= today)]
                for date, row in recent.iterrows():
                    surprise = row.get('Surprise(%)', None)
                    surprise_str = f"{surprise:+.2f}%" if pd.notna(surprise) else "-"
                    css_class = 'positive' if surprise > 0 else 'negative' if surprise < 0 else ''
                    surprise_html = f'<span class="{css_class}">{surprise_str}</span>' if css_class else surprise_str

                    eps_estimate = f"{row['EPS Estimate']:.2f}" if pd.notna(row.get('EPS Estimate')) else "-"
                    reported_eps = f"{row['Reported EPS']:.2f}" if pd.notna(row.get('Reported EPS')) else "-"

                    revenue_est = row.get('Revenue Estimate')
                    revenue_estimate = f"${revenue_est:,.0f}" if pd.notna(revenue_est) else "-"

                    reported_rev = row.get('Reported Revenue')
                    reported_revenue = f"${reported_rev:,.0f}" if pd.notna(reported_rev) else "-"

                    past_rows.append([
                        ticker,
                        date.date().isoformat(),
                        eps_estimate,
                        reported_eps,
                        surprise_html,
                        revenue_estimate,
                        reported_revenue
                    ])
        except Exception:
            pass

        # Upcoming Earnings
        try:
            cal = stock.calendar
            if 'Earnings Date' in cal.index:
                earnings_date = pd.to_datetime(cal.loc['Earnings Date']).date()
                if earnings_date >= today:
                    highlight_class = 'highlight-soon' if earnings_date <= three_days_from_now else ''
                    upcoming_rows.append((earnings_date, f'<tr class="{highlight_class}"><td>{ticker}</td><td>{earnings_date}</td></tr>'))
        except Exception:
            pass

    except Exception as e:
        print(f"Failed to process {ticker}: {e}")

# Save Past Earnings Table (sorted by date descending)
if past_rows:
    df_past = pd.DataFrame(past_rows, columns=[
        'Ticker', 'Earnings Date', 'EPS Estimate', 'Reported EPS',
        'Surprise', 'Revenue Estimate', 'Reported Revenue'
    ])
    df_past['Earnings Date'] = pd.to_datetime(df_past['Earnings Date'])
    df_past.sort_values(by='Earnings Date', ascending=False, inplace=True)
    html_past = df_past.to_html(escape=False, index=False, classes='center-table', border=0)
    with open(PAST_HTML_PATH, 'w', encoding='utf-8') as f:
        f.write(html_past)
else:
    with open(PAST_HTML_PATH, 'w', encoding='utf-8') as f:
        f.write("<p>No earnings in the past 7 days.</p>")

# Save Upcoming Earnings Table (sorted by soonest date ascending)
if upcoming_rows:
    sorted_upcoming = sorted(upcoming_rows, key=lambda x: x[0])
    table_html = "<table class='center-table'><thead><tr><th>Ticker</th><th>Upcoming Earnings Date</th></tr></thead><tbody>"
    table_html += ''.join(row_html for _, row_html in sorted_upcoming)
    table_html += "</tbody></table>"
    with open(UPCOMING_HTML_PATH, 'w', encoding='utf-8') as f:
        f.write(table_html)
else:
    with open(UPCOMING_HTML_PATH, 'w', encoding='utf-8') as f:
        f.write("<p>No upcoming earnings scheduled.</p>")
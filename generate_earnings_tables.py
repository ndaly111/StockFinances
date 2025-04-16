# generate_earnings_tables.py

import os
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from ticker_manager import read_tickers, modify_tickers

# Paths
TICKERS_FILE_PATH = 'tickers.csv'
OUTPUT_DIR = 'charts'
PAST_HTML_PATH = os.path.join(OUTPUT_DIR, 'earnings_past.html')
UPCOMING_HTML_PATH = os.path.join(OUTPUT_DIR, 'earnings_upcoming.html')

# Setup
today = datetime.now().date()
seven_days_ago = today - timedelta(days=7)
three_days_from_now = today + timedelta(days=3)

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load tickers
tickers = modify_tickers(read_tickers(TICKERS_FILE_PATH), is_remote=True)

# Containers for table data
past_rows = []
upcoming_rows = []

for ticker in tickers:
    try:
        stock = yf.Ticker(ticker)

        # Past Earnings
        try:
            df = stock.earnings_dates
            recent = df[(df.index.date >= seven_days_ago) & (df.index.date <= today)]
            for date, row in recent.iterrows():
                surprise = row['Surprise(%)']
                surprise_str = f"{surprise:+.2f}%" if pd.notna(surprise) else "-"
                css_class = 'positive' if surprise > 0 else 'negative' if surprise < 0 else ''
                past_rows.append([
                    ticker,
                    date.date().isoformat(),
                    f"{row['EPS Estimate']:.2f}" if pd.notna(row['EPS Estimate']) else "-",
                    f"{row['Reported EPS']:.2f}" if pd.notna(row['Reported EPS']) else "-",
                    f'<td class="{css_class}">{surprise_str}</td>'
                ])
        except Exception:
            pass

        # Upcoming Earnings
        try:
            cal = stock.calendar
            if 'Earnings Date' in cal.index:
                date = cal.loc['Earnings Date'][0].date()
                if date >= today:
                    highlight_class = 'highlight-soon' if date <= three_days_from_now else ''
                    upcoming_rows.append([f'<tr class="{highlight_class}"><td>{ticker}</td><td>{date.isoformat()}</td></tr>'])
        except Exception:
            pass

    except Exception as e:
        print(f"Failed to process {ticker}: {e}")

# Create and save Past Earnings Table
if past_rows:
    df_past = pd.DataFrame(past_rows, columns=['Ticker', 'Earnings Date', 'EPS Estimate', 'Reported EPS', 'Surprise'])
    html_past = df_past.to_html(escape=False, index=False, classes='center-table', border=0)
    with open(PAST_HTML_PATH, 'w', encoding='utf-8') as f:
        f.write(html_past)
else:
    with open(PAST_HTML_PATH, 'w', encoding='utf-8') as f:
        f.write("<p>No earnings in the past 7 days.</p>")

# Create and save Upcoming Earnings Table
if upcoming_rows:
    table_html = "<table class='center-table'><thead><tr><th>Ticker</th><th>Upcoming Earnings Date</th></tr></thead><tbody>"
    table_html += ''.join([row[0] for row in upcoming_rows])
    table_html += "</tbody></table>"
    with open(UPCOMING_HTML_PATH, 'w', encoding='utf-8') as f:
        f.write(table_html)
else:
    with open(UPCOMING_HTML_PATH, 'w', encoding='utf-8') as f:
        f.write("<p>No upcoming earnings scheduled.</p>")

# generate_earnings_tables_upgraded_debug_v2.py

import os
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from ticker_manager import read_tickers, modify_tickers

# Setup output directory and tz cache
OUTPUT_DIR = 'charts'
os.makedirs(OUTPUT_DIR, exist_ok=True)
tz_cache = os.path.join(OUTPUT_DIR, 'tz_cache.json')
open(tz_cache, 'a').close()
yf.set_tz_cache_location(tz_cache)

# Constants
TICKERS_FILE_PATH = 'tickers.csv'
PAST_HTML_PATH = os.path.join(OUTPUT_DIR, 'earnings_past.html')
UPCOMING_HTML_PATH = os.path.join(OUTPUT_DIR, 'earnings_upcoming.html')

today = datetime.now().date()
seven_days_ago = today - timedelta(days=7)
three_days_from_now = today + timedelta(days=3)

tickers = modify_tickers(read_tickers(TICKERS_FILE_PATH), is_remote=True)

past_rows, upcoming_rows, reporting_today = [], [], []

print("\n=== STARTING COLLECTION ===\n")

for ticker in tickers:
    print(f"\n--- Processing {ticker} ---")
    try:
        stock = yf.Ticker(ticker)
        cal = stock.calendar
        print("Calendar contents:", cal)

        try:
            df = stock.earnings_dates
            if isinstance(df, pd.DataFrame):
                print(f"Pulled earnings_dates for {ticker}:")
                print(df.head())

                recent = df[(df.index.date >= seven_days_ago) & (df.index.date <= today)]
                for date, row in recent.iterrows():
                    if date.date() == today:
                        reporting_today.append(ticker)

                    surprise = row.get('Surprise(%)', None)
                    surprise_val = float(surprise) if pd.notna(surprise) else None
                    css_class = 'positive' if surprise_val and surprise_val > 0 else 'negative' if surprise_val and surprise_val < 0 else ''
                    surprise_str = f'<span class="{css_class}">{surprise_val:+.2f}%</span>' if surprise_val is not None else '-'

                    eps_est = f"{row['EPS Estimate']:.2f}" if pd.notna(row.get('EPS Estimate')) else "-"
                    rpt_eps = f"{row['Reported EPS']:.2f}" if pd.notna(row.get('Reported EPS')) else "-"

                    print(f"  Date: {date.date()}, EPS: {rpt_eps}, Surprise: {surprise_str}")

                    past_rows.append([
                        ticker,
                        date.date().isoformat(),
                        eps_est, rpt_eps,
                        surprise_val,
                        surprise_str
                    ])
            else:
                print(f"No earnings_dates dataframe for {ticker}.")
        except Exception as e:
            print(f"Error processing past earnings for {ticker}: {e}")

        try:
            if isinstance(cal, dict) and 'Earnings Date' in cal:
                ed = cal['Earnings Date']
                if isinstance(ed, list) and ed:
                    ed_date = ed[0]
                else:
                    ed_date = ed
                if isinstance(ed_date, pd.Timestamp):
                    ed_date = ed_date.date()

                if ed_date and ed_date >= today:
                    print(f"Upcoming earnings detected: {ed_date}")
                    upcoming_rows.append((ed_date, ticker))
                else:
                    print(f"No valid upcoming earnings for {ticker}.")
            else:
                print(f"No 'Earnings Date' key in calendar for {ticker}.")
        except Exception as e:
            print(f"Error processing upcoming earnings for {ticker}: {e}")

    except Exception as e:
        print(f"General error processing {ticker}: {e}")

print("\n=== FINISHED COLLECTION ===\n")

# Save Past Earnings with Reporting Today + Accordion
if past_rows:
    dfp = pd.DataFrame(past_rows, columns=[
        'Ticker', 'Earnings Date', 'EPS Estimate', 'Reported EPS',
        'Surprise Value', 'Surprise HTML'
    ])
    dfp['Earnings Date'] = pd.to_datetime(dfp['Earnings Date'])
    dfp.sort_values('Earnings Date', ascending=False, inplace=True)

    note = f"<p>Showing earnings from {seven_days_ago} to {today}.</p>"

    # Reporting Today line
    reporting_html = ""
    if reporting_today:
        tickers_today = ", ".join(sorted(set(reporting_today)))
        reporting_html = f"<p><strong>Reporting Today:</strong> {tickers_today}</p>"

    # Top 5 Beats and Misses
    beats = dfp.sort_values('Surprise Value', ascending=False).head(5)
    misses = dfp.sort_values('Surprise Value', ascending=True).head(5)
    summary_html = "<h3>Top 5 Earnings Beats</h3><ul>"
    for _, row in beats.iterrows():
        summary_html += f"<li>{row['Ticker']}: {row['Surprise Value']:+.2f}%</li>"
    summary_html += "</ul><h3>Top 5 Earnings Misses</h3><ul>"
    for _, row in misses.iterrows():
        summary_html += f"<li>{row['Ticker']}: {row['Surprise Value']:+.2f}%</li>"
    summary_html += "</ul>"

    dfp.drop(columns=['Surprise Value'], inplace=True)
    table_head = dfp.head(10).to_html(escape=False, index=False, classes='center-table', border=0)

    if len(dfp) > 10:
        table_rest = dfp.iloc[10:].to_html(escape=False, index=False, classes='center-table', border=0)
        table_html = table_head + (
            "<details><summary>Show More</summary>" +
            table_rest +
            "</details>"
        )
    else:
        table_html = table_head

    final_html = note + reporting_html + summary_html + table_html

    with open(PAST_HTML_PATH, 'w', encoding='utf-8') as f:
        f.write(final_html)
else:
    with open(PAST_HTML_PATH, 'w', encoding='utf-8') as f:
        f.write("<p>No earnings in the past 7 days.</p>")

# Save Upcoming Earnings
if upcoming_rows:
    upcoming_rows.sort()
    half = (len(upcoming_rows) + 1) // 2
    left = upcoming_rows[:half]
    right = upcoming_rows[half:]

    table_html = (
        "<table class='center-table'><thead><tr><th>Ticker</th><th>Date</th><th>Ticker</th><th>Date</th></tr></thead><tbody>"
    )
    for i in range(half):
        row = "<tr>"
        if i < len(left):
            row += f"<td>{left[i][1]}</td><td>{left[i][0]}</td>"
        else:
            row += "<td></td><td></td>"
        if i < len(right):
            row += f"<td>{right[i][1]}</td><td>{right[i][0]}</td>"
        else:
            row += "<td></td><td></td>"
        row += "</tr>"
        table_html += row
    table_html += "</tbody></table>"

    with open(UPCOMING_HTML_PATH, 'w', encoding='utf-8') as f:
        f.write(table_html)
else:
    with open(UPCOMING_HTML_PATH, 'w', encoding='utf-8') as f:
        f.write("<p>No upcoming earnings scheduled.</p>")
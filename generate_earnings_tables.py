# generate_earnings_tables_upgraded.py

import os
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from ticker_manager import read_tickers, modify_tickers

# Disable peewee caching properly
OUTPUT_DIR = 'charts'
os.makedirs(OUTPUT_DIR, exist_ok=True)
tz_cache = os.path.join(OUTPUT_DIR, 'tz_cache.json')
open(tz_cache, 'a').close()
yf.set_tz_cache_location(tz_cache)

# Constants
TICKERS_FILE_PATH    = 'tickers.csv'
PAST_HTML_PATH       = os.path.join(OUTPUT_DIR, 'earnings_past.html')
UPCOMING_HTML_PATH   = os.path.join(OUTPUT_DIR, 'earnings_upcoming.html')

today               = datetime.now().date()
seven_days_ago      = today - timedelta(days=7)
three_days_from_now = today + timedelta(days=3)

tickers = modify_tickers(read_tickers(TICKERS_FILE_PATH), is_remote=True)

past_rows, upcoming_rows = [], []

# Collect earnings data
for ticker in tickers:
    try:
        stock = yf.Ticker(ticker)
        cal   = stock.calendar

        # Past earnings
        try:
            df = stock.earnings_dates
            if isinstance(df, pd.DataFrame):
                recent = df[(df.index.date >= seven_days_ago) & (df.index.date <= today)]
                for date, row in recent.iterrows():
                    surprise      = row.get('Surprise(%)', None)
                    surprise_val  = float(surprise) if pd.notna(surprise) else None
                    css_class     = 'positive' if surprise_val and surprise_val > 0 else 'negative' if surprise_val and surprise_val < 0 else ''
                    surprise_str  = f'<span class="{css_class}">{surprise_val:+.2f}%</span>' if surprise_val is not None else '-'

                    eps_est      = f"{row['EPS Estimate']:.2f}" if pd.notna(row.get('EPS Estimate')) else "-"
                    rpt_eps      = f"{row['Reported EPS']:.2f}" if pd.notna(row.get('Reported EPS')) else "-"
                    rev_est      = row.get('Revenue Estimate')
                    rev_est_str  = f"${rev_est:,.0f}" if pd.notna(rev_est) else "-"
                    rpt_rev      = row.get('Reported Revenue')
                    rpt_rev_str  = f"${rpt_rev:,.0f}" if pd.notna(rpt_rev) else "-"

                    past_rows.append([
                        ticker,
                        date.date().isoformat(),
                        eps_est, rpt_eps,
                        surprise_val,  # store numeric value separately
                        surprise_str,  # store HTML version separately
                        rev_est_str, rpt_rev_str
                    ])
        except Exception:
            pass

        # Upcoming earnings
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
                    upcoming_rows.append((ed_date, ticker))
        except Exception:
            pass

    except Exception:
        pass

# Save Past Earnings with Summary
if past_rows:
    dfp = pd.DataFrame(past_rows, columns=[
        'Ticker', 'Earnings Date', 'EPS Estimate', 'Reported EPS',
        'Surprise Value', 'Surprise HTML', 'Revenue Estimate', 'Reported Revenue'
    ])
    dfp['Earnings Date'] = pd.to_datetime(dfp['Earnings Date'])
    dfp.sort_values('Earnings Date', ascending=False, inplace=True)

    # Build Top 5 Beats and Misses
    beats = dfp.sort_values('Surprise Value', ascending=False).head(5)
    misses = dfp.sort_values('Surprise Value', ascending=True).head(5)

    summary_html = "<h3>Top 5 Earnings Beats</h3><ul>"
    for _, row in beats.iterrows():
        summary_html += f"<li>{row['Ticker']}: {row['Surprise Value']:+.2f}%</li>"
    summary_html += "</ul><h3>Top 5 Earnings Misses</h3><ul>"
    for _, row in misses.iterrows():
        summary_html += f"<li>{row['Ticker']}: {row['Surprise Value']:+.2f}%</li>"
    summary_html += "</ul>"

    table_html = dfp.drop(columns=['Surprise Value']).to_html(
        escape=False, index=False, classes='center-table', border=0
    )

    note = f"<p>Showing earnings from {seven_days_ago} to {today}.</p>"
    final_html = note + summary_html + table_html

    with open(PAST_HTML_PATH, 'w', encoding='utf-8') as f:
        f.write(final_html)
else:
    with open(PAST_HTML_PATH, 'w', encoding='utf-8') as f:
        f.write("<p>No earnings in the past 7 days.</p>")

# Save Upcoming Earnings (2 columns layout)
if upcoming_rows:
    upcoming_rows.sort()
    # Split into two columns
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
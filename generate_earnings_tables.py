# generate_earnings_tables.py

import os
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from ticker_manager import read_tickers, modify_tickers

# ==== DISABLE PEEWEE CACHING ====
# Create a dummy tz cache file so yfinance won't try to stat(None)
OUTPUT_DIR = 'charts'
os.makedirs(OUTPUT_DIR, exist_ok=True)
dummy_tz = os.path.join(OUTPUT_DIR, 'tz_cache.json')
open(dummy_tz, 'a').close()
yf.set_tz_cache_location(dummy_tz)

# Constants
TICKERS_FILE_PATH   = 'tickers.csv'
PAST_HTML_PATH      = os.path.join(OUTPUT_DIR, 'earnings_past.html')
UPCOMING_HTML_PATH  = os.path.join(OUTPUT_DIR, 'earnings_upcoming.html')

today               = datetime.now().date()
seven_days_ago      = today - timedelta(days=7)
three_days_from_now = today + timedelta(days=3)

# Load & normalize tickers
tickers = modify_tickers(read_tickers(TICKERS_FILE_PATH), is_remote=True)

past_rows, upcoming_rows = [], []

print("\n=== STARTING EARNINGS COLLECTION ===\n")
for ticker in tickers:
    print(f"--- Processing {ticker} ---")
    try:
        stock = yf.Ticker(ticker)
        cal   = stock.calendar

        # 1) Show raw calendar dict
        print("  calendar type:", type(cal))
        if isinstance(cal, dict):
            print("  calendar keys:", list(cal.keys()))
            print("  raw 'Earnings Date':", cal.get('Earnings Date'))
        else:
            print("  calendar (non-dict):\n", cal)

        # 2) Past Earnings (unchanged)
        try:
            df = stock.earnings_dates
            if isinstance(df, pd.DataFrame):
                recent = df[(df.index.date >= seven_days_ago) & (df.index.date <= today)]
                print("  Past earnings dates:", list(recent.index.date) if not recent.empty else "None")
                for date, row in recent.iterrows():
                    surprise      = row.get('Surprise(%)', None)
                    surprise_str  = f"{surprise:+.2f}%" if pd.notna(surprise) else "-"
                    css_class     = 'positive' if surprise>0 else 'negative' if surprise<0 else ''
                    surprise_html = f'<span class="{css_class}">{surprise_str}</span>' if css_class else surprise_str

                    eps_est      = f"{row['EPS Estimate']:.2f}" if pd.notna(row.get('EPS Estimate')) else "-"
                    rpt_eps      = f"{row['Reported EPS']:.2f}"   if pd.notna(row.get('Reported EPS'))   else "-"
                    rev_est      = row.get('Revenue Estimate')
                    rev_est_str  = f"${rev_est:,.0f}"             if pd.notna(rev_est)                  else "-"
                    rpt_rev      = row.get('Reported Revenue')
                    rpt_rev_str  = f"${rpt_rev:,.0f}"             if pd.notna(rpt_rev)                  else "-"

                    past_rows.append([
                        ticker,
                        date.date().isoformat(),
                        eps_est, rpt_eps,
                        surprise_html,
                        rev_est_str, rpt_rev_str
                    ])
            else:
                print("  No past-earnings DataFrame.")
        except Exception as e:
            print("  Past block error:", e)

        # 3) Upcoming Earnings (dict handling)
        try:
            if isinstance(cal, dict) and 'Earnings Date' in cal:
                ed = cal['Earnings Date']
                # unwrap list vs single
                if isinstance(ed, list) and ed:
                    ed_date = ed[0]
                else:
                    ed_date = ed
                if isinstance(ed_date, pd.Timestamp):
                    ed_date = ed_date.date()
                print("  Parsed earnings_date:", ed_date)
                if ed_date and ed_date >= today:
                    cls = 'highlight-soon' if ed_date<=three_days_from_now else ''
                    upcoming_rows.append((
                        ed_date,
                        f'<tr class="{cls}"><td>{ticker}</td><td>{ed_date}</td></tr>'
                    ))
                    print(f"  → Upcoming earnings for {ticker} on {ed_date}")
                else:
                    print(f"  → {ticker} earnings date is past or None")
            else:
                print("  No 'Earnings Date' key in calendar")
        except Exception as e:
            print("  Upcoming block error:", e)

    except Exception as e:
        print("  Ticker-level error:", e)

print("\n=== DONE COLLECTING ===\n")

# 4) Write Past Earnings HTML
if past_rows:
    dfp = pd.DataFrame(past_rows, columns=[
        'Ticker','Earnings Date','EPS Estimate','Reported EPS',
        'Surprise','Revenue Estimate','Reported Revenue'
    ])
    dfp['Earnings Date']=pd.to_datetime(dfp['Earnings Date'])
    dfp.sort_values('Earnings Date',ascending=False,inplace=True)

    note = f"<p>Showing earnings from {seven_days_ago} to {today}.</p>"
    table = dfp.to_html(escape=False,index=False,classes='center-table',border=0)
    open(PAST_HTML_PATH,'w',encoding='utf-8').write(note+table)
    print("Past earnings table saved.")
else:
    open(PAST_HTML_PATH,'w',encoding='utf-8').write("<p>No earnings in the past 7 days.</p>")
    print("No past earnings to save.")

# 5) Write Upcoming Earnings HTML AND PRINT IT
if upcoming_rows:
    upcoming_rows.sort(key=lambda x: x[0])
    body = "".join(row for _,row in upcoming_rows)
    html = (
        "<table class='center-table'>"
        "<thead><tr><th>Ticker</th><th>Upcoming Earnings Date</th></tr></thead>"
        "<tbody>" + body + "</tbody></table>"
    )
    open(UPCOMING_HTML_PATH,'w',encoding='utf-8').write(html)
    print("Upcoming earnings table saved.")
    print("\n--- GENERATED Upcoming Earnings HTML ---\n")
    print(html)
    print("\n----------------------------------------\n")
else:
    open(UPCOMING_HTML_PATH,'w',encoding='utf-8').write("<p>No upcoming earnings scheduled.</p>")
    print("No upcoming earnings to save.")

# 6) Summary
print("\n=== SUMMARY ===")
print("Tickers:", len(tickers))
print("Past events:", len(past_rows))
print("Upcoming events:", len(upcoming_rows))
print("=== END ===\n")
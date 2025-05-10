# generate_earnings_tables_to_db.py

import os
import sqlite3
import logging
import pandas as pd
from datetime import datetime
import yfinance as yf
from ticker_manager import read_tickers, modify_tickers

# ——— Setup ———
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
OUTPUT_DIR         = 'charts'
DB_PATH            = 'stock data.db'            # ← lowercase, with space
PAST_HTML_PATH     = os.path.join(OUTPUT_DIR, 'earnings_past.html')
UPCOMING_HTML_PATH = os.path.join(OUTPUT_DIR, 'earnings_upcoming.html')

os.makedirs(OUTPUT_DIR, exist_ok=True)
yf.set_tz_cache_location(os.path.join(OUTPUT_DIR, 'tz_cache'))

# ——— Connect & ensure tables ———
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS earnings_past (
    ticker           TEXT,
    earnings_date    TEXT,
    eps_estimate     TEXT,
    reported_eps     TEXT,
    surprise_percent REAL,
    timestamp        TEXT,
    PRIMARY KEY (ticker, earnings_date)
)
''')
cursor.execute('''
CREATE TABLE IF NOT EXISTS earnings_upcoming (
    ticker         TEXT,
    earnings_date  TEXT,
    timestamp      TEXT,
    PRIMARY KEY (ticker, earnings_date)
)
''')

# ——— Time boundaries ———
today            = pd.to_datetime(datetime.now().date())
seven_days_ago   = today - pd.Timedelta(days=7)
seven_days_out   = today + pd.Timedelta(days=7)
ninety_days_out  = today + pd.Timedelta(days=90)

# ——— Load tickers ———
tickers = modify_tickers(read_tickers('tickers.csv'), is_remote=True)
reporting_today = set()

# ——— Fetch & store ———
for ticker in tickers:
    logging.info(f"Processing {ticker}")
    try:
        stock = yf.Ticker(ticker)
        df = stock.get_earnings_dates(limit=30)
        if df is None or df.empty:
            continue

        # normalize index
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()

        # Past earnings
        recent = df[(df.index >= seven_days_ago) & (df.index <= today)]
        for edate, row in recent.iterrows():
            surprise = pd.to_numeric(row.get('Surprise(%)'), errors='coerce')
            eps_est = row.get('EPS Estimate')
            rpt_eps = row.get('Reported EPS')
            if edate == today:
                reporting_today.add(ticker)

            cursor.execute('''
                INSERT OR REPLACE INTO earnings_past
                  (ticker, earnings_date, eps_estimate, reported_eps, surprise_percent, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                ticker,
                edate.date().isoformat(),
                f"{eps_est:.2f}" if pd.notna(eps_est) else None,
                f"{rpt_eps:.2f}" if pd.notna(rpt_eps) else None,
                float(surprise) if pd.notna(surprise) else None,
                datetime.utcnow().isoformat()
            ))

        # Upcoming earnings (next 90 days)
        future = df[(df.index > today) & (df.index <= ninety_days_out)]
        for fdate in future.index:
            cursor.execute('''
                INSERT OR REPLACE INTO earnings_upcoming
                  (ticker, earnings_date, timestamp)
                VALUES (?, ?, ?)
            ''', (
                ticker,
                fdate.date().isoformat(),
                datetime.utcnow().isoformat()
            ))

    except Exception as e:
        logging.error(f"Error processing {ticker}: {e}")

# commit all inserts
conn.commit()
conn.close()

# ——— Render Past Earnings HTML ———
conn = sqlite3.connect(DB_PATH)
dfp = pd.read_sql_query(f"""
    SELECT * FROM earnings_past
    WHERE earnings_date BETWEEN '{seven_days_ago.date()}' AND '{today.date()}'
""", conn, parse_dates=['earnings_date'])
conn.close()

if not dfp.empty:
    dfp['Surprise Value'] = pd.to_numeric(dfp['surprise_percent'], errors='coerce')
    dfp['Surprise HTML']  = dfp['Surprise Value'].apply(
        lambda x: (
            f'<span class="{"positive" if x>0 else "negative" if x<0 else ""}">'
            f"{x:+.2f}%</span>"
        ) if pd.notna(x) else "-"
    )
    dfp.sort_values('earnings_date', ascending=False, inplace=True)

    note = f"<p>Showing earnings from {seven_days_ago.date()} to {today.date()}.</p>"
    reporting_html = (
        f"<p><strong>Reporting Today:</strong> {', '.join(sorted(reporting_today))}</p>"
        if reporting_today else ""
    )

    beats  = dfp.nlargest(5, 'Surprise Value')
    misses = dfp[dfp['Surprise Value'] < 0].nsmallest(5, 'Surprise Value')
    summary_html = (
        "<h3>Top 5 Earnings Beats</h3><ul>"
        + "".join(f"<li>{r['ticker']}: {r['Surprise Value']:+.2f}%</li>" for _, r in beats.iterrows())
        + "</ul><h3>Top 5 Earnings Misses</h3><ul>"
        + "".join(f"<li>{r['ticker']}: {r['Surprise Value']:+.2f}%</li>" for _, r in misses.iterrows())
        + "</ul>"
    )

    dfp_display = (
        dfp[['ticker','earnings_date','eps_estimate','reported_eps','Surprise HTML']]
        .rename(columns={
            'ticker':        'Ticker',
            'earnings_date': 'Earnings Date',
            'eps_estimate':  'EPS Estimate',
            'reported_eps':  'Reported EPS',
            'Surprise HTML': 'Surprise'
        })
    )

    head_html = dfp_display.head(10).to_html(
        escape=False, index=False, classes='center-table', border=0
    )
    if len(dfp_display) > 10:
        rest_html  = dfp_display.iloc[10:].to_html(
            escape=False, index=False, classes='center-table', border=0
        )
        table_html = head_html + f"<details><summary>Show More</summary>{rest_html}</details>"
    else:
        table_html = head_html

    with open(PAST_HTML_PATH, 'w', encoding='utf-8') as f:
        f.write(note + reporting_html + summary_html + table_html)
else:
    with open(PAST_HTML_PATH, 'w', encoding='utf-8') as f:
        f.write("<p>No earnings in the past 7 days.</p>")

# ——— Render Upcoming Earnings HTML ———
conn = sqlite3.connect(DB_PATH)
dfu = pd.read_sql_query(f"""
    SELECT * FROM earnings_upcoming
    WHERE earnings_date > '{today.date()}' 
      AND earnings_date <= '{ninety_days_out.date()}'
""", conn, parse_dates=['earnings_date'])
conn.close()

if not dfu.empty:
    dfu['Date'] = pd.to_datetime(dfu['earnings_date'])
    dfu.sort_values('Date', inplace=True)

    early, later = [], []
    for date, group in dfu.groupby(dfu['Date'].dt.date):
        if date <= seven_days_out.date():
            early.append((date, group))
        else:
            later.append((date, group))

    html = ""
    for date, group in early:
        html += f"<h3>{date}</h3><ul>"
        for _, row in group.iterrows():
            html += f"<li>{row['ticker']}</li>"
        html += "</ul>"

    if later:
        html += '<details><summary>Show More Upcoming Earnings</summary>'
        for date, group in later:
            html += f"<h3>{date}</h3><ul>"
            for _, row in group.iterrows():
                html += f"<li>{row['ticker']}</li>"
            html += "</ul>"
        html += "</details>"

    with open(UPCOMING_HTML_PATH, 'w', encoding='utf-8') as f:
        f.write(html)
else:
    with open(UPCOMING_HTML_PATH, 'w', encoding='utf-8') as f:
        f.write("<p>No upcoming earnings in the next 90 days.</p>")
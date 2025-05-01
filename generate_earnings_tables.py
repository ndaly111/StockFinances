# generate_earnings_tables_upgraded_debug_v2.py

import os
import logging
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from ticker_manager import read_tickers, modify_tickers

# ——— Configuration ———
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
OUTPUT_DIR = 'charts'
TICKERS_FILE_PATH = 'tickers.csv'
PAST_HTML_PATH = os.path.join(OUTPUT_DIR, 'earnings_past.html')
UPCOMING_HTML_PATH = os.path.join(OUTPUT_DIR, 'earnings_upcoming.html')

# ——— Prepare output directory & yfinance cache ———
os.makedirs(OUTPUT_DIR, exist_ok=True)
tz_cache_dir = os.path.join(OUTPUT_DIR, 'tz_cache')
os.makedirs(tz_cache_dir, exist_ok=True)
yf.set_tz_cache_location(tz_cache_dir)

# ——— Date boundaries ———
today = datetime.now().date()
seven_days_ago = today - timedelta(days=7)

# ——— Load and normalize tickers ———
tickers = modify_tickers(read_tickers(TICKERS_FILE_PATH), is_remote=True)

past_rows = []
upcoming_rows = []
reporting_today = set()

logging.info("=== STARTING COLLECTION ===")

for ticker in tickers:
    logging.info(f"Processing {ticker}")
    try:
        stock = yf.Ticker(ticker)

        # Try modern get_earnings_dates; fallback to .earnings_dates
        try:
            df = stock.get_earnings_dates(limit=30)
        except (TypeError, AttributeError):
            df = stock.earnings_dates

        if df is None or df.empty:
            logging.info("  No earnings data returned")
            continue

        df = pd.DataFrame(df) if not isinstance(df, pd.DataFrame) else df.copy()
        df.index = pd.to_datetime(df.index).date

        # Past earnings within range
        recent = df.loc[seven_days_ago:today]
        for edate, row in recent.iterrows():
            if edate == today:
                reporting_today.add(ticker)

            surprise_val = pd.to_numeric(row.get('Surprise(%)'), errors='coerce')
            css = 'positive' if surprise_val > 0 else 'negative' if surprise_val < 0 else ''
            surprise_html = f'<span class="{css}">{surprise_val:+.2f}%</span>' if pd.notna(surprise_val) else '-'
            eps_est = f"{row.get('EPS Estimate'):.2f}" if pd.notna(row.get('EPS Estimate')) else "-"
            rpt_eps = f"{row.get('Reported EPS'):.2f}" if pd.notna(row.get('Reported EPS')) else "-"
            past_rows.append([ticker, edate.isoformat(), eps_est, rpt_eps, surprise_val, surprise_html])

        # First future earnings date
        future = df.loc[df.index > today]
        if not future.empty:
            future_date = future.index[0]
            if isinstance(future_date, pd.Timestamp):
                future_date = future_date.date()
            upcoming_rows.append((ticker, future_date))
            logging.info(f"  Upcoming on {future_date}")
        else:
            logging.info("  No upcoming earnings ≥ today")

    except Exception as e:
        logging.error(f"General error processing {ticker}: {e}")

logging.info(f"=== FINISHED COLLECTION: {len(past_rows)} past rows, {len(upcoming_rows)} upcoming rows ===")

# ---------- Past Earnings HTML ----------
if past_rows:
    dfp = pd.DataFrame(past_rows, columns=[
        'Ticker', 'Earnings Date', 'EPS Estimate', 'Reported EPS',
        'Surprise Value', 'Surprise HTML'
    ])
    dfp['Earnings Date'] = pd.to_datetime(dfp['Earnings Date'])
    dfp['Surprise Value'] = pd.to_numeric(dfp['Surprise Value'], errors='coerce')
    dfp.sort_values('Earnings Date', ascending=False, inplace=True)

    note = f"<p>Showing earnings from {seven_days_ago} to {today}.</p>"
    reporting_html = f"<p><strong>Reporting Today:</strong> {', '.join(sorted(reporting_today))}</p>" if reporting_today else ""

    beats = dfp.nlargest(5, 'Surprise Value')
    misses = dfp.nsmallest(5, 'Surprise Value')
    summary_html = (
        "<h3>Top 5 Earnings Beats</h3><ul>"
        + "".join(f"<li>{r['Ticker']}: {r['Surprise Value']:+.2f}%</li>" for _, r in beats.iterrows())
        + "</ul><h3>Top 5 Earnings Misses</h3><ul>"
        + "".join(f"<li>{r['Ticker']}: {r['Surprise Value']:+.2f}%</li>" for _, r in misses.iterrows())
        + "</ul>"
    )

    dfp.drop(columns=['Surprise Value'], inplace=True)
    head_html = dfp.head(10).to_html(escape=False, index=False, classes='center-table', border=0)
    if len(dfp) > 10:
        rest_html = dfp.iloc[10:].to_html(escape=False, index=False, classes='center-table', border=0)
        table_html = head_html + f"<details><summary>Show More</summary>{rest_html}</details>"
    else:
        table_html = head_html

    with open(PAST_HTML_PATH, 'w', encoding='utf-8') as f:
        f.write(note + reporting_html + summary_html + table_html)
else:
    with open(PAST_HTML_PATH, 'w', encoding='utf-8') as f:
        f.write("<p>No earnings in the past 7 days.</p>")

# ---------- Upcoming Earnings HTML ----------
if upcoming_rows:
    df_up = pd.DataFrame(upcoming_rows, columns=['Ticker', 'Date'])
    half = (len(df_up) + 1) // 2
    left, right = df_up.iloc[:half], df_up.iloc[half:]
    html = (
        "<table class='center-table'><thead><tr><th>Ticker</th><th>Date</th>"
        "<th>Ticker</th><th>Date</th></tr></thead><tbody>"
    )
    for i in range(half):
        l = left.iloc[i] if i < len(left) else {'Ticker': '', 'Date': ''}
        r = right.iloc[i] if i < len(right) else {'Ticker': '', 'Date': ''}
        html += f"<tr><td>{l.Ticker}</td><td>{l.Date}</td><td>{r.Ticker}</td><td>{r.Date}</td></tr>"
    html += "</tbody></table>"
    with open(UPCOMING_HTML_PATH, 'w', encoding='utf-8') as f:
        f.write(html)
else:
    with open(UPCOMING_HTML_PATH, 'w', encoding='utf-8') as f:
        f.write("<p>No upcoming earnings scheduled.</p>")
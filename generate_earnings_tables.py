# Replace your entire script with this version

import os
import logging
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from ticker_manager import read_tickers, modify_tickers

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
OUTPUT_DIR = 'charts'
TICKERS_FILE_PATH = 'tickers.csv'
PAST_HTML_PATH = os.path.join(OUTPUT_DIR, 'earnings_past.html')
UPCOMING_HTML_PATH = os.path.join(OUTPUT_DIR, 'earnings_upcoming.html')

os.makedirs(OUTPUT_DIR, exist_ok=True)
tz_cache_dir = os.path.join(OUTPUT_DIR, 'tz_cache')
os.makedirs(tz_cache_dir, exist_ok=True)
yf.set_tz_cache_location(tz_cache_dir)

today = pd.to_datetime(datetime.now().date())
seven_days_ago = today - pd.Timedelta(days=7)
three_days_out = today + pd.Timedelta(days=3)

tickers = modify_tickers(read_tickers(TICKERS_FILE_PATH), is_remote=True)
past_rows, upcoming_rows = [], []
reporting_today = set()

logging.info("=== STARTING COLLECTION ===")

for ticker in tickers:
    logging.info(f"Processing {ticker}")
    try:
        stock = yf.Ticker(ticker)
        try:
            df = stock.get_earnings_dates(limit=30)
        except (TypeError, AttributeError):
            df = stock.earnings_dates

        if df is None or df.empty:
            logging.info("  No earnings data returned")
            continue

        df = pd.DataFrame(df) if not isinstance(df, pd.DataFrame) else df.copy()
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()

        recent = df.loc[(df.index >= seven_days_ago) & (df.index <= today)]
        for edate, row in recent.iterrows():
            if edate == today:
                reporting_today.add(ticker)

            surprise_val = pd.to_numeric(row.get('Surprise(%)'), errors='coerce')
            css = 'positive' if surprise_val > 0 else 'negative' if surprise_val < 0 else ''
            surprise_html = f'<span class="{css}">{surprise_val:+.2f}%</span>' if pd.notna(surprise_val) else '-'
            eps_est = f"{row.get('EPS Estimate'):.2f}" if pd.notna(row.get('EPS Estimate')) else "-"
            rpt_eps = f"{row.get('Reported EPS'):.2f}" if pd.notna(row.get('Reported EPS')) else "-"
            past_rows.append([ticker, edate.date().isoformat(), eps_est, rpt_eps, surprise_val, surprise_html])

        future = df.loc[df.index > today]
        for fdate in future.index:
            upcoming_rows.append((ticker, fdate.normalize()))
        logging.info(f"  Upcoming count: {len(future)}")

    except Exception as e:
        logging.error(f"General error processing {ticker}: {e}")

logging.info(f"=== FINISHED COLLECTION: {len(past_rows)} past rows, {len(upcoming_rows)} upcoming rows ===")

# Past earnings HTML
if past_rows:
    dfp = pd.DataFrame(past_rows, columns=['Ticker', 'Earnings Date', 'EPS Estimate', 'Reported EPS', 'Surprise Value', 'Surprise HTML'])
    dfp['Earnings Date'] = pd.to_datetime(dfp['Earnings Date'])
    dfp['Surprise Value'] = pd.to_numeric(dfp['Surprise Value'], errors='coerce')
    dfp.sort_values('Earnings Date', ascending=False, inplace=True)

    note = f"<p>Showing earnings from {seven_days_ago.date()} to {today.date()}.</p>"
    reporting_html = f"<p><strong>Reporting Today:</strong> {', '.join(sorted(reporting_today))}</p>" if reporting_today else ""

    beats = dfp.nlargest(5, 'Surprise Value')
    misses = dfp[dfp['Surprise Value'] < 0].nsmallest(5, 'Surprise Value')
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

# Upcoming earnings HTML
if upcoming_rows:
    df_up = pd.DataFrame(upcoming_rows, columns=['Ticker', 'Date'])
    df_up['Date'] = pd.to_datetime(df_up['Date'])
    df_up.sort_values('Date', inplace=True)

    half = (len(df_up) + 1) // 2
    left, right = df_up.iloc[:half], df_up.iloc[half:]

    html = """
    <script>
    function toggleEarnings() {
        const btn = document.getElementById('toggle-btn');
        const longRows = document.querySelectorAll('.long-term');
        const showingAll = btn.dataset.state === 'all';
        longRows.forEach(row => {
            row.style.display = showingAll ? 'none' : 'table-row';
        });
        btn.textContent = showingAll ? 'Show All Upcoming Earnings' : 'Show Only Next 3 Days';
        btn.dataset.state = showingAll ? 'short' : 'all';
    }
    </script>
    <button id="toggle-btn" onclick="toggleEarnings()" data-state="short" style="margin: 10px 0;">Show All Upcoming Earnings</button>
    <table class='center-table'>
        <thead><tr><th>Ticker</th><th>Date</th><th>Ticker</th><th>Date</th></tr></thead>
        <tbody>
    """

    for i in range(half):
        l = left.iloc[i] if i < len(left) else {'Ticker': '', 'Date': pd.NaT}
        r = right.iloc[i] if i < len(right) else {'Ticker': '', 'Date': pd.NaT}

        def classify(row_date):
            if pd.isna(row_date):
                return 'long-term'
            d = row_date.date()
            if d == today.date():
                return 'reporting-today'
            elif d <= three_days_out.date():
                return 'near-term'
            else:
                return 'long-term'

        l_class = classify(l['Date'])
        r_class = classify(r['Date'])
        row_class = 'reporting-today' if 'reporting-today' in (l_class, r_class) else 'near-term' if 'near-term' in (l_class, r_class) else 'long-term'

        display_style = '' if row_class in ('near-term', 'reporting-today') else ' style="display:none;"'
        bg_color = ' style="background-color:#fff3b0;"' if row_class == 'reporting-today' else ''

        l_date_str = l['Date'].date() if pd.notna(l['Date']) else ''
        r_date_str = r['Date'].date() if pd.notna(r['Date']) else ''

        html += f"<tr class='{row_class}'{display_style}{bg_color}>"
        html += f"<td>{l['Ticker']}</td><td>{l_date_str}</td><td>{r['Ticker']}</td><td>{r_date_str}</td></tr>"

    html += "</tbody></table>"

    with open(UPCOMING_HTML_PATH, 'w', encoding='utf-8') as f:
        f.write(html)
else:
    with open(UPCOMING_HTML_PATH, 'w', encoding='utf-8') as f:
        f.write("<p>No upcoming earnings scheduled.</p>")
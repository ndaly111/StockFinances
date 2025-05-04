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
three_days_out = today + timedelta(days=3)

true_today = pd.to_datetime(today)  # Convert to datetime for comparison
true_seven_days_ago = pd.to_datetime(seven_days_ago)
true_three_days_out = pd.to_datetime(three_days_out)

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

        try:
            df = stock.get_earnings_dates(limit=30)
        except (TypeError, AttributeError):
            df = stock.earnings_dates

        if df is None or df.empty:
            logging.info("  No earnings data returned")
            continue

        df = pd.DataFrame(df) if not isinstance(df, pd.DataFrame) else df.copy()
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()

        # Past earnings within range
        recent = df.loc[(df.index >= true_seven_days_ago) & (df.index <= true_today)]
        for edate, row in recent.iterrows():
            if edate.date() == today:
                reporting_today.add(ticker)

            surprise_val = pd.to_numeric(row.get('Surprise(%)'), errors='coerce')
            css = 'positive' if surprise_val > 0 else 'negative' if surprise_val < 0 else ''
            surprise_html = f'<span class="{css}">{surprise_val:+.2f}%</span>' if pd.notna(surprise_val) else '-'
            eps_est = f"{row.get('EPS Estimate'):.2f}" if pd.notna(row.get('EPS Estimate')) else "-"
            rpt_eps = f"{row.get('Reported EPS'):.2f}" if pd.notna(row.get('Reported EPS')) else "-"
            past_rows.append([ticker, edate.date().isoformat(), eps_est, rpt_eps, surprise_val, surprise_html])

        # All future earnings
        future = df.loc[df.index > true_today]
        if not future.empty:
            for fdate in future.index:
                future_date = fdate.date() if isinstance(fdate, pd.Timestamp) else fdate
                upcoming_rows.append((ticker, future_date))
            logging.info(f"  Upcoming count: {len(future)}")
        else:
            logging.info("  No upcoming earnings")

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

# ---------- Upcoming Earnings HTML (with Toggle and Highlighting) ----------
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
        l = left.iloc[i] if i < len(left) else {'Ticker': '', 'Date': ''}
        r = right.iloc[i] if i < len(right) else {'Ticker': '', 'Date': ''}

        def classify(row_date):
            d = row_date.date() if isinstance(row_date, pd.Timestamp) else row_date
            if d == today:
                return 'reporting-today'
            elif d <= three_days_out:
                return 'near-term'
            else:
                return 'long-term'

        l_class = classify(l['Date']) if isinstance(l, pd.Series) else ''
        r_class = classify(r['Date']) if isinstance(r, pd.Series) else ''
        row_class = 'long-term'
        if 'reporting-today' in (l_class, r_class):
            row_class = 'reporting-today'
        elif 'near-term' in (l_class, r_class):
            row_class = 'near-term'

        display_style = '' if row_class in ('near-term', 'reporting-today') else ' style="display:none;"'
        bg_color = ' style="background-color:#fff3b0;"' if row_class == 'reporting-today' else ''

        html += f"<tr class='{row_class}'{display_style}{bg_color}>"
        html += f"<td>{l['Ticker']}</td><td>{l['Date'].date()}</td><td>{r['Ticker']}</td><td>{r['Date'].date()}</td></tr>"

    html += "</tbody></table>"

    with open(UPCOMING_HTML_PATH, 'w', encoding='utf-8') as f:
        f.write(html)
else:
    with open(UPCOMING_HTML_PATH, 'w', encoding='utf-8') as f:
        f.write("<p>No upcoming earnings scheduled.</p>")
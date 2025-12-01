import os
import sqlite3
import logging
import pandas as pd
from datetime import datetime
import yfinance as yf
from ticker_manager import read_tickers, modify_tickers

# yfinance relies on requests; fail fast if it's missing so callers see a
# clear error instead of silent empty tables.
try:  # pragma: no cover - import check only
    import requests  # noqa: F401
except Exception as exc:  # pragma: no cover - error path
    raise ImportError("The 'requests' package is required to fetch earnings data") from exc

def ensure_column(cursor: sqlite3.Cursor, table: str, name: str, definition: str) -> None:
    """Add a column to *table* if it does not already exist."""

    cursor.execute(f"PRAGMA table_info({table})")
    existing = {row[1] for row in cursor.fetchall()}
    if name not in existing:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")


def parse_revenue_value(value):
    """Return a float revenue number in dollars from Yahoo's formatted values."""

    if pd.isna(value):
        return None

    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip().replace(",", "")
    if not text:
        return None

    multiplier = 1
    if text.endswith("B"):
        multiplier = 1e9
        text = text[:-1]
    elif text.endswith("M"):
        multiplier = 1e6
        text = text[:-1]
    elif text.endswith("K"):
        multiplier = 1e3
        text = text[:-1]

    try:
        return float(text) * multiplier
    except ValueError:
        return None


def format_compact_currency(value: float) -> str:
    """Format a revenue figure as a compact currency string."""

    if value is None or pd.isna(value):
        return "-"

    abs_val = abs(value)
    if abs_val >= 1e9:
        return f"${value / 1e9:.2f}B"
    if abs_val >= 1e6:
        return f"${value / 1e6:.2f}M"
    if abs_val >= 1e3:
        return f"${value / 1e3:.0f}K"
    return f"${value:,.0f}"


def compute_surprise_percent(estimate: float, actual: float) -> float | None:
    """Compute the surprise percent from estimate and actual when available."""

    if estimate in (None, 0) or actual is None:
        return None

    try:
        return ((actual - estimate) / abs(estimate)) * 100
    except Exception:
        return None


def format_revenue(estimate, reported, surprise, surprise_formatter) -> str:
    """Build a compact revenue summary with coloured surprise when available."""

    est_txt = format_compact_currency(estimate)
    rpt_txt = format_compact_currency(reported)
    surprise_txt = surprise_formatter(surprise)

    if est_txt == "-" and rpt_txt == "-" and surprise_txt == "-":
        return "-"

    return f"{est_txt} → {rpt_txt} ({surprise_txt})"

def generate_earnings_tables():
    # ——— Setup ———
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    OUTPUT_DIR         = 'charts'
    DB_PATH            = 'Stock Data.db'
    PAST_HTML_PATH     = os.path.join(OUTPUT_DIR, 'earnings_past.html')
    UPCOMING_HTML_PATH = os.path.join(OUTPUT_DIR, 'earnings_upcoming.html')

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    yf.set_tz_cache_location(os.path.join(OUTPUT_DIR, 'tz_cache'))

    # ——— Connect & ensure tables ———
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS earnings_past (
        ticker TEXT,
        earnings_date TEXT,
        eps_estimate TEXT,
        reported_eps TEXT,
        surprise_percent REAL,
        revenue_estimate REAL,
        reported_revenue REAL,
        revenue_surprise_percent REAL,
        timestamp TEXT,
        PRIMARY KEY (ticker, earnings_date)
    )''')
    ensure_column(cursor, "earnings_past", "revenue_estimate", "REAL")
    ensure_column(cursor, "earnings_past", "reported_revenue", "REAL")
    ensure_column(cursor, "earnings_past", "revenue_surprise_percent", "REAL")
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS earnings_upcoming (
        ticker TEXT,
        earnings_date TEXT,
        timestamp TEXT,
        PRIMARY KEY (ticker, earnings_date)
    )''')

    today = pd.to_datetime(datetime.now().date())
    seven_days_ago   = today - pd.Timedelta(days=7)
    seven_days_out   = today + pd.Timedelta(days=7)
    ninety_days_out  = today + pd.Timedelta(days=90)

    tickers = modify_tickers(read_tickers('tickers.csv'), is_remote=True)
    reporting_today = set()
    any_rows = False

    for ticker in tickers:
        logging.info(f"Processing {ticker}")
        try:
            stock = yf.Ticker(ticker)
            df = stock.get_earnings_dates(limit=30)
            if df is None or df.empty:
                continue

            idx = pd.to_datetime(df.index)
            try:
                if getattr(idx, "tz", None) is not None:
                    idx = idx.tz_convert(None)
                idx = idx.normalize()
            except Exception as tz_err:
                logging.warning(f"Timezone conversion issue for {ticker}: {tz_err}")
                idx = idx.tz_localize(None, errors="ignore").normalize()
            df.index = idx

            historical = df[df.index <= today]
            for edate, row in historical.iterrows():
                surprise = pd.to_numeric(row.get('Surprise(%)'), errors='coerce')
                eps_est  = row.get('EPS Estimate')
                rpt_eps  = row.get('Reported EPS')
                rev_est_raw = next(
                    (row.get(key) for key in ('Revenue Estimate', 'Revenue Est', 'Est. Revenue') if pd.notna(row.get(key))),
                    None,
                )
                rev_rpt_raw = next(
                    (row.get(key) for key in ('Reported Revenue', 'Revenue Actual', 'Actual Revenue') if pd.notna(row.get(key))),
                    None,
                )
                rev_surprise_raw = row.get('Revenue Surprise(%)')
                if pd.isna(rev_surprise_raw):
                    rev_surprise_raw = row.get('Revenue Surprise (%)')

                rev_est  = parse_revenue_value(rev_est_raw)
                rev_rpt  = parse_revenue_value(rev_rpt_raw)
                rev_surprise = pd.to_numeric(rev_surprise_raw, errors='coerce')
                if pd.isna(rev_surprise):
                    rev_surprise = compute_surprise_percent(rev_est, rev_rpt)
                if edate == today:
                    reporting_today.add(ticker)
                cursor.execute('''
                    INSERT OR REPLACE INTO earnings_past
                    (ticker, earnings_date, eps_estimate, reported_eps, surprise_percent,
                     revenue_estimate, reported_revenue, revenue_surprise_percent, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    ticker,
                    edate.date().isoformat(),
                    f"{eps_est:.2f}" if pd.notna(eps_est) else None,
                    f"{rpt_eps:.2f}" if pd.notna(rpt_eps) else None,
                    float(surprise) if pd.notna(surprise) else None,
                    float(rev_est) if rev_est is not None else None,
                    float(rev_rpt) if rev_rpt is not None else None,
                    float(rev_surprise) if pd.notna(rev_surprise) else None,
                    datetime.utcnow().isoformat()
                ))
                any_rows = True

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
                any_rows = True

        except Exception as e:
            logging.error(f"Error processing {ticker}: {e}")

    cursor.execute(
        "DELETE FROM earnings_upcoming WHERE earnings_date <= ?",
        (today.date().isoformat(),)
    )

    conn.commit()
    conn.close()

    if not any_rows:
        logging.warning("No new earnings data fetched; regenerating HTML from existing tables")

    # ——— Past Earnings HTML ———
    conn = sqlite3.connect(DB_PATH)
    dfp = pd.read_sql_query(f"""
        SELECT * FROM earnings_past
        WHERE earnings_date BETWEEN '{seven_days_ago.date()}' AND '{today.date()}'
    """, conn, parse_dates=['earnings_date'])
    conn.close()

    if not dfp.empty:
        def format_surprise(value: float) -> str:
            """Return HTML for the surprise value with a colour-coded span."""
            if pd.isna(value):
                return "-"

            css_class = "positive" if value > 0 else "negative" if value < 0 else ""
            formatted = f"{value:+.2f}%"
            return f'<span class="{css_class}">{formatted}</span>' if css_class else formatted

        dfp['Surprise Value'] = pd.to_numeric(dfp['surprise_percent'], errors='coerce')
        dfp['Surprise HTML']  = dfp['Surprise Value'].apply(format_surprise)

        dfp['Revenue Surprise Value'] = pd.to_numeric(dfp['revenue_surprise_percent'], errors='coerce')
        dfp['Revenue Surprise Value'] = dfp['Revenue Surprise Value'].combine_first(
            dfp.apply(lambda r: compute_surprise_percent(r['revenue_estimate'], r['reported_revenue']), axis=1)
        )
        dfp['Revenue Display'] = dfp.apply(
            lambda r: format_revenue(r['revenue_estimate'], r['reported_revenue'], r['Revenue Surprise Value'], format_surprise),
            axis=1
        )

        has_revenue_data = dfp[['revenue_estimate', 'reported_revenue', 'Revenue Surprise Value']].notna().any().any()

        note = f"<p>Showing earnings from {seven_days_ago.date()} to {today.date()}.</p>"
        reporting_html = (
            f"<p><strong>Reporting Today:</strong> {', '.join(sorted(reporting_today))}</p>"
            if reporting_today else ""
        )

        # only count actual beats (> 0%)
        beats = dfp[dfp['Surprise Value'] > 0].nlargest(5, 'Surprise Value')
        misses = dfp[dfp['Surprise Value'] < 0].nsmallest(5, 'Surprise Value')
        summary_html = (
            "<h3>Top 5 Earnings Beats</h3><ul>"
            + "".join(
                f"<li>{r['ticker']}: {format_surprise(r['Surprise Value'])}</li>"
                for _, r in beats.iterrows()
            )
            + "</ul><h3>Top 5 Earnings Misses</h3><ul>"
            + "".join(
                f"<li>{r['ticker']}: {format_surprise(r['Surprise Value'])}</li>"
                for _, r in misses.iterrows()
            )
            + "</ul>"
        )

        display_columns = ['ticker', 'earnings_date', 'eps_estimate', 'reported_eps', 'Surprise HTML']
        rename_map = {
            'ticker':        'Ticker',
            'earnings_date': 'Earnings Date',
            'eps_estimate':  'EPS Estimate',
            'reported_eps':  'Reported EPS',
            'Surprise HTML': 'Surprise'
        }

        if has_revenue_data:
            display_columns.append('Revenue Display')
            rename_map['Revenue Display'] = 'Revenue (Est → Reported, Surprise)'

        dfp_display = (
            dfp.sort_values('earnings_date', ascending=False)
               [display_columns]
               .rename(columns=rename_map)
        )

        head_html = dfp_display.head(10).to_html(escape=False, index=False, classes='center-table', border=0)
        if len(dfp_display) > 10:
            rest_html  = dfp_display.iloc[10:].to_html(escape=False, index=False, classes='center-table', border=0)
            table_html = head_html + f"<details><summary>Show More</summary>{rest_html}</details>"
        else:
            table_html = head_html

        with open(PAST_HTML_PATH, 'w', encoding='utf-8') as f:
            f.write(note + reporting_html + summary_html + table_html)
    else:
        with open(PAST_HTML_PATH, 'w', encoding='utf-8') as f:
            f.write("<p>No earnings in the past 7 days.</p>")

    # ——— Upcoming Earnings HTML ———
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

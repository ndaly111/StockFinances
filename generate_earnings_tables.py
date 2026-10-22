import os
import sqlite3
import logging
import pandas as pd
from datetime import datetime, timezone
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


def fetch_revenue_snapshot(ticker: str) -> dict | None:
    """Pull Yahoo's quoteSummary earnings feed for revenue.

    get_earnings_dates() carries EPS only, and quarterly_income_stmt lags a
    report by weeks.  The `earnings` module updates with the report itself:
    financialsChart has per-quarter revenue and earningsChart the matching
    actual EPS (used to confirm which report the latest quarter belongs to).
    calendarEvents.revenueAverage is the consensus for the NEXT report only,
    so it has to be snapshotted while that report is still upcoming.
    """
    try:
        tk = yf.Ticker(ticker)
        raw = tk._data.get_raw_json(
            f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}",
            params={"modules": "earnings,calendarEvents"},
        )
        res = raw["quoteSummary"]["result"][0]
    except Exception as e:
        logging.warning(f"Revenue snapshot failed for {ticker}: {e}")
        return None

    earnings = res.get("earnings") or {}
    revenue_by_q = {
        q.get("date"): (q.get("revenue") or {}).get("raw")
        for q in (earnings.get("financialsChart") or {}).get("quarterly", [])
    }
    eps_q = (earnings.get("earningsChart") or {}).get("quarterly", [])
    latest = eps_q[-1] if eps_q else {}
    return {
        "latest_eps": (latest.get("actual") or {}).get("raw"),
        "latest_revenue": revenue_by_q.get(latest.get("date")),
        "next_revenue_estimate": ((res.get("calendarEvents") or {}).get("earnings") or {})
                                 .get("revenueAverage", {}).get("raw"),
    }


def eps_matches(a, b) -> bool:
    try:
        return a is not None and b is not None and abs(float(a) - float(b)) < 0.011
    except (TypeError, ValueError):
        return False

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
    # Pre-report revenue consensus, captured while the report is upcoming
    # (Yahoo rolls the estimate to the next quarter once results are out).
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS earnings_revenue_estimates (
        ticker TEXT,
        earnings_date TEXT,
        revenue_estimate REAL,
        captured_at TEXT,
        PRIMARY KEY (ticker, earnings_date)
    )''')

    today = pd.to_datetime(datetime.now().date())
    seven_days_ago   = today - pd.Timedelta(days=7)
    seven_days_out   = today + pd.Timedelta(days=7)
    ninety_days_out  = today + pd.Timedelta(days=90)

    tickers = modify_tickers(read_tickers('tickers.csv'), is_remote=True)
    reporting_today = set()
    any_rows = False

    # Fetch every ticker's earnings dates in parallel up front (the sequential
    # loop spent ~90s/run waiting on Yahoo); processing below is unchanged.
    from concurrent.futures import ThreadPoolExecutor

    def _fetch_dates(t):
        try:
            return yf.Ticker(t).get_earnings_dates(limit=30), None
        except Exception as e:
            return None, e

    with ThreadPoolExecutor(max_workers=8) as ex:
        fetched = dict(zip(tickers, ex.map(_fetch_dates, tickers)))

    # Revenue feed only for tickers with a report in the past week (actuals)
    # or the next two weeks (estimate snapshot) — keeps the extra calls small.
    def _near_report(t):
        df = fetched[t][0]
        if df is None or df.empty:
            return False
        dates = pd.to_datetime(df.index)
        if getattr(dates, "tz", None) is not None:
            dates = dates.tz_convert(None)
        dates = dates.normalize()
        return bool(((dates >= seven_days_ago) & (dates <= today + pd.Timedelta(days=14))).any())

    rev_tickers = [t for t in tickers if _near_report(t)]
    with ThreadPoolExecutor(max_workers=8) as ex:
        revenue_feed = dict(zip(rev_tickers, ex.map(fetch_revenue_snapshot, rev_tickers)))

    for ticker in tickers:
        logging.info(f"Processing {ticker}")
        try:
            df, fetch_err = fetched[ticker]
            if fetch_err is not None:
                raise fetch_err
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
            feed = revenue_feed.get(ticker) or {}
            reported_rows = historical[pd.notna(historical['Reported EPS'])] if 'Reported EPS' in historical else historical.iloc[0:0]
            latest_report = reported_rows.index.max() if not reported_rows.empty else None
            # Yahoo's latest quarter belongs to our latest report only when its
            # actual EPS agrees with the one get_earnings_dates() shows.
            feed_in_sync = (
                latest_report is not None
                and eps_matches(feed.get('latest_eps'), reported_rows.loc[latest_report, 'Reported EPS'])
            )
            now_iso = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()

            for edate, row in historical.iterrows():
                surprise = pd.to_numeric(row.get('Surprise(%)'), errors='coerce')
                eps_est  = row.get('EPS Estimate')
                rpt_eps  = row.get('Reported EPS')
                edate_iso = edate.date().isoformat()

                est_row = cursor.execute(
                    "SELECT revenue_estimate FROM earnings_revenue_estimates WHERE ticker=? AND earnings_date=?",
                    (ticker, edate_iso),
                ).fetchone()
                rev_est = est_row[0] if est_row else None
                rev_rpt = feed.get('latest_revenue') if (feed_in_sync and edate == latest_report) else None
                rev_surprise = compute_surprise_percent(rev_est, rev_rpt)
                if edate == today:
                    reporting_today.add(ticker)
                # Revenue fields only ever fill in: a later run where Yahoo's
                # feed has moved on must not blank an earlier capture.
                cursor.execute('''
                    INSERT INTO earnings_past
                    (ticker, earnings_date, eps_estimate, reported_eps, surprise_percent,
                     revenue_estimate, reported_revenue, revenue_surprise_percent, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(ticker, earnings_date) DO UPDATE SET
                        eps_estimate     = excluded.eps_estimate,
                        reported_eps     = excluded.reported_eps,
                        surprise_percent = excluded.surprise_percent,
                        revenue_estimate = COALESCE(excluded.revenue_estimate, earnings_past.revenue_estimate),
                        reported_revenue = COALESCE(excluded.reported_revenue, earnings_past.reported_revenue),
                        revenue_surprise_percent = COALESCE(excluded.revenue_surprise_percent,
                                                            earnings_past.revenue_surprise_percent),
                        timestamp        = excluded.timestamp
                ''', (
                    ticker,
                    edate_iso,
                    f"{eps_est:.2f}" if pd.notna(eps_est) else None,
                    f"{rpt_eps:.2f}" if pd.notna(rpt_eps) else None,
                    float(surprise) if pd.notna(surprise) else None,
                    float(rev_est) if rev_est is not None else None,
                    float(rev_rpt) if rev_rpt is not None else None,
                    float(rev_surprise) if rev_surprise is not None else None,
                    now_iso
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
                    now_iso
                ))
                any_rows = True

            # Snapshot the consensus for the next report while it is still
            # upcoming; only when the feed is in sync, so the estimate is known
            # to be for the next quarter rather than the one just reported.
            next_est = feed.get('next_revenue_estimate')
            if next_est and feed_in_sync and not future.empty:
                next_date = future.index.min()
                if next_date <= today + pd.Timedelta(days=14):
                    cursor.execute('''
                        INSERT OR REPLACE INTO earnings_revenue_estimates
                        (ticker, earnings_date, revenue_estimate, captured_at)
                        VALUES (?, ?, ?, ?)
                    ''', (ticker, next_date.date().isoformat(), float(next_est), now_iso))

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
        dfp['Rev Est HTML']      = dfp['revenue_estimate'].apply(format_compact_currency)
        dfp['Rev Reported HTML'] = dfp['reported_revenue'].apply(format_compact_currency)
        dfp['Rev Surprise HTML'] = dfp['Revenue Surprise Value'].apply(format_surprise)

        has_revenue_data = dfp[['revenue_estimate', 'reported_revenue']].notna().any().any()

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
            display_columns += ['Rev Est HTML', 'Rev Reported HTML', 'Rev Surprise HTML']
            rename_map.update({
                'Rev Est HTML':      'Revenue Estimate',
                'Rev Reported HTML': 'Reported Revenue',
                'Rev Surprise HTML': 'Revenue Surprise',
            })

        dfp_display = dfp.sort_values('earnings_date', ascending=False)[display_columns].copy()
        dfp_display['earnings_date'] = dfp_display['earnings_date'].dt.strftime('%b %d')
        dfp_display = dfp_display.fillna('-').rename(columns=rename_map)

        # Eight columns overflow a phone screen; let the table swipe sideways.
        def _scroll(df):
            inner = df.to_html(escape=False, index=False, classes='center-table', border=0)
            return f'<div style="overflow-x:auto;-webkit-overflow-scrolling:touch">{inner}</div>'

        head_html = _scroll(dfp_display.head(10))
        if len(dfp_display) > 10:
            rest_html  = _scroll(dfp_display.iloc[10:])
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

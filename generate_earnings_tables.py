import os
import sqlite3
import logging
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from ticker_manager import read_tickers, modify_tickers

# ────── CONFIG ──────
DB_PATH              = 'Stock Data.db'
OUTPUT_DIR           = 'charts'
PAST_HTML            = os.path.join(OUTPUT_DIR, 'earnings_past.html')
UPCOMING_HTML        = os.path.join(OUTPUT_DIR, 'earnings_upcoming.html')
PAST_WINDOW_DAYS     = 7    # ← parameterized
UPCOMING_WINDOW_DAYS = 90   # ← parameterized

LOG = logging.getLogger(__name__)


def _ensure_tables(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute("""
      CREATE TABLE IF NOT EXISTS earnings_past (
        ticker           TEXT,
        earnings_date    TEXT,
        eps_estimate     REAL,
        reported_eps     REAL,
        surprise_percent REAL,
        timestamp        TEXT,
        PRIMARY KEY(ticker, earnings_date)
      );
    """)
    cur.execute("""
      CREATE TABLE IF NOT EXISTS earnings_upcoming (
        ticker        TEXT,
        earnings_date TEXT,
        timestamp     TEXT,
        PRIMARY KEY(ticker, earnings_date)
      );
    """)
    # index for fast date-range queries
    cur.execute("CREATE INDEX IF NOT EXISTS idx_past_date    ON earnings_past(earnings_date);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_upcoming_date ON earnings_upcoming(earnings_date);")
    conn.commit()


def _write_past(conn: sqlite3.Connection, ticker: str, edate: datetime, eps_est, rpt_eps, surprise):
    conn.execute("""
      INSERT OR REPLACE INTO earnings_past
        (ticker, earnings_date, eps_estimate, reported_eps, surprise_percent, timestamp)
      VALUES (?, ?, ?, ?, ?, ?)
    """, (
      ticker,
      edate.date().isoformat(),
      float(eps_est)    if pd.notna(eps_est)    else None,
      float(rpt_eps)    if pd.notna(rpt_eps)    else None,
      float(surprise)   if pd.notna(surprise)   else None,
      datetime.utcnow().isoformat()
    ))
    LOG.debug("Upsert past earnings: %s @ %s", ticker, edate.date())


def _write_upcoming(conn: sqlite3.Connection, ticker: str, edate: datetime):
    conn.execute("""
      INSERT OR REPLACE INTO earnings_upcoming
        (ticker, earnings_date, timestamp)
      VALUES (?, ?, ?)
    """, (
      ticker,
      edate.date().isoformat(),
      datetime.utcnow().isoformat()
    ))
    LOG.debug("Upsert upcoming earnings: %s @ %s", ticker, edate.date())


def _fetch_and_store(conn: sqlite3.Connection, tickers: list[str]):
    today         = datetime.utcnow().date()
    past_cutoff   = today - timedelta(days=PAST_WINDOW_DAYS)
    upcoming_cutoff = today + timedelta(days=UPCOMING_WINDOW_DAYS)

    reporting_today = set()
    for tic in tickers:
        try:
            LOG.info("Fetching earnings for %s", tic)
            df = yf.Ticker(tic).get_earnings_dates(limit=30)
            if df is None or df.empty:
                continue

            # normalize index to date
            df.index = pd.to_datetime(df.index).tz_localize(None).date

            # --- past window ---
            past_df = df[(df.index >= past_cutoff) & (df.index <= today)]
            for ed, row in past_df.iterrows():
                surprise = pd.to_numeric(row.get('Surprise(%)'), errors='coerce')
                eps_est  = row.get('EPS Estimate')
                rpt_eps  = row.get('Reported EPS')
                if ed == today:
                    reporting_today.add(tic)
                _write_past(conn, tic, pd.Timestamp(ed), eps_est, rpt_eps, surprise)

            # --- upcoming window ---
            up_df = df[(df.index > today) & (df.index <= upcoming_cutoff)]
            for ed in up_df.index:
                _write_upcoming(conn, tic, pd.Timestamp(ed))

        except Exception:
            LOG.exception("Failed to fetch/store for %s", tic)

    conn.commit()
    return reporting_today


def _render_past_html(conn: sqlite3.Connection, reporting_today: set[str]) -> str:
    today       = datetime.utcnow().date()
    past_cutoff = today - timedelta(days=PAST_WINDOW_DAYS)

    dfp = pd.read_sql_query(
        "SELECT * FROM earnings_past WHERE earnings_date BETWEEN ? AND ? ORDER BY earnings_date DESC",
        conn,
        params=[past_cutoff.isoformat(), today.isoformat()],
        parse_dates=['earnings_date']
    )
    if dfp.empty:
        return f"<p>No earnings in the past {PAST_WINDOW_DAYS} days.</p>"

    dfp['Surprise'] = pd.to_numeric(dfp['surprise_percent'], errors='coerce')
    dfp['Surprise_HTML'] = dfp['Surprise'].map(
        lambda x: f'<span class="{"positive" if x>0 else "negative" if x<0 else ""}">{x:+.2f}%</span>'
        if pd.notna(x) else '-'
    )

    beats = dfp[dfp.Surprise > 0].nlargest(5, 'Surprise')
    misses= dfp[dfp.Surprise < 0].nsmallest(5, 'Surprise')

    header = (
        f"<p>Showing earnings from {past_cutoff} to {today}.</p>"
        + (f"<p><strong>Reporting Today:</strong> {', '.join(sorted(reporting_today))}</p>" if reporting_today else "")
        + "<h3>Top 5 Beats</h3><ul>"
        + "".join(f"<li>{r.ticker}: {r.Surprise:+.2f}%</li>" for _, r in beats.iterrows())
        + "</ul><h3>Top 5 Misses</h3><ul>"
        + "".join(f"<li>{r.ticker}: {r.Surprise:+.2f}%</li>" for _, r in misses.iterrows())
        + "</ul>"
    )

    display = (
        dfp[['ticker','earnings_date','eps_estimate','reported_eps','Surprise_HTML']]
        .rename(columns={
            'ticker':'Ticker',
            'earnings_date':'Date',
            'eps_estimate':'EPS Est',
            'reported_eps':'Reported EPS',
            'Surprise_HTML':'Surprise'
        })
    )

    head = display.head(10).to_html(escape=False, index=False, classes='center-table', border=0)
    if len(display) > 10:
        rest  = display.iloc[10:].to_html(escape=False, index=False, classes='center-table', border=0)
        table = head + f"<details><summary>Show More</summary>{rest}</details>"
    else:
        table = head

    return header + table


def _render_upcoming_html(conn: sqlite3.Connection) -> str:
    today           = datetime.utcnow().date()
    upcoming_cutoff = today + timedelta(days=UPCOMING_WINDOW_DAYS)

    dfu = pd.read_sql_query(
        "SELECT * FROM earnings_upcoming WHERE earnings_date > ? AND earnings_date <= ? ORDER BY earnings_date",
        conn,
        params=[today.isoformat(), upcoming_cutoff.isoformat()],
        parse_dates=['earnings_date']
    )
    if dfu.empty:
        return f"<p>No upcoming earnings in the next {UPCOMING_WINDOW_DAYS} days.</p>"

    dfu['Date'] = pd.to_datetime(dfu['earnings_date']).dt.date
    html = ""
    # group into "Next 7 days" vs "Beyond"
    cutoff = today + timedelta(days=PAST_WINDOW_DAYS)
    early = dfu[dfu.Date <= cutoff]
    later = dfu[dfu.Date > cutoff]

    if not early.empty:
        html += "<h3>Next 7 Days</h3><ul>" + "".join(f"<li>{r.ticker} — {r.Date}</li>" for _, r in early.iterrows()) + "</ul>"
    if not later.empty:
        html += "<details><summary>Beyond 7 Days</summary><ul>" + "".join(f"<li>{r.ticker} — {r.Date}</li>" for _, r in later.iterrows()) + "</ul></details>"

    return html


def generate_earnings_tables():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    yf.set_tz_cache_location(os.path.join(OUTPUT_DIR, 'tz_cache'))

    tickers = modify_tickers(read_tickers('tickers.csv'), is_remote=True)

    with sqlite3.connect(DB_PATH) as conn:
        _ensure_tables(conn)
        reporting_today = _fetch_and_store(conn, tickers)
        past_html       = _render_past_html(conn, reporting_today)
        upcoming_html   = _render_upcoming_html(conn)

    # write out HTML files
    with open(PAST_HTML,     'w', encoding='utf-8') as f: f.write(past_html)
    with open(UPCOMING_HTML, 'w', encoding='utf-8') as f: f.write(upcoming_html)

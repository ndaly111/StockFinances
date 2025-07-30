import os, sqlite3, logging, math
import pandas as pd, yfinance as yf, requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, timezone
from ticker_manager import read_tickers, modify_tickers

# CONFIG
DB_PATH = "Stock Data.db"
OUTPUT_DIR = "charts"
PAST_HTML = os.path.join(OUTPUT_DIR, "earnings_past.html")
UPCOMING_HTML = os.path.join(OUTPUT_DIR, "earnings_upcoming.html")
PAST_WINDOW_DAYS = 7
UPCOMING_WINDOW_DAYS = 90
LOG = logging.getLogger(__name__)

# DB Setup
def _ensure_tables(conn):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS earnings_past (
        ticker TEXT, earnings_date TEXT,
        eps_estimate REAL, reported_eps REAL, surprise_percent REAL,
        timestamp TEXT, PRIMARY KEY(ticker, earnings_date));
    CREATE TABLE IF NOT EXISTS earnings_upcoming (
        ticker TEXT, earnings_date TEXT,
        timestamp TEXT, PRIMARY KEY(ticker, earnings_date));
    CREATE INDEX IF NOT EXISTS idx_past_date ON earnings_past(earnings_date);
    CREATE INDEX IF NOT EXISTS idx_upcoming_date ON earnings_upcoming(earnings_date);
    """)
    conn.commit()

# Utilities
def _clean(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)

def _calc_surprise(est, actual, surprise):
    if surprise is not None:
        return surprise
    if est is None or actual is None or est == 0:
        return None
    return round((actual - est) / abs(est) * 100, 2)

# Fallback Finviz EPS Next Q
def fetch_eps_next_q(ticker):
    url = f'https://finviz.com/quote.ashx?t={ticker}'
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, 'html.parser')
        table = soup.find("table", class_="snapshot-table2")
        for row in table.find_all('tr'):
            cells = row.find_all('td')
            for i in range(0, len(cells), 2):
                if cells[i].text.strip() == 'EPS next Q':
                    eps_next_q = cells[i+1].text.strip()
                    return float(eps_next_q) if eps_next_q.replace('.','',1).replace('-','',1).isdigit() else None
    except Exception as e:
        LOG.warning(f"Finviz EPS fetch failed for {ticker}: {e}")
    return None

# Fetch and Store with Robust Fallback
def _fetch_and_store(conn, tickers):
    today = datetime.now(timezone.utc).date()
    past_cutoff = today - timedelta(days=PAST_WINDOW_DAYS)
    upcoming_cutoff = today + timedelta(days=UPCOMING_WINDOW_DAYS)
    reporting_today = set()

    for tic in tickers:
        try:
            yf_tic = yf.Ticker(tic)
            cal = yf_tic.get_earnings_dates(limit=60)
            if cal is None or cal.empty:
                continue
            cal.index = pd.to_datetime(cal.index).tz_localize(None).date

            for ed, row in cal.iterrows():
                if past_cutoff <= ed <= today:
                    if ed == today:
                        reporting_today.add(tic)
                    est = _clean(row.get('EPS Estimate') or row.get('epsestimate'))
                    actual = _clean(row.get('Reported EPS') or row.get('epsactual'))

                    # Fallback if no EPS data
                    if actual is None:
                        fin = yf_tic.quarterly_financials
                        if not fin.empty:
                            latest_stmt = fin.columns[0].to_pydatetime().date()
                            days_diff = abs((latest_stmt - ed).days)
                            if days_diff <= 15 and 'Net Income' in fin.index:
                                shares = yf_tic.info.get('sharesOutstanding') or yf_tic.info.get('sharesShort') or 1
                                net_income = fin.loc['Net Income'][0]
                                actual = round(net_income / shares, 2) if shares > 0 else None

                    # Fallback for EPS estimate
                    if est is None:
                        est = fetch_eps_next_q(tic)

                    surprise = _calc_surprise(est, actual, None)
                    conn.execute("""
                        INSERT INTO earnings_past
                        (ticker, earnings_date, eps_estimate, reported_eps, surprise_percent, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(ticker, earnings_date) DO UPDATE SET
                        eps_estimate=excluded.eps_estimate,
                        reported_eps=excluded.reported_eps,
                        surprise_percent=excluded.surprise_percent,
                        timestamp=excluded.timestamp;
                    """, (tic, ed.isoformat(), est, actual, surprise, datetime.utcnow().isoformat()))

                elif today < ed <= upcoming_cutoff:
                    conn.execute("""
                        INSERT OR REPLACE INTO earnings_upcoming
                        (ticker, earnings_date, timestamp)
                        VALUES (?, ?, ?);
                    """, (tic, ed.isoformat(), datetime.utcnow().isoformat()))

        except Exception:
            LOG.exception(f"Ticker {tic} failed.")
    conn.commit()
    return reporting_today

# Renderers unchanged (your existing HTML renderers go here unchanged)
# [Keep your existing _render_past_html and _render_upcoming_html here]

# MAIN (unchanged)
def generate_earnings_tables():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    yf.set_tz_cache_location(os.path.join(OUTPUT_DIR, "tz_cache"))
    tickers = modify_tickers(read_tickers("tickers.csv"), is_remote=True)
    with sqlite3.connect(DB_PATH) as conn:
        _ensure_tables(conn)
        reporting_today = _fetch_and_store(conn, tickers)
        past_html = _render_past_html(conn, reporting_today)
        upcoming_html = _render_upcoming_html(conn)
    with open(PAST_HTML, "w", encoding="utf-8") as f:
        f.write(past_html)
    with open(UPCOMING_HTML, "w", encoding="utf-8") as f:
        f.write(upcoming_html)

if __name__ == "__main__":
    generate_earnings_tables()

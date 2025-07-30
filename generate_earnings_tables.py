# ─────────────────────────────────────────────────────────────
#  generate_earnings_tables.py   (29-Jul-2025, Zacks edition)
# ─────────────────────────────────────────────────────────────
import os, sqlite3, logging, math, re, requests
from datetime import datetime, timedelta, timezone

import pandas as pd, yfinance as yf
from bs4 import BeautifulSoup
from ticker_manager import read_tickers, modify_tickers

# CONFIG
DB_PATH, OUTPUT_DIR = "Stock Data.db", "charts"
PAST_HTML  = os.path.join(OUTPUT_DIR, "earnings_past.html")
UPCOMING_HTML = os.path.join(OUTPUT_DIR, "earnings_upcoming.html")
PAST_WINDOW_DAYS, UPCOMING_WINDOW_DAYS = 7, 90
HEADERS = {"User-Agent": "Mozilla/5.0"}
LOG = logging.getLogger(__name__)

# ─── DB helpers ──────────────────────────────────────────────
def _ensure_tables(conn):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS earnings_past (
      ticker TEXT, earnings_date TEXT,
      eps_estimate REAL, reported_eps REAL, surprise_percent REAL,
      timestamp TEXT, PRIMARY KEY(ticker, earnings_date));
    CREATE TABLE IF NOT EXISTS earnings_upcoming (
      ticker TEXT, earnings_date TEXT,
      timestamp TEXT, PRIMARY KEY(ticker, earnings_date));
    CREATE INDEX IF NOT EXISTS idx_past_date    ON earnings_past(earnings_date);
    CREATE INDEX IF NOT EXISTS idx_upcoming_date ON earnings_upcoming(earnings_date);
    """)
    conn.commit()

# ─── utilities ───────────────────────────────────────────────
def _clean(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)

def _calc_surprise(est, act):
    if est is None or act is None or est == 0:
        return None
    return round((act - est) / abs(est) * 100, 2)

# Zacks scraper: current-quarter EPS estimate
_zacks_rx = re.compile(r"Current Qtr\)</td><td.*?>\s*([+-]?\d+\.\d+)")
def _zacks_estimate(tic: str):
    url = f"https://www.zacks.com/stock/quote/{tic}?q={tic}"
    try:
        html = requests.get(url, headers=HEADERS, timeout=10).text
        match = _zacks_rx.search(html)
        if match:
            return float(match.group(1))
    except Exception as e:
        LOG.warning("Zacks scrape failed for %s: %s", tic, e)
    return None

# ─── core fetch/store ───────────────────────────────────────
def _fetch_and_store(conn, tickers):
    today   = datetime.now(timezone.utc).date()
    past_ct = today - timedelta(days=PAST_WINDOW_DAYS)
    up_ct   = today + timedelta(days=UPCOMING_WINDOW_DAYS)
    reporting_today = set()

    for tic in tickers:
        try:
            cal = yf.Ticker(tic).get_earnings_dates(limit=60)
            if cal is None or cal.empty:
                continue
            cal.index = pd.to_datetime(cal.index).tz_localize(None).date
            for ed, row in cal.iterrows():
                est = _clean(row.get("EPS Estimate") or row.get("epsestimate"))
                act = _clean(row.get("Reported EPS") or row.get("epsactual"))

                # fallback: Zacks estimate
                if est is None:
                    est = _zacks_estimate(tic)

                surpr = _calc_surprise(est, act)

                if past_ct <= ed <= today:
                    if ed == today:
                        reporting_today.add(tic)
                    conn.execute("""
                      INSERT INTO earnings_past
                      (ticker, earnings_date, eps_estimate, reported_eps,
                       surprise_percent, timestamp)
                      VALUES (?,?,?,?,?,?)
                      ON CONFLICT(ticker, earnings_date) DO UPDATE SET
                        eps_estimate     = excluded.eps_estimate,
                        reported_eps     = excluded.reported_eps,
                        surprise_percent = excluded.surprise_percent,
                        timestamp        = excluded.timestamp;
                    """, (tic, ed.isoformat(), est, act, surpr,
                          datetime.utcnow().isoformat()))
                elif today < ed <= up_ct:
                    conn.execute("""
                      INSERT OR REPLACE INTO earnings_upcoming
                      (ticker, earnings_date, timestamp)
                      VALUES (?,?,?);
                    """, (tic, ed.isoformat(), datetime.utcnow().isoformat()))
        except Exception:
            LOG.exception("Failed on %s", tic)

    conn.commit()
    return reporting_today

# ─── HTML renderers ──────────────────────────────────────────
def _render_past_html(conn, reporting_today):
    today = datetime.utcnow().date()
    past_ct = today - timedelta(days=PAST_WINDOW_DAYS)
    df = pd.read_sql("SELECT * FROM earnings_past WHERE earnings_date BETWEEN ? AND ? ORDER BY earnings_date DESC",
                     conn, params=[past_ct.isoformat(), today.isoformat()],
                     parse_dates=["earnings_date"])

    if df.empty:
        return f"<p>No earnings in the past {PAST_WINDOW_DAYS} days.</p>"

    df["surprise_percent"] = pd.to_numeric(df["surprise_percent"], errors="coerce")
    df["Surprise_HTML"] = df["surprise_percent"].apply(
        lambda x: "-" if pd.isna(x) else
        f'<span class="{"positive" if x>0 else "negative"}">{x:+.2f}%</span>'
    )

    beats  = df[df.surprise_percent > 0].nlargest(5, "surprise_percent")
    misses = df[df.surprise_percent < 0].nsmallest(5, "surprise_percent")

    header = (
        f"<p>Showing earnings from {past_ct} to {today}.</p>"
        + (f"<p><b>Reporting Today:</b> {', '.join(sorted(reporting_today))}</p>"
           if reporting_today else "")
        + "<h3>Top 5 Beats</h3><ul>"
        + "".join(f"<li>{r.ticker}: {r.surprise_percent:+.2f}%</li>" for _, r in beats.iterrows())
        + "</ul><h3>Top 5 Misses</h3><ul>"
        + "".join(f"<li>{r.ticker}: {r.surprise_percent:+.2f}%</li>" for _, r in misses.iterrows())
        + "</ul>"
    )

    display = (df[["ticker","earnings_date","eps_estimate","reported_eps","Surprise_HTML"]]
               .rename(columns={"ticker":"Ticker","earnings_date":"Date",
                                "eps_estimate":"EPS Est","reported_eps":"Reported EPS",
                                "Surprise_HTML":"Surprise"}))

    head = display.head(10).to_html(index=False, escape=False, classes="center-table", border=0)
    if len(display) > 10:
        rest  = display.iloc[10:].to_html(index=False, escape=False, classes="center-table", border=0)
        table = head + f"<details><summary>Show More</summary>{rest}</details>"
    else:
        table = head
    return header + table

def _render_upcoming_html(conn):
    today = datetime.utcnow().date()
    up_ct = today + timedelta(days=UPCOMING_WINDOW_DAYS)
    df = pd.read_sql("SELECT * FROM earnings_upcoming WHERE earnings_date>? AND earnings_date<=? ORDER BY earnings_date",
                     conn, params=[today.isoformat(), up_ct.isoformat()],
                     parse_dates=["earnings_date"])
    if df.empty:
        return f"<p>No upcoming earnings in the next {UPCOMING_WINDOW_DAYS} days.</p>"

    df["Date"] = df["earnings_date"].dt.date
    early_ct = today + timedelta(days=PAST_WINDOW_DAYS)
    early, later = df[df.Date <= early_ct], df[df.Date > early_ct]

    html = ""
    if not early.empty:
        html += "<h3>Next 7 Days</h3><ul>" + \
                "".join(f"<li>{r.ticker} — {r.Date}</li>" for _, r in early.iterrows()) + \
                "</ul>"
    if not later.empty:
        html += ("<details><summary>Beyond 7 Days</summary><ul>" +
                 "".join(f"<li>{r.ticker} — {r.Date}</li>" for _, r in later.iterrows()) +
                 "</ul></details>")
    return html

# ─── main ────────────────────────────────────────────────────
def generate_earnings_tables():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    yf.set_tz_cache_location(os.path.join(OUTPUT_DIR, "tz_cache"))

    tickers = modify_tickers(read_tickers("tickers.csv"), is_remote=True)

    with sqlite3.connect(DB_PATH) as conn:
        _ensure_tables(conn)
        reporting_today = _fetch_and_store(conn, tickers)
        past_html       = _render_past_html(conn, reporting_today)
        upcoming_html   = _render_upcoming_html(conn)

    with open(PAST_HTML, "w", encoding="utf-8")     as f: f.write(past_html)
    with open(UPCOMING_HTML, "w", encoding="utf-8") as f: f.write(upcoming_html)

if __name__ == "__main__":
    generate_earnings_tables()

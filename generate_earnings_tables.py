# ──────────────────────────────────────────────────────────────
#  generate_earnings_tables.py   (full working version, 29-Jul-2025)
# ──────────────────────────────────────────────────────────────
"""
Builds / refreshes earnings tables for the dashboard.

Logic order
-----------
1. Try Yahoo Finance calendar (get_earnings_dates).
2. If missing EPS: check latest quarterly statement ±15 days for actual EPS.
3. If missing estimate: scrape Finviz “EPS next Q”.
4. Store to SQLite and render HTML tables.

Requires:  requests, beautifulsoup4
"""

import os, sqlite3, logging, math, requests
from datetime import datetime, timedelta, timezone

import pandas as pd, yfinance as yf
from bs4 import BeautifulSoup

from ticker_manager import read_tickers, modify_tickers   # unchanged

# ─── CONFIG ─────────────────────────────────────────────
DB_PATH              = "Stock Data.db"
OUTPUT_DIR           = "charts"
PAST_HTML            = os.path.join(OUTPUT_DIR, "earnings_past.html")
UPCOMING_HTML        = os.path.join(OUTPUT_DIR, "earnings_upcoming.html")
PAST_WINDOW_DAYS     = 7
UPCOMING_WINDOW_DAYS = 90
HEADERS              = {'User-Agent': 'Mozilla/5.0'}
LOG                  = logging.getLogger(__name__)

# ─── DB SET-UP ──────────────────────────────────────────
def _ensure_tables(conn):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS earnings_past (
        ticker TEXT,
        earnings_date TEXT,
        eps_estimate REAL,
        reported_eps REAL,
        surprise_percent REAL,
        timestamp TEXT,
        PRIMARY KEY(ticker, earnings_date));

    CREATE TABLE IF NOT EXISTS earnings_upcoming (
        ticker TEXT,
        earnings_date TEXT,
        timestamp TEXT,
        PRIMARY KEY(ticker, earnings_date));

    CREATE INDEX IF NOT EXISTS idx_past_date    ON earnings_past(earnings_date);
    CREATE INDEX IF NOT EXISTS idx_upcoming_date ON earnings_upcoming(earnings_date);
    """)
    conn.commit()

# ─── LITTLE HELPERS ────────────────────────────────────
def _clean(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)

def _calc_surprise(est, actual, supplied):
    if supplied is not None:
        return supplied
    if est is None or actual is None or est == 0:
        return None
    return round((actual - est) / abs(est) * 100, 2)

def _finviz_eps_next_q(tic):
    """Scrape EPS next Q from Finviz snapshot table."""
    url = f"https://finviz.com/quote.ashx?t={tic}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        table = soup.find("table", class_="snapshot-table2")
        for row in table.find_all("tr"):
            cells = row.find_all("td")
            for i in range(0, len(cells), 2):
                if cells[i].text.strip() == "EPS next Q":
                    txt = cells[i+1].text.strip()
                    return float(txt) if txt.replace('.', '', 1).replace('-', '', 1).isdigit() else None
    except Exception as e:
        LOG.warning("Finviz scrape failed for %s: %s", tic, e)
    return None

# ─── FETCH + STORE ─────────────────────────────────────
def _fetch_and_store(conn, tickers):
    today           = datetime.now(timezone.utc).date()
    past_cutoff     = today - timedelta(days=PAST_WINDOW_DAYS)
    upcoming_cutoff = today + timedelta(days=UPCOMING_WINDOW_DAYS)
    reporting_today = set()

    for tic in tickers:
        try:
            yf_tic = yf.Ticker(tic)
            cal    = yf_tic.get_earnings_dates(limit=60)
            if cal is None or cal.empty:
                continue
            cal.index = pd.to_datetime(cal.index).tz_localize(None).date

            for ed, row in cal.iterrows():
                est    = _clean(row.get("EPS Estimate")   or row.get("epsestimate"))
                actual = _clean(row.get("Reported EPS")   or row.get("epsactual"))
                supplied_surprise = _clean(
                    pd.to_numeric(
                        row.get("Surprise (%)") or row.get("Surprise(%)") or
                        row.get("surprise(%)")  or row.get("epssurprisepct"),
                        errors="coerce"
                    )
                )

                # Fallback: quarterly statement within ±15 days
                if actual is None:
                    fin = yf_tic.quarterly_financials
                    if not fin.empty:
                        stmt_date = fin.columns[0].to_pydatetime().date()
                        if abs((stmt_date - ed).days) <= 15 and 'Net Income' in fin.index:
                            net_income = fin.loc['Net Income'][0]
                            shares     = yf_tic.info.get('sharesOutstanding') or 0
                            if shares:
                                actual = round(net_income / shares, 2)

                # Fallback: Finviz estimate
                if est is None:
                    est = _finviz_eps_next_q(tic)

                surprise = _calc_surprise(est, actual, supplied_surprise)

                if past_cutoff <= ed <= today:
                    if ed == today:
                        reporting_today.add(tic)
                    conn.execute("""
                        INSERT INTO earnings_past
                        (ticker, earnings_date, eps_estimate, reported_eps,
                         surprise_percent, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(ticker, earnings_date) DO UPDATE SET
                          eps_estimate     = excluded.eps_estimate,
                          reported_eps     = excluded.reported_eps,
                          surprise_percent = excluded.surprise_percent,
                          timestamp        = excluded.timestamp;
                    """, (tic, ed.isoformat(), est, actual, surprise,
                          datetime.utcnow().isoformat()))
                elif today < ed <= upcoming_cutoff:
                    conn.execute("""
                        INSERT OR REPLACE INTO earnings_upcoming
                        (ticker, earnings_date, timestamp)
                        VALUES (?, ?, ?);
                    """, (tic, ed.isoformat(), datetime.utcnow().isoformat()))

        except Exception:
            LOG.exception("Failed on %s", tic)

    conn.commit()
    return reporting_today

# ─── HTML RENDERERS ────────────────────────────────────
def _render_past_html(conn, reporting_today):
    today       = datetime.utcnow().date()
    past_cutoff = today - timedelta(days=PAST_WINDOW_DAYS)
    df = pd.read_sql_query(
        "SELECT * FROM earnings_past WHERE earnings_date BETWEEN ? AND ? "
        "ORDER BY earnings_date DESC",
        conn,
        params=[past_cutoff.isoformat(), today.isoformat()],
        parse_dates=["earnings_date"]
    )

    if df.empty:
        return f"<p>No earnings in the past {PAST_WINDOW_DAYS} days.</p>"

    df["Surprise_HTML"] = df["surprise_percent"].apply(
        lambda x: "-" if pd.isna(x) else
        f'<span class="{"positive" if x>0 else "negative"}">{x:+.2f}%</span>'
    )

    beats  = df[df.surprise_percent > 0].nlargest(5, "surprise_percent")
    misses = df[df.surprise_percent < 0].nsmallest(5, "surprise_percent")

    header = (
        f"<p>Showing earnings from {past_cutoff} to {today}.</p>"
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

    head = display.head(10).to_html(index=False, escape=False,
                                    classes="center-table", border=0)
    if len(display) > 10:
        rest = display.iloc[10:].to_html(index=False, escape=False,
                                         classes="center-table", border=0)
        table = head + f"<details><summary>Show More</summary>{rest}</details>"
    else:
        table = head
    return header + table

def _render_upcoming_html(conn):
    today = datetime.utcnow().date()
    upcoming_cutoff = today + timedelta(days=UPCOMING_WINDOW_DAYS)
    df = pd.read_sql_query(
        "SELECT * FROM earnings_upcoming WHERE earnings_date>? AND earnings_date<=? "
        "ORDER BY earnings_date",
        conn,
        params=[today.isoformat(), upcoming_cutoff.isoformat()],
        parse_dates=["earnings_date"]
    )

    if df.empty:
        return f"<p>No upcoming earnings in the next {UPCOMING_WINDOW_DAYS} days.</p>"

    df["Date"] = df["earnings_date"].dt.date
    early_cut  = today + timedelta(days=PAST_WINDOW_DAYS)
    early, later = df[df.Date <= early_cut], df[df.Date > early_cut]

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

# ─── MAIN ENTRYPOINT ──────────────────────────────────
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
# ──────────────────────────────────────────────────────────────

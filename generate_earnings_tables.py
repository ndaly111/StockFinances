"""
generate_earnings_tables.py
Builds / refreshes earnings tables and HTML for your dashboard.

Key fixes
─────────
1. Handles yfinance's new lower-case column names.
2. Always overwrites prior NULL rows in the DB.
3. Converts NaN → None before writing, so HTML shows blanks instead of “None”.
"""

import os, sqlite3, logging, math, pandas as pd, yfinance as yf
from datetime import datetime, timedelta, timezone
from ticker_manager import read_tickers, modify_tickers  # ← unchanged

# ────── CONFIG ──────
DB_PATH = "Stock Data.db"
OUTPUT_DIR = "charts"
PAST_HTML = os.path.join(OUTPUT_DIR, "earnings_past.html")
UPCOMING_HTML = os.path.join(OUTPUT_DIR, "earnings_upcoming.html")
PAST_WINDOW_DAYS = 7
UPCOMING_WINDOW_DAYS = 90
LOG = logging.getLogger(__name__)


# ────────────────── DB HELPERS ──────────────────
def _ensure_tables(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.executescript(
        """
      CREATE TABLE IF NOT EXISTS earnings_past (
        ticker TEXT,
        earnings_date TEXT,
        eps_estimate REAL,
        reported_eps REAL,
        surprise_percent REAL,
        timestamp TEXT,
        PRIMARY KEY(ticker, earnings_date)
      );
      CREATE TABLE IF NOT EXISTS earnings_upcoming (
        ticker TEXT,
        earnings_date TEXT,
        timestamp TEXT,
        PRIMARY KEY(ticker, earnings_date)
      );
      CREATE INDEX IF NOT EXISTS idx_past_date    ON earnings_past(earnings_date);
      CREATE INDEX IF NOT EXISTS idx_upcoming_date ON earnings_upcoming(earnings_date);
    """
    )
    conn.commit()


def _upsert_past(conn, tic, edate, est, actual, surpr):
    conn.execute(
        """
        INSERT INTO earnings_past
          (ticker, earnings_date, eps_estimate, reported_eps,
           surprise_percent, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker, earnings_date) DO UPDATE SET
            eps_estimate     = excluded.eps_estimate,
            reported_eps     = excluded.reported_eps,
            surprise_percent = excluded.surprise_percent,
            timestamp        = excluded.timestamp;
    """,
        (tic, edate.isoformat(), est, actual, surpr, datetime.utcnow().isoformat()),
    )


def _upsert_upcoming(conn, tic, edate):
    conn.execute(
        """
        INSERT OR REPLACE INTO earnings_upcoming
          (ticker, earnings_date, timestamp)
        VALUES (?, ?, ?)
    """,
        (tic, edate.isoformat(), datetime.utcnow().isoformat()),
    )


# ────────────────── SMALL UTILITIES ──────────────────
def _pick(row, *names):
    """Return the first non-missing value among the provided column names."""
    for n in names:
        if n in row and pd.notna(row[n]):
            return row[n]
    return None


def _to_pct(x):
    return pd.to_numeric(x, errors="coerce")


def _clean(val):
    """Convert NaN → None; cast numbers to float; leave strings untouched."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    return float(val) if isinstance(val, (int, float)) else val


def _calc_surprise(est, actual, surprise):
    """Return surprise pct, computing if missing."""
    if surprise is not None:
        return surprise
    if est is None or actual is None or est == 0:
        return None
    return round((actual - est) / abs(est) * 100, 2)


# ────────────────── FETCH & STORE ──────────────────
def _fetch_and_store(conn: sqlite3.Connection, tickers: list[str]):
    today = datetime.now(timezone.utc).date()
    past_cutoff = today - timedelta(days=PAST_WINDOW_DAYS)
    upcoming_cutoff = today + timedelta(days=UPCOMING_WINDOW_DAYS)
    reporting_today = set()

    for tic in tickers:
        try:
            cal = yf.Ticker(tic).get_earnings_dates(limit=60)  # past + future
            if cal is None or cal.empty:
                continue
            cal.index = pd.to_datetime(cal.index).tz_localize(None).date

            # ----- Past & today -----
            for ed, row in cal.iterrows():
                if past_cutoff <= ed <= today:
                    if ed == today:
                        reporting_today.add(tic)
                    _upsert_past(
                        conn,
                        tic,
                        ed,
                        _clean(_pick(row, "EPS Estimate", "epsestimate")),
                        _clean(_pick(row, "Reported EPS", "epsactual")),
                        _calc_surprise(
                            _clean(_pick(row, "EPS Estimate", "epsestimate")),
                            _clean(_pick(row, "Reported EPS", "epsactual")),
                            _clean(
                                _to_pct(
                                    _pick(
                                        row,
                                        "Surprise (%)",
                                        "Surprise(%)",
                                        "surprise(%)",
                                        "epssurprisepct",
                                    )
                                )
                            ),
                        ),
                    )

            # ----- Upcoming -----
            for ed in cal.index[(cal.index > today) & (cal.index <= upcoming_cutoff)]:
                _upsert_upcoming(conn, tic, ed)

            # ----- Back-fill via history -----
            hist = yf.Ticker(tic).get_earnings_history()
            if hist is not None and not hist.empty:
                for _, h in hist.iterrows():
                    ed = pd.to_datetime(h["startdatetime"]).date()
                    if past_cutoff <= ed <= today:
                        _upsert_past(
                            conn,
                            tic,
                            ed,
                            _clean(_pick(h, "epsEstimate", "epsestimate")),
                            _clean(_pick(h, "epsActual", "epsactual")),
                            _calc_surprise(
                                _clean(_pick(h, "epsEstimate", "epsestimate")),
                                _clean(_pick(h, "epsActual", "epsactual")),
                                _clean(
                                    _to_pct(
                                        _pick(
                                            h,
                                            "surprisePercent",
                                            "surprisepercent",
                                            "epssurprisepct",
                                        )
                                    )
                                ),
                            ),
                        )

        except Exception:
            LOG.exception("Failed on %s", tic)

    conn.commit()
    return reporting_today


# ────────────────── RENDERERS (unchanged) ──────────────────
def _render_past_html(conn, reporting_today):
    today = datetime.utcnow().date()
    past_cutoff = today - timedelta(days=PAST_WINDOW_DAYS)
    df = pd.read_sql_query(
        "SELECT * FROM earnings_past WHERE earnings_date BETWEEN ? AND ? "
        "ORDER BY earnings_date DESC",
        conn,
        params=[past_cutoff.isoformat(), today.isoformat()],
        parse_dates=["earnings_date"],
    )
    if df.empty:
        return f"<p>No earnings in the past {PAST_WINDOW_DAYS} days.</p>"

    df["Surprise"] = pd.to_numeric(df["surprise_percent"], errors="coerce")
    df["Surprise_HTML"] = df["Surprise"].map(
        lambda x: (
            "-"
            if pd.isna(x)
            else f'<span class="{"positive" if x>0 else "negative"}">{x:+.2f}%</span>'
        )
    )

    beats = df[df.Surprise > 0].nlargest(5, "Surprise")
    misses = df[df.Surprise < 0].nsmallest(5, "Surprise")

    header = (
        f"<p>Showing earnings from {past_cutoff} to {today}.</p>"
        + (
            f"<p><b>Reporting Today:</b> {', '.join(sorted(reporting_today))}</p>"
            if reporting_today
            else ""
        )
        + "<h3>Top 5 Beats</h3><ul>"
        + "".join(
            f"<li>{r.ticker}: {r.Surprise:+.2f}%</li>" for _, r in beats.iterrows()
        )
        + "</ul><h3>Top 5 Misses</h3><ul>"
        + "".join(
            f"<li>{r.ticker}: {r.Surprise:+.2f}%</li>" for _, r in misses.iterrows()
        )
        + "</ul>"
    )

    display = df[
        ["ticker", "earnings_date", "eps_estimate", "reported_eps", "Surprise_HTML"]
    ].rename(
        columns={
            "ticker": "Ticker",
            "earnings_date": "Date",
            "eps_estimate": "EPS Est",
            "reported_eps": "Reported EPS",
            "Surprise_HTML": "Surprise",
        }
    )

    head = display.head(10).to_html(
        index=False, escape=False, classes="center-table", border=0
    )
    if len(display) > 10:
        rest = display.iloc[10:].to_html(
            index=False, escape=False, classes="center-table", border=0
        )
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
        parse_dates=["earnings_date"],
    )
    if df.empty:
        return f"<p>No upcoming earnings in the next {UPCOMING_WINDOW_DAYS} days.</p>"

    df["Date"] = pd.to_datetime(df["earnings_date"]).dt.date
    early_cut = today + timedelta(days=PAST_WINDOW_DAYS)
    early, later = df[df.Date <= early_cut], df[df.Date > early_cut]

    html = ""
    if not early.empty:
        html += (
            "<h3>Next 7 Days</h3><ul>"
            + "".join(f"<li>{r.ticker} — {r.Date}</li>" for _, r in early.iterrows())
            + "</ul>"
        )
    if not later.empty:
        html += (
            "<details><summary>Beyond 7 Days</summary><ul>"
            + "".join(f"<li>{r.ticker} — {r.Date}</li>" for _, r in later.iterrows())
            + "</ul></details>"
        )
    return html


# ────────────────────────── MAIN ──────────────────────────
def generate_earnings_tables():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
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


# Mini-main entry-point
if __name__ == "__main__":
    generate_earnings_tables()

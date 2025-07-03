# forward_estimates_scraper.py – Robust + Fast version 2025-07-03
# ─────────────────────────────────────────────────────────────────────
"""
Scrapes Zacks annual consensus EPS & revenue forecasts and stores them
into a SQLite DB (`ForwardFinancialData`).  Safe for multithreaded use.
"""
# ─────────────────────────────────────────────────────────────────────
import re, calendar, logging, sqlite3, time, traceback
from datetime import datetime
from contextlib import contextmanager
from typing import List, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
from sqlite3 import OperationalError

# ─────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────
DB_PATH    = "Stock Data.db"
TABLE_NAME = "ForwardFinancialData"
HEADERS = {
    "User-Agent":
        ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
         "AppleWebKit/537.36 (KHTML, like Gecko) "
         "Chrome/124.0.0.0 Safari/537.3")
}
NUM_RE = re.compile(r"([0-9.\-]+)([MBT]?)")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ————————————————————————————————————————————————————————————————
# SQLite helpers
# ————————————————————————————————————————————————————————————————
def _ensure_table(db_path: str, table: str) -> None:
    """Create table & columns exactly once per run."""
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")                # better concurrency
        cur = conn.cursor()
        # Base table with primary-key on (Ticker, Date)
        cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {table} (
            Ticker TEXT NOT NULL,
            Date   TEXT NOT NULL,
            ForwardEPS   REAL,
            ForwardRevenue REAL,
            LastUpdated TEXT,
            ForwardEPSAnalysts INTEGER,
            ForwardRevenueAnalysts INTEGER,
            PRIMARY KEY (Ticker, Date)
        );
        """)
        # Add new columns if they didn’t exist in an older DB
        cur.execute(f"PRAGMA table_info({table})")
        cols = {c[1] for c in cur.fetchall()}
        for col in ("ForwardEPSAnalysts", "ForwardRevenueAnalysts"):
            if col not in cols:
                cur.execute(f"ALTER TABLE {table} ADD COLUMN {col} INTEGER")
        conn.commit()

@contextmanager
def _connect(db_path: str):
    """SQLite connection that retries once on 'database is locked'."""
    try:
        conn = sqlite3.connect(db_path, timeout=30, isolation_level=None)  # autocommit
        yield conn
    except OperationalError as e:
        if "database is locked" in str(e):
            time.sleep(1)
            conn = sqlite3.connect(db_path, timeout=30, isolation_level=None)
            yield conn
        else:
            raise
    finally:
        try:
            conn.close()
        except Exception:
            pass

# ————————————————————————————————————————————————————————————————
# Utility fns
# ————————————————————————————————————————————————————————————————
def _last_day(date_str: str) -> str:
    """Convert '12/2025' → '2025-12-31'."""
    try:
        m, y = map(int, date_str.split("/"))
        return f"{y}-{m:02d}-{calendar.monthrange(y, m)[1]:02d}"
    except Exception:
        return ""

def _to_number(series: pd.Series) -> pd.Series:
    def _conv(val: str) -> float:
        if not isinstance(val, str):
            return float(val or 0)
        m = NUM_RE.match(val.replace(",", "").strip())
        if not m:
            return 0.0
        num, unit = m.groups()
        mult = {"": 1, "M": 1e6, "B": 1e9, "T": 1e12}[unit]
        return float(num) * mult
    return series.map(_conv)

# ————————————————————————————————————————————————————————————————
# Scraping core
# ————————————————————————————————————————————————————————————————
def _fetch_html(ticker: str, session: requests.Session) -> Optional[BeautifulSoup]:
    url = f"https://www.zacks.com/stock/quote/{ticker.replace('-', '.')}/detailed-earning-estimates"
    try:
        r = session.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        return BeautifulSoup(r.text, "lxml")
    except (requests.RequestException, Exception) as e:
        logging.warning(f"{ticker}: HTTP error – {e}")
        return None

def _parse(soup: BeautifulSoup) -> Optional[pd.DataFrame]:
    sections = soup.select("section#detailed_earnings_estimates")
    if len(sections) < 2:
        return None
    # Grab first table in each section with pandas (fast C parser)
    sales_df , earnings_df = (pd.read_html(str(sec.find("table")), flavor="lxml")[0]
                              for sec in sections[:2])
    # Ensure at least two forecast columns exist
    if sales_df.shape[1] < 5 or earnings_df.shape[1] < 5:
        return None

    # Analyst counts
    def _extract_analysts(df) -> Optional[int]:
        s = df.loc[df.iloc[:,0].str.contains("# of Estimates", na=False), 1]
        if s.empty: return None
        return int(str(s.iat[0]).replace(",", "") or 0)

    rev_analysts = _extract_analysts(sales_df)
    eps_analysts = _extract_analysts(earnings_df)
    if rev_analysts is None or eps_analysts is None:
        return None

    # Consensus rows
    cons_rev = sales_df.loc[sales_df.iloc[:,0].str.contains("Consensus", na=False)]
    cons_eps = earnings_df.loc[earnings_df.iloc[:,0].str.contains("Consensus", na=False)]
    if cons_rev.empty or cons_eps.empty:
        return None
    cons_rev, cons_eps = cons_rev.iloc[0], cons_eps.iloc[0]

    # Period headers
    this_hdr, next_hdr = sales_df.columns[3:5]
    this_date = _last_day(this_hdr.split("(")[-1].rstrip(")"))
    next_date = _last_day(next_hdr.split("(")[-1].rstrip(")"))
    if not this_date or not next_date:
        return None

    data = pd.DataFrame({
        "Period"                 : ["Current Year", "Next Year"],
        "Date"                   : [this_date, next_date],
        "ForwardRevenue"         : _to_number(cons_rev.iloc[3:5]).values,
        "ForwardEPS"             : pd.to_numeric(cons_eps.iloc[3:5].str.replace(",",""),
                                                 errors="coerce").values,
        "ForwardRevenueAnalysts" : rev_analysts,
        "ForwardEPSAnalysts"     : eps_analysts
    })
    return data

def scrape_annual_estimates(ticker: str, session: Optional[requests.Session]=None) -> pd.DataFrame:
    s = session or requests.Session()
    soup = _fetch_html(ticker, s)
    if soup is None:
        return pd.DataFrame()
    return _parse(soup) or pd.DataFrame()

# ————————————————————————————————————————————————————————————————
# Storage
# ————————————————————————————————————————————————————————————————
def _store(df: pd.DataFrame, ticker: str) -> None:
    with _connect(DB_PATH) as conn:
        cur = conn.cursor()
        # Remove existing rows for the same forecast dates
        cur.execute(f"DELETE FROM {TABLE_NAME} WHERE Ticker = ?", (ticker,))
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for _, r in df.iterrows():
            cur.execute(f"""
            INSERT INTO {TABLE_NAME}
              (Ticker, Date, ForwardEPS, ForwardRevenue, LastUpdated,
               ForwardEPSAnalysts, ForwardRevenueAnalysts)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(Ticker, Date) DO UPDATE SET
              ForwardEPS=excluded.ForwardEPS,
              ForwardRevenue=excluded.ForwardRevenue,
              ForwardEPSAnalysts=excluded.ForwardEPSAnalysts,
              ForwardRevenueAnalysts=excluded.ForwardRevenueAnalysts,
              LastUpdated=excluded.LastUpdated;
            """, (
                ticker,
                r["Date"],
                r["ForwardEPS"],
                r["ForwardRevenue"],
                now,
                r["ForwardEPSAnalysts"],
                r["ForwardRevenueAnalysts"],
            ))
        conn.commit()
    logging.info(f"{ticker}: stored {len(df)} rows")

# ————————————————————————————————————————————————————————————————
# Public API
# ————————————————————————————————————————————————————————————————
def scrape_forward_data(ticker: str) -> None:
    """Scrape & store a single ticker."""
    df = scrape_annual_estimates(ticker, SESSION)
    if df.empty:
        logging.info(f"{ticker}: no data scraped")
        return
    _store(df, ticker)

# ————————————————————————————————————————————————————————————————
# Batch helper (threads)
# ————————————————————————————————————————————————————————————————
def scrape_forward_data_batch(tickers: List[str], max_workers: int = 6) -> None:
    """I/O-bound => threads are fine.  Each thread gets its own Session."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    def _worker(tkr: str):
        with requests.Session() as sess:
            df = scrape_annual_estimates(tkr, sess)
            if not df.empty:
                _store(df, tkr)
            else:
                logging.info(f"{tkr}: skipped (no data)")
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut_to_tkr = {ex.submit(_worker, t): t for t in tickers}
        for fut in as_completed(fut_to_tkr):
            tkr = fut_to_tkr[fut]
            try:
                fut.result()
                logging.info(f"{tkr}: done")
            except Exception:
                logging.error(f"{tkr}: FAILED\n{traceback.format_exc()}")

# ————————————————————————————————————————————————————————————————
# Init & example usage
# ————————————————————————————————————————————————————————————————
if __name__ == "__main__":
    _ensure_table(DB_PATH, TABLE_NAME)         # set up once

    # Single-ticker example
    scrape_forward_data("AAPL")

    # Batch example – uncomment if needed
    # scrape_forward_data_batch(["AAPL", "MSFT", "GOOG", "AMZN"], max_workers=4)

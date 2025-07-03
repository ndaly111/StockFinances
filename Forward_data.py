# Forward_data.py  – 2025-07-03  (safe analysts lookup)
# ───────────────────────────────────────────────────────────────────
"""
Scrape Zacks annual consensus EPS / revenue forecasts and store them in
SQLite.  Safe for single-ticker or multithreaded batch use.
"""
# ───────────────────────────────────────────────────────────────────
import re, calendar, logging, sqlite3, time, traceback, math
from contextlib import contextmanager
from datetime import datetime
from typing import List, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
from sqlite3 import OperationalError

# ───────────────────────────────────────────────────────────────────
# Config
# ───────────────────────────────────────────────────────────────────
DB_PATH    = "Stock Data.db"
TABLE_NAME = "ForwardFinancialData"
HEADERS = {
    "User-Agent":
        ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
         "AppleWebKit/537.36 (KHTML, like Gecko) "
         "Chrome/124.0.0.0 Safari/537.3")
}
NUM_RE  = re.compile(r"([0-9.\-]+)([MBT]?)")
SESSION = requests.Session()

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s: %(message)s")

# ───────────────────────────────────────────────────────────────────
# SQLite helpers
# ───────────────────────────────────────────────────────────────────
def _ensure_table() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
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
        conn.commit()

@contextmanager
def _connect():
    try:
        conn = sqlite3.connect(DB_PATH, timeout=30, isolation_level=None)
        yield conn
    except OperationalError as e:
        if "database is locked" in str(e):
            time.sleep(1)
            conn = sqlite3.connect(DB_PATH, timeout=30, isolation_level=None)
            yield conn
        else:
            raise
    finally:
        try:
            conn.close()
        except Exception:
            pass

# ───────────────────────────────────────────────────────────────────
# Utility
# ───────────────────────────────────────────────────────────────────
def _last_day(date_str: str) -> str:
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
        return float(num) * {"":1,"M":1e6,"B":1e9,"T":1e12}[unit]
    return series.map(_conv)

# ───────────────────────────────────────────────────────────────────
# Scraping
# ───────────────────────────────────────────────────────────────────
def _fetch_html(ticker: str, session: requests.Session) -> Optional[BeautifulSoup]:
    url = f"https://www.zacks.com/stock/quote/{ticker.replace('-', '.')}/detailed-earning-estimates"
    try:
        r = session.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        return BeautifulSoup(r.text, "lxml")
    except requests.RequestException as e:
        logging.warning(f"{ticker}: HTTP error – {e}")
        return None

def _parse(soup: BeautifulSoup) -> Optional[pd.DataFrame]:
    sections = soup.select("section#detailed_earnings_estimates")
    if len(sections) < 2:
        return None

    sales_df , earnings_df = (pd.read_html(str(sec.find("table")), flavor="lxml")[0]
                              for sec in sections[:2])

    if sales_df.shape[1] < 5 or earnings_df.shape[1] < 5:
        return None

    # ————— safe analyst-count extractor —————
    def _analysts(df) -> Optional[int]:
        mask = df.iloc[:, 0].astype(str).str.contains("# of Estimates", na=False)
        if not mask.any():
            return None
        # use integer-pos .iloc to avoid KeyError when columns aren’t numeric
        val = str(df.loc[mask].iloc[0, 1]).replace(",", "")
        return int(val) if val.isdigit() else None

    rev_analysts, eps_analysts = _analysts(sales_df), _analysts(earnings_df)
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
    this_date, next_date = _last_day(this_hdr.split("(")[-1].rstrip(")")), _last_day(next_hdr.split("(")[-1].rstrip(")"))
    if not this_date or not next_date:
        return None

    data = pd.DataFrame({
        "Period"                 : ["Current Year","Next Year"],
        "Date"                   : [this_date,next_date],
        "ForwardRevenue"         : _to_number(cons_rev.iloc[3:5]).values,
        "ForwardEPS"             : pd.to_numeric(cons_eps.iloc[3:5].str.replace(",",""), errors="coerce").values,
        "ForwardRevenueAnalysts" : rev_analysts,
        "ForwardEPSAnalysts"     : eps_analysts
    })
    return data

def scrape_annual_estimates(ticker: str, session: Optional[requests.Session]=None) -> pd.DataFrame:
    s = session or requests.Session()
    soup = _fetch_html(ticker, s)
    return _parse(soup) or pd.DataFrame()

# ───────────────────────────────────────────────────────────────────
# Storage
# ───────────────────────────────────────────────────────────────────
def _store(df: pd.DataFrame, ticker: str) -> None:
    with _connect() as conn:
        cur = conn.cursor()
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
                ticker, r["Date"], r["ForwardEPS"], r["ForwardRevenue"],
                now, r["ForwardEPSAnalysts"], r["ForwardRevenueAnalysts"]
            ))
        conn.commit()
    logging.info(f"{ticker}: stored {len(df)} rows")

# ───────────────────────────────────────────────────────────────────
# Public API
# ───────────────────────────────────────────────────────────────────
def scrape_forward_data(ticker: str) -> None:
    _ensure_table()
    df = scrape_annual_estimates(ticker, SESSION)
    if df.empty:
        logging.info(f"{ticker}: no data scraped")
        return
    _store(df, ticker)

# ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    scrape_forward_data("AAPL")

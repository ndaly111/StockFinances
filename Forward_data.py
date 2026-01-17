# Forward_data.py  – 2025-07-03  (fast + robust)
# ───────────────────────────────────────────────────────────────────────
"""
Scrapes Zacks ‘Detailed Earnings Estimates’ (annual consensus EPS &
revenue) and stores results in the SQLite table ForwardFinancialData.

• Re-uses a single requests.Session (faster).
• Parses tables with pandas.read_html / lxml (10× faster than BS loops).
• Handles M/B/T unit suffixes vector-wise.
• Safe against empty analyst rows & layout drift.
• Thread-safe batch helper included.
"""
# ───────────────────────────────────────────────────────────────────────
import re, calendar, logging, sqlite3, time, traceback
from contextlib import contextmanager
from datetime import datetime
from typing import List, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
from sqlite3 import OperationalError

# ───────────────────────────────────────────────────────────────────────
# Config
# ───────────────────────────────────────────────────────────────────────
DB_PATH    = "Stock Data.db"
TABLE_NAME = "ForwardFinancialData"
FY_HIST_TABLE = "Forward_EPS_FY_History"

HEADERS = {
    "User-Agent":
        ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
         "AppleWebKit/537.36 (KHTML, like Gecko) "
         "Chrome/124.0.0.0 Safari/537.3")
}
NUM_RE  = re.compile(r"([0-9.\-]+)\s*([MBT]?)")   # fast regex
SESSION = requests.Session()

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s: %(message)s")

# ───────────────────────────────────────────────────────────────────────
# SQLite helpers
# ───────────────────────────────────────────────────────────────────────
def _ensure_table(
    db_path: str = DB_PATH,
    table_name: str = TABLE_NAME,
    conn: sqlite3.Connection | None = None,
) -> None:
    """Create tables once per run. If conn is provided, use it (avoid extra connections)."""
    if conn is None:
        with sqlite3.connect(db_path) as created_conn:
            created_conn.execute("PRAGMA journal_mode=WAL")
            created_conn.execute("PRAGMA busy_timeout=30000")
            _ensure_table(db_path=db_path, table_name=table_name, conn=created_conn)
            created_conn.commit()
        return

    try:
        if not getattr(conn, "in_transaction", False):
            conn.execute("PRAGMA journal_mode=WAL")
    except Exception:
        pass
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
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
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {FY_HIST_TABLE} (
            date_recorded TEXT NOT NULL,
            ticker        TEXT NOT NULL,
            period_end    TEXT NOT NULL,
            period_label  TEXT,
            forward_eps   REAL,
            eps_analysts  INTEGER,
            source        TEXT,
            PRIMARY KEY (date_recorded, ticker, period_end)
        );
        """)
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS idx_fy_eps_hist_ticker_period_date "
        f"ON {FY_HIST_TABLE} (ticker, period_end, date_recorded)"
    )
    _ensure_fy_hist_columns(conn)


def _ensure_fy_hist_columns(conn: sqlite3.Connection) -> None:
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({FY_HIST_TABLE})")
    columns = {row[1] for row in cursor.fetchall()}
    if "fiscal_year" not in columns:
        try:
            cursor.execute(f"ALTER TABLE {FY_HIST_TABLE} ADD COLUMN fiscal_year INTEGER")
        except sqlite3.OperationalError as exc:
            if "duplicate column name" not in str(exc).lower():
                raise

@contextmanager
def _connect(db_path: str = DB_PATH):
    """SQLite connection that retries once if locked."""
    try:
        conn = sqlite3.connect(db_path, timeout=30, isolation_level=None)
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

# ───────────────────────────────────────────────────────────────────────
# Utility functions
# ───────────────────────────────────────────────────────────────────────
def _last_day(date_str: str) -> str:
    """Convert '12/2025' ➜ '2025-12-31'."""
    try:
        m, y = map(int, date_str.split("/"))
        return f"{y}-{m:02d}-{calendar.monthrange(y, m)[1]:02d}"
    except Exception:
        return ""

def _to_number(series: pd.Series) -> pd.Series:
    """Vectorised numeric conversion with M/B/T suffixes."""
    def _conv(val: str) -> float:
        if not isinstance(val, str):
            return float(val or 0)
        m = NUM_RE.match(val.replace(",", ""))
        if not m:
            return None
        num, unit = m.groups()
        return float(num) * {"":1, "M":1e6, "B":1e9, "T":1e12}[unit]
    return series.map(_conv)

# ───────────────────────────────────────────────────────────────────────
# Scraping internals
# ───────────────────────────────────────────────────────────────────────
def _fetch_html(ticker: str, session: requests.Session) -> Optional[BeautifulSoup]:
    url = f"https://www.zacks.com/stock/quote/{ticker.replace('-', '.')}/detailed-earning-estimates"
    try:
        r = session.get(url, headers=HEADERS, timeout=15)
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

    def _analysts(df) -> Optional[int]:
        mask = df.iloc[:, 0].astype(str).str.contains("# of Estimates", na=False)
        if not mask.any():
            return None
        val = str(df.loc[mask].iloc[0, 1]).replace(",", "")
        return int(val) if val.isdigit() else None

    rev_analysts, eps_analysts = _analysts(sales_df), _analysts(earnings_df)
    if rev_analysts is None or eps_analysts is None:
        return None

    cons_rev = sales_df.loc[sales_df.iloc[:,0].str.contains("Consensus", na=False)]
    cons_eps = earnings_df.loc[earnings_df.iloc[:,0].str.contains("Consensus", na=False)]
    if cons_rev.empty or cons_eps.empty:
        return None
    cons_rev, cons_eps = cons_rev.iloc[0], cons_eps.iloc[0]

    this_hdr, next_hdr = sales_df.columns[3:5]
    this_date, next_date = _last_day(this_hdr.split("(")[-1].rstrip(")")), _last_day(next_hdr.split("(")[-1].rstrip(")"))
    if not this_date or not next_date:
        return None

    data = pd.DataFrame({
        "Period"                 : ["Current Year", "Next Year"],
        "Date"                   : [this_date, next_date],
        "ForwardRevenue"         : _to_number(cons_rev.iloc[3:5]).values,
        "ForwardEPS"             : pd.to_numeric(cons_eps.iloc[3:5].str.replace(",", ""),
                                                 errors="coerce").values,
        "ForwardRevenueAnalysts" : rev_analysts,
        "ForwardEPSAnalysts"     : eps_analysts
    })
    data["Period"] = ["This FY", "Next FY"]
    return data

def scrape_annual_estimates(ticker: str,
                            session: Optional[requests.Session] = None
                           ) -> pd.DataFrame:
    s = session or requests.Session()
    soup = _fetch_html(ticker, s)
    df = _parse(soup)
    return df if df is not None else pd.DataFrame()

# ───────────────────────────────────────────────────────────────────────
# Storage
# ───────────────────────────────────────────────────────────────────────
def _store(
    df: pd.DataFrame,
    ticker: str,
    db_path: str = DB_PATH,
    table_name: str = TABLE_NAME,
    conn: sqlite3.Connection | None = None,
    cursor: sqlite3.Cursor | None = None,
    commit: bool = True,
) -> None:
    own_conn = False
    if cursor is None:
        if conn is None:
            conn = sqlite3.connect(db_path, timeout=30)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=30000")
            own_conn = True
        cursor = conn.cursor()

    if own_conn:
        _ensure_table(db_path=db_path, table_name=table_name, conn=cursor.connection)

    cursor.execute(f"DELETE FROM {table_name} WHERE Ticker = ?", (ticker,))
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    today = datetime.now().strftime("%Y-%m-%d")

    for _, r in df.iterrows():
        period_end = r["Date"]
        fiscal_year = None
        try:
            fiscal_year = int(str(period_end)[:4])
        except Exception:
            fiscal_year = None
        cursor.execute(f"""
        INSERT INTO {table_name}
          (Ticker, Date, ForwardEPS, ForwardRevenue, LastUpdated,
           ForwardEPSAnalysts, ForwardRevenueAnalysts)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(Ticker, Date) DO UPDATE SET
          ForwardEPS             = excluded.ForwardEPS,
          ForwardRevenue         = excluded.ForwardRevenue,
          ForwardEPSAnalysts     = excluded.ForwardEPSAnalysts,
          ForwardRevenueAnalysts = excluded.ForwardRevenueAnalysts,
          LastUpdated            = excluded.LastUpdated;
        """, (
            ticker, period_end, r["ForwardEPS"], r["ForwardRevenue"],
            now, r.get("ForwardEPSAnalysts", None), r.get("ForwardRevenueAnalysts", None)
        ))
        cursor.execute(f"""
        INSERT INTO {FY_HIST_TABLE}
          (date_recorded, ticker, period_end, period_label, forward_eps, eps_analysts, source, fiscal_year)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(date_recorded, ticker, period_end) DO UPDATE SET
          period_label = excluded.period_label,
          forward_eps  = excluded.forward_eps,
          eps_analysts = excluded.eps_analysts,
          source       = excluded.source,
          fiscal_year  = excluded.fiscal_year;
        """, (
            today,
            ticker,
            period_end,
            r.get("Period", None),
            r["ForwardEPS"],
            r.get("ForwardEPSAnalysts", None),
            "zacks.detailed-earning-estimates",
            fiscal_year,
        ))

    cursor.execute(
        f"DELETE FROM {FY_HIST_TABLE} WHERE ticker = ? AND date_recorded < date('now', '-6 years')",
        (ticker,),
    )

    if commit or own_conn:
        cursor.connection.commit()
    if own_conn:
        conn.close()
    logging.info(f"{ticker}: stored {len(df)} rows")

# ───────────────────────────────────────────────────────────────────────
# Public API
# ───────────────────────────────────────────────────────────────────────
def scrape_forward_data(
    ticker: str,
    conn: sqlite3.Connection | None = None,
    cursor: sqlite3.Cursor | None = None,
    commit: bool = True,
) -> None:
    if conn is None and cursor is None:
        _ensure_table()
    else:
        _ensure_table(conn=(conn or cursor.connection))
    df = scrape_annual_estimates(ticker, SESSION)
    if df.empty:
        logging.info(f"{ticker}: no data scraped")
        return
    _store(df, ticker, conn=conn, cursor=cursor, commit=commit)


def ensure_forward_schema(conn: sqlite3.Connection | None = None) -> None:
    _ensure_table(conn=conn)

def scrape_forward_data_batch(tickers: List[str], max_workers: int = 6) -> None:
    """Handy helper for multi-ticker runs (I/O-bound → threads are fine)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    def _worker(tkr):
        with requests.Session() as sess:
            d = scrape_annual_estimates(tkr, sess)
            if not d.empty:
                _store(d, tkr)
            else:
                logging.info(f"{tkr}: no data")
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut_to_tkr = {ex.submit(_worker, t): t for t in tickers}
        for fut in as_completed(fut_to_tkr):
            tkr = fut_to_tkr[fut]
            try:
                fut.result()
                logging.info(f"{tkr}: done")
            except Exception:
                logging.error(f"{tkr}: FAILED\n{traceback.format_exc()}")

# ───────────────────────────────────────────────────────────────────────
# Backwards-compatible API (used by main.py)
# ───────────────────────────────────────────────────────────────────────
def scrape_and_prepare_data(
    ticker: str,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    """Legacy wrapper for main.py compatibility."""
    return scrape_annual_estimates(ticker, session or SESSION)


def store_in_database(
    df: pd.DataFrame,
    ticker: str,
    db_path: str = DB_PATH,
    table_name: str = TABLE_NAME,
) -> None:
    """Legacy wrapper for main.py compatibility."""
    _ensure_table(db_path, table_name)
    _store(df, ticker, db_path, table_name)

# ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    scrape_forward_data("AAPL")

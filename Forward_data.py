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
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import pandas as pd
import requests
import yfinance as yf
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
    _ensure_source_column(conn)


def _ensure_source_column(conn: sqlite3.Connection) -> None:
    """Add Source column to ForwardFinancialData if missing.

    Pre-existing rows (from Zacks scraping) will have Source=NULL; the migration
    script tags them 'zacks' so we can distinguish historical sources from new
    Yahoo-sourced rows going forward.
    """
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({TABLE_NAME})")
    cols = {row[1] for row in cur.fetchall()}
    if "Source" not in cols:
        try:
            cur.execute(f"ALTER TABLE {TABLE_NAME} ADD COLUMN Source TEXT")
            cur.execute(f"UPDATE {TABLE_NAME} SET Source = 'zacks' WHERE Source IS NULL")
        except sqlite3.OperationalError as exc:
            if "duplicate column name" not in str(exc).lower():
                raise


def _ensure_fy_hist_columns(conn: sqlite3.Connection) -> None:
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({FY_HIST_TABLE})")
    columns = {row[1] for row in cursor.fetchall()}
    # forward_revenue + revenue_analysts were added 2026-05-20 so the same
    # snapshot row can drive both the EPS-history and revenue-history charts.
    # Historical rows will be NULL for these — that's fine, the charts just
    # treat them as missing data.
    new_columns = [
        ("fiscal_year", "INTEGER"),
        ("forward_revenue", "REAL"),
        ("revenue_analysts", "INTEGER"),
    ]
    for name, decl in new_columns:
        if name in columns:
            continue
        try:
            cursor.execute(f"ALTER TABLE {FY_HIST_TABLE} ADD COLUMN {name} {decl}")
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
    """Zacks scraper (legacy path; kept as fallback)."""
    s = session or requests.Session()
    soup = _fetch_html(ticker, s)
    df = _parse(soup)
    return df if df is not None else pd.DataFrame()


# ───────────────────────────────────────────────────────────────────────
# Yahoo Finance path (preferred — avoids anti-scraping on GitHub Actions)
# ───────────────────────────────────────────────────────────────────────
def _ts_to_date(ts) -> Optional[str]:
    """Yahoo gives FY-end as a Unix timestamp at UTC midnight; use UTC so
    we don't shift into the prior day for users west of UTC."""
    if not ts:
        return None
    try:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d")
    except (TypeError, ValueError, OSError):
        return None


def _safe_num(row, key) -> Optional[float]:
    try:
        v = row[key]
        if pd.isna(v):
            return None
        return float(v)
    except (KeyError, TypeError):
        return None


def _safe_int(row, key) -> Optional[int]:
    try:
        v = row[key]
        if pd.isna(v):
            return None
        return int(v)
    except (KeyError, TypeError, ValueError):
        return None


def yahoo_annual_estimates(ticker: str,
                           info: Optional[dict] = None
                          ) -> pd.DataFrame:
    """Yahoo Finance replacement for Zacks scrape_annual_estimates.

    Pulls forward-EPS, forward-revenue, and analyst counts for the current
    fiscal year ('0y') and next fiscal year ('+1y') from yfinance's structured
    earnings_estimate / revenue_estimate endpoints. Returns the same DataFrame
    shape as scrape_annual_estimates so the storage layer is unchanged.

    Returns an empty DataFrame if Yahoo has no data for the ticker (caller can
    then try Zacks as a fallback).
    """
    try:
        tk = yf.Ticker(ticker)
        eps_df = tk.earnings_estimate
        rev_df = tk.revenue_estimate
    except Exception as exc:
        logging.warning(f"{ticker}: Yahoo earnings_estimate failed -- {exc}")
        return pd.DataFrame()

    if eps_df is None or eps_df.empty or "0y" not in eps_df.index:
        return pd.DataFrame()

    # Fiscal year-end dates. nextFiscalYearEnd is the FY currently being
    # estimated (i.e. "This FY"); the year after is +1y. If .info is too
    # expensive to fetch per-ticker, callers can pre-fetch and pass it in.
    if info is None:
        try:
            info = tk.info or {}
        except Exception:
            info = {}
    next_fy_ts = info.get("nextFiscalYearEnd")
    this_fy_date = _ts_to_date(next_fy_ts)
    if not this_fy_date:
        # Fallback: assume Dec 31 of the current calendar year
        this_fy_date = f"{datetime.now().year}-12-31"

    # +1y is one year after this_fy_date (preserves the fiscal-year month)
    try:
        d = datetime.strptime(this_fy_date, "%Y-%m-%d")
        next_fy_d = d.replace(year=d.year + 1)
    except ValueError:
        next_fy_d = datetime.strptime(this_fy_date, "%Y-%m-%d") + timedelta(days=365)
    next_fy_date = next_fy_d.strftime("%Y-%m-%d")

    cur_eps = eps_df.loc["0y"]
    nxt_eps = eps_df.loc["+1y"] if "+1y" in eps_df.index else None
    cur_rev = rev_df.loc["0y"] if (rev_df is not None and not rev_df.empty
                                    and "0y" in rev_df.index) else None
    nxt_rev = rev_df.loc["+1y"] if (rev_df is not None and not rev_df.empty
                                    and "+1y" in rev_df.index) else None

    rows = [{
        "Period": "This FY",
        "Date": this_fy_date,
        "ForwardEPS": _safe_num(cur_eps, "avg"),
        "ForwardRevenue": _safe_num(cur_rev, "avg") if cur_rev is not None else None,
        "ForwardEPSAnalysts": _safe_int(cur_eps, "numberOfAnalysts"),
        "ForwardRevenueAnalysts": _safe_int(cur_rev, "numberOfAnalysts")
            if cur_rev is not None else None,
    }]
    if nxt_eps is not None:
        rows.append({
            "Period": "Next FY",
            "Date": next_fy_date,
            "ForwardEPS": _safe_num(nxt_eps, "avg"),
            "ForwardRevenue": _safe_num(nxt_rev, "avg") if nxt_rev is not None else None,
            "ForwardEPSAnalysts": _safe_int(nxt_eps, "numberOfAnalysts"),
            "ForwardRevenueAnalysts": _safe_int(nxt_rev, "numberOfAnalysts")
                if nxt_rev is not None else None,
        })

    data = pd.DataFrame(rows)
    # If EPS is missing for every row, treat as no data -> let caller fall back to Zacks
    if data["ForwardEPS"].isna().all():
        return pd.DataFrame()
    return data


def fetch_annual_estimates(ticker: str,
                           prefer: str = "yahoo",
                           session: Optional[requests.Session] = None,
                           info: Optional[dict] = None,
                          ) -> tuple[pd.DataFrame, str]:
    """Return (DataFrame, source_tag). Tries Yahoo first by default, falls back
    to Zacks if Yahoo returns empty. source_tag is one of {'yahoo', 'zacks', ''}.
    """
    if prefer == "yahoo":
        df = yahoo_annual_estimates(ticker, info=info)
        if not df.empty:
            return df, "yahoo"
        logging.info(f"{ticker}: Yahoo returned empty; falling back to Zacks")
        df = scrape_annual_estimates(ticker, session)
        return (df, "zacks") if not df.empty else (df, "")
    else:
        df = scrape_annual_estimates(ticker, session)
        if not df.empty:
            return df, "zacks"
        df = yahoo_annual_estimates(ticker, info=info)
        return (df, "yahoo") if not df.empty else (df, "")

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
    source: str = "zacks",
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
           ForwardEPSAnalysts, ForwardRevenueAnalysts, Source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(Ticker, Date) DO UPDATE SET
          ForwardEPS             = excluded.ForwardEPS,
          ForwardRevenue         = excluded.ForwardRevenue,
          ForwardEPSAnalysts     = excluded.ForwardEPSAnalysts,
          ForwardRevenueAnalysts = excluded.ForwardRevenueAnalysts,
          LastUpdated            = excluded.LastUpdated,
          Source                 = excluded.Source;
        """, (
            ticker, period_end, r["ForwardEPS"], r["ForwardRevenue"],
            now, r.get("ForwardEPSAnalysts", None), r.get("ForwardRevenueAnalysts", None),
            source,
        ))
        # FY_HIST_TABLE.source describes the upstream endpoint, not just the vendor
        fy_hist_source = ("yahoo.earnings_estimate" if source == "yahoo"
                          else "zacks.detailed-earning-estimates" if source == "zacks"
                          else source)
        cursor.execute(f"""
        INSERT INTO {FY_HIST_TABLE}
          (date_recorded, ticker, period_end, period_label, forward_eps, eps_analysts,
           source, fiscal_year, forward_revenue, revenue_analysts)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(date_recorded, ticker, period_end) DO UPDATE SET
          period_label     = excluded.period_label,
          forward_eps      = excluded.forward_eps,
          eps_analysts     = excluded.eps_analysts,
          source           = excluded.source,
          fiscal_year      = excluded.fiscal_year,
          forward_revenue  = excluded.forward_revenue,
          revenue_analysts = excluded.revenue_analysts;
        """, (
            today,
            ticker,
            period_end,
            r.get("Period", None),
            r["ForwardEPS"],
            r.get("ForwardEPSAnalysts", None),
            fy_hist_source,
            fiscal_year,
            r.get("ForwardRevenue", None),
            r.get("ForwardRevenueAnalysts", None),
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
    prefer: str = "yahoo",
) -> None:
    if conn is None and cursor is None:
        _ensure_table()
    else:
        _ensure_table(conn=(conn or cursor.connection))
    df, src = fetch_annual_estimates(ticker, prefer=prefer, session=SESSION)
    if df.empty:
        logging.info(f"{ticker}: no data (tried {prefer} + fallback)")
        return
    _store(df, ticker, conn=conn, cursor=cursor, commit=commit, source=src)


def ensure_forward_schema(conn: sqlite3.Connection | None = None) -> None:
    _ensure_table(conn=conn)


def scrape_forward_data_batch(tickers: List[str], max_workers: int = 10,
                              prefer: str = "yahoo") -> None:
    """Multi-ticker fetch in parallel.

    Defaults to Yahoo (avoids Zacks anti-scraping risk on GitHub Actions); falls
    back to Zacks per-ticker if Yahoo returns empty. Set prefer='zacks' to use
    the legacy path.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _worker(tkr: str):
        sess: Optional[requests.Session] = None
        try:
            # Reuse the per-run prefetched .info (filled in main_remote before
            # this batch) so yahoo_annual_estimates doesn't re-fetch .info per
            # ticker just for nextFiscalYearEnd. Cache miss -> single live fetch.
            try:
                from forecasted_earnings_chart import get_cached_yf_info
                _info = get_cached_yf_info(tkr)
            except Exception:
                _info = None
            if prefer == "zacks":
                sess = requests.Session()
            d, src = fetch_annual_estimates(tkr, prefer=prefer, session=sess, info=_info)
            if not d.empty:
                _store(d, tkr, source=src)
                return src
            else:
                logging.info(f"{tkr}: no data")
                return None
        finally:
            if sess is not None:
                sess.close()

    counts = {"yahoo": 0, "zacks": 0, "none": 0}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut_to_tkr = {ex.submit(_worker, t): t for t in tickers}
        for fut in as_completed(fut_to_tkr):
            tkr = fut_to_tkr[fut]
            try:
                src = fut.result()
                counts[src or "none"] += 1
                logging.info(f"{tkr}: done ({src or 'no data'})")
            except Exception:
                counts["none"] += 1
                logging.error(f"{tkr}: FAILED\n{traceback.format_exc()}")
    logging.info(f"forward batch summary: {counts}")

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

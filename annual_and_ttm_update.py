# ─────────────────────────────────────────────────────────────
#  annual_and_ttm_update.py   ←  FULL REPLACEMENT
# ─────────────────────────────────────────────────────────────
import logging, os, sqlite3, numpy as np, pandas as pd, yfinance as yf
from datetime import datetime, timedelta
from functools import lru_cache

DB_PATH = "Stock Data.db"
CHART_DIR = "charts"
os.makedirs(CHART_DIR, exist_ok=True)

# ───────────────────────── DB helpers ─────────────────────────
def get_db_connection(db_path: str) -> sqlite3.Connection:
    """
    Ensure the tables have UNIQUE indexes that match the
    ON CONFLICT clauses used later in the code.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Annual_Data  (Symbol, Date) must be unique
    cur.execute(
        """CREATE TABLE IF NOT EXISTS Annual_Data(
               Symbol TEXT,
               Date   TEXT,
               Revenue REAL,
               Net_Income REAL,
               EPS REAL,
               Last_Updated TEXT,
               PRIMARY KEY(Symbol, Date)
           );"""
    )

    # TTM_Data  (Symbol, Quarter) must be unique
    cur.execute(
        """CREATE TABLE IF NOT EXISTS TTM_Data(
               Symbol TEXT,
               TTM_Revenue REAL,
               TTM_Net_Income REAL,
               TTM_EPS REAL,
               Shares_Outstanding REAL,
               Quarter TEXT,
               Last_Updated TEXT,
               PRIMARY KEY(Symbol, Quarter)
           );"""
    )

    # Extra indexes for read speed
    cur.execute("CREATE INDEX IF NOT EXISTS idx_annual_symbol ON Annual_Data(Symbol)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_ttm_symbol   ON TTM_Data(Symbol)")
    conn.commit()
    return conn

# ───────────────────────── utilities ─────────────────────────
def _to_float(v):
    try:
        return float(v)
    except Exception:
        return np.nan

def clean_financial_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ("Revenue", "Net_Income", "EPS"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    df.dropna(how="all", subset=["Revenue", "Net_Income", "EPS"], inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df

# ───────────────────── fetch from Yahoo ──────────────────────
@lru_cache(maxsize=32)
def fetch_annual_data_from_yahoo(tkr: str) -> pd.DataFrame:
    logging.info("Fetching annual data from Yahoo Finance")
    fin = yf.Ticker(tkr).financials
    if fin is None or fin.empty:
        return pd.DataFrame()
    fin = fin.T
    fin["Date"] = fin.index
    fin.rename(
        columns={
            "Total Revenue": "Revenue",
            "Net Income": "Net_Income",
            "Basic EPS": "EPS",
        },
        inplace=True,
    )
    return clean_financial_df(fin)

@lru_cache(maxsize=32)
def fetch_ttm_from_yahoo(tkr: str) -> dict | None:
    logging.info("Fetching TTM data from Yahoo Finance")
    q = yf.Ticker(tkr).quarterly_financials
    if q is None or q.empty:
        return None
    data = {
        "TTM_Revenue": q.loc["Total Revenue"].iloc[:4].sum(),
        "TTM_Net_Income": q.loc["Net Income"].iloc[:4].sum(),
        "TTM_EPS": yf.Ticker(tkr).info.get("trailingEps", np.nan),
        "Shares_Outstanding": yf.Ticker(tkr).info.get("sharesOutstanding", np.nan),
        "Quarter": q.columns[0].strftime("%Y-%m-%d"),
    }
    return data

# ───────────────────── storage helpers ───────────────────────
def store_annual(tkr, df, cur):
    for _, row in df.iterrows():
        d = row["Date"].strftime("%Y-%m-%d") if hasattr(row["Date"], "strftime") else row["Date"]
        cur.execute(
            """INSERT INTO Annual_Data
               (Symbol,Date,Revenue,Net_Income,EPS,Last_Updated)
               VALUES (?,?,?,?,?,CURRENT_TIMESTAMP)
               ON CONFLICT(Symbol,Date) DO UPDATE SET
                   Revenue      = COALESCE(EXCLUDED.Revenue,      Revenue),
                   Net_Income   = COALESCE(EXCLUDED.Net_Income,   Net_Income),
                   EPS          = COALESCE(EXCLUDED.EPS,          EPS),
                   Last_Updated = CURRENT_TIMESTAMP"""
            ,
            (tkr, d, _to_float(row["Revenue"]), _to_float(row["Net_Income"]), _to_float(row["EPS"]))
        )

def store_ttm(tkr, dct, cur):
    cur.execute(
        """INSERT INTO TTM_Data
           (Symbol,TTM_Revenue,TTM_Net_Income,TTM_EPS,Shares_Outstanding,Quarter,Last_Updated)
           VALUES (?,?,?,?,?,?,CURRENT_TIMESTAMP)
           ON CONFLICT(Symbol,Quarter) DO UPDATE SET
               TTM_Revenue        = COALESCE(EXCLUDED.TTM_Revenue,        TTM_Revenue),
               TTM_Net_Income     = COALESCE(EXCLUDED.TTM_Net_Income,     TTM_Net_Income),
               TTM_EPS            = COALESCE(EXCLUDED.TTM_EPS,            TTM_EPS),
               Shares_Outstanding = COALESCE(EXCLUDED.Shares_Outstanding, Shares_Outstanding),
               Last_Updated       = CURRENT_TIMESTAMP""",
        (
            tkr,
            _to_float(dct["TTM_Revenue"]),
            _to_float(dct["TTM_Net_Income"]),
            _to_float(dct["TTM_EPS"]),
            _to_float(dct["Shares_Outstanding"]),
            dct["Quarter"],
        ),
    )

# ───────────────────────- main updater -──────────────────────
def annual_and_ttm_update(tkr: str, db_path=DB_PATH):
    conn = get_db_connection(db_path)
    cur = conn.cursor()

    # Annual
    cur.execute("SELECT * FROM Annual_Data WHERE Symbol=?", (tkr,))
    have_annual = cur.fetchall()
    if not have_annual:
        df = fetch_annual_data_from_yahoo(tkr)
        if not df.empty:
            store_annual(tkr, df, cur)

    # TTM
    cur.execute("SELECT * FROM TTM_Data WHERE Symbol=?", (tkr,))
    have_ttm = cur.fetchall()
    if not have_ttm:
        d = fetch_ttm_from_yahoo(tkr)
        if d:
            store_ttm(tkr, d, cur)

    conn.commit()
    conn.close()
    logging.info("[%s] update complete (§)", tkr)

# stand-alone test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    annual_and_ttm_update("AAPL")

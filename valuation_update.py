# valuation_update.py
import os, csv, sqlite3, requests, pandas as pd, matplotlib.pyplot as plt
import yfinance as yf
from bs4 import BeautifulSoup

DB_PATH = "Stock Data.db"       # single source of truth

# ────────────────────────────────────────────────────────────
# DB HELPERS ── market-cap lives in Tickers_Info.marketcap
# ────────────────────────────────────────────────────────────
def ensure_marketcap_column(conn: sqlite3.Connection):
    """Add `marketcap` column to Tickers_Info if it’s not there yet."""
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(Tickers_Info);")
    cols = [row[1].lower() for row in cur.fetchall()]
    if "marketcap" not in cols:
        cur.execute("ALTER TABLE Tickers_Info ADD COLUMN marketcap REAL;")
        conn.commit()

def _get_marketcap_from_db(ticker: str, conn: sqlite3.Connection) -> float | None:
    """
    Returns the most-recent market-cap saved in Tickers_Info.marketcap
    (NULL → None).
    """
    ensure_marketcap_column(conn)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT marketcap
        FROM   Tickers_Info
        WHERE  ticker = ?
        ORDER  BY COALESCE(Last_Updated, '1900-01-01') DESC
        LIMIT 1;
        """,
        (ticker,),
    )
    row = cur.fetchone()
    return float(row[0]) if row and row[0] not in (None, "") else None


# ────────────────────────────────────────────────────────────
# FINVIZ 5-YEAR GROWTH SCRAPER
# ────────────────────────────────────────────────────────────
def finviz_five_yr(ticker, cursor):
    """Fetch 5-year EPS growth from Finviz and store in DB."""
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    hdrs = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
            " AppleWebKit/537.36 (KHTML, like Gecko)"
            " Chrome/122.0.0.0 Safari/537.36"
        )
    }
    resp = requests.get(url, headers=hdrs, timeout=15)
    if resp.status_code != 200:
        print(f"[{ticker}] Finviz error {resp.status_code}")
        return

    soup = BeautifulSoup(resp.content, "html.parser")
    cell = soup.find("td", string="EPS next 5Y")
    if not cell:
        print(f"[{ticker}] EPS next 5Y not found on Finviz")
        return

    try:
        pct = float(cell.find_next_sibling("td").text.strip("%"))
    except ValueError:
        print(f"[{ticker}] could not parse 5-yr growth")
        return

    cursor.execute("INSERT OR IGNORE INTO Tickers_Info(ticker) VALUES(?);", (ticker,))
    cursor.execute(
        "UPDATE Tickers_Info SET FINVIZ_5yr_gwth = ? WHERE ticker = ?;",
        (pct, ticker),
    )
    cursor.connection.commit()


# ────────────────────────────────────────────────────────────
# DATA FETCH
# ────────────────────────────────────────────────────────────
def fetch_financial_valuation_data(ticker: str, conn: sqlite3.Connection):
    """Pull TTM + forward numbers plus growth estimates from DB."""
    # ― TTM
    ttm = pd.read_sql_query(
        """
        SELECT 'TTM' AS Year,
               TTM_Revenue  AS Revenue,
               TTM_EPS      AS EPS
        FROM   TTM_Data
        WHERE  Symbol = ?
        ORDER BY Last_Updated DESC
        LIMIT 1;
        """,
        conn,
        params=(ticker,),
    )

    # ― Forward (max 3 rows, earliest first)
    fwd = pd.read_sql_query(
        """
        SELECT strftime('%Y', Date) AS Year,
               ForwardRevenue       AS Revenue,
               ForwardEPS           AS EPS
        FROM   ForwardFinancialData
        WHERE  Ticker = ?
        ORDER  BY Date
        LIMIT 3;
        """,
        conn,
        params=(ticker,),
    )

    # ― Growth / margin assumptions
    growth = pd.read_sql_query(
        """
        SELECT nicks_growth_rate,
               FINVIZ_5yr_gwth,
               projected_profit_margin
        FROM   Tickers_Info
        WHERE  ticker = ?;
        """,
        conn,
        params=(ticker,),
    )

    combined = pd.concat([ttm, fwd]).reset_index(drop=True)
    return combined, growth, fwd


# ────────────────────────────────────────────────────────────
# VALUATION MATH
# ────────────────────────────────────────────────────────────
def calculate_valuations(
    df: pd.DataFrame,
    growth: pd.DataFrame,
    treasury_yield_pct: float,
    current_price: float,
    marketcap: float | None,
):
    """Add valuation columns; skip Rev/Share when market-cap missing."""
    t_y = treasury_yield_pct / 100.0

    def _safe(val):
        return float(val) if pd.notna(val) else 0.0

    nicks_g = _safe(growth["nicks_growth_rate"].iloc[0]) / 100
    finviz_g = _safe(growth["FINVIZ_5yr_gwth"].iloc[0]) / 100
    profit_m = _safe(growth["projected_profit_margin"].iloc[0]) / 100

    nicks_pe  = ((nicks_g  - t_y + 1) ** 10) * 10
    finviz_pe = ((finviz_g - t_y + 1) ** 10) * 10
    nicks_ps  = nicks_pe  * profit_m
    finviz_ps = finviz_pe * profit_m

    # ─ Revenue per share only if marketcap known
    if marketcap:
        df["Revenue_Per_Share"] = df["Revenue"].astype(float) / marketcap * current_price
    else:
        df["Revenue_Per_Share"] = pd.NA

    def _row_val(r):
        if r["EPS"] > 0:
            n_v = r["EPS"] * nicks_pe
            f_v = r["EPS"] * finviz_pe if finviz_pe else None
            return "EPS", r["EPS"], n_v, f_v
        if pd.notna(r["Revenue_Per_Share"]):
            n_v = r["Revenue_Per_Share"] * nicks_ps
            f_v = r["Revenue_Per_Share"] * finviz_ps if finviz_ps else None
            return "Revenue", r["Revenue_Per_Share"], n_v, f_v
        # cannot value without EPS or MarketCap
        return "N/A", None, None, None

    basis, basis_val, n_val, f_val = zip(*( _row_val(r) for _, r in df.iterrows()))
    df["Basis_Type"]      = basis
    df["Basis_Value"]     = basis_val
    df["Nicks_Valuation"] = n_val
    df["Finviz_Valuation"]= f_val

    return df, nicks_pe, finviz_pe, nicks_ps, finviz_ps


# ────────────────────────────────────────────────────────────
# MAIN ENTRY: valuation_update()
# ────────────────────────────────────────────────────────────
def valuation_update(
    ticker: str,
    cursor: sqlite3.Cursor,
    treasury_yield: float,
    marketcap_param: float | None,
    dashboard_data: list,
):
    """
    • Refresh Finviz growth.  
    • Pull data from DB, fetch *market-cap from DB*, run valuation, plot & tables.  
    • Append one line to dashboard_data.
    """
    conn = cursor.connection  # same DB handle

    # 1) make sure we have latest Finviz growth
    finviz_five_yr(ticker, cursor)

    # 2) resolve market-cap
    marketcap = marketcap_param or _get_marketcap_from_db(ticker, conn)
    if not marketcap:
        print(f"[{ticker}] Market-cap missing → EPS-only valuation.")
    else:
        print(f"[{ticker}] Market-cap from DB = {marketcap:,.0f}")

    # 3) financial series
    combined_df, growth_df, fwd_df = fetch_financial_valuation_data(ticker, conn)
    if fwd_df.empty:
        print(f"[{ticker}] no forward numbers → skip valuation.")
        return
    if growth_df.empty:
        print(f"[{ticker}] growth assumptions absent → skip valuation.")
        return

    # 4) pull current share price once
    current_price = yf.Ticker(ticker).info.get("currentPrice")

    combined_df, n_pe, f_pe, n_ps, f_ps = calculate_valuations(
        combined_df, growth_df, treasury_yield, current_price, marketcap
    )

    # 5) (plots + html tables would be called here – unchanged in your code)
    #     >>  plot_valuation_chart(...)
    #     >>  generate_valuation_tables(...)

    # 6) push a single line to dashboard_data (unchanged logic)
    #    -- simplified for brevity, keep your exact formatting here --
    dashboard_data.append(
        [
            ticker,
            f"${current_price:,.2f}",
            f"${combined_df['Nicks_Valuation'].iloc[0]:,.2f}",
            "...",  # keep existing percentage-vs-price logic
        ]
    )

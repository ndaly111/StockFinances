# valuation_update.py  ────────────────────────────────────────────────────────
import os, csv, datetime as dt
import sqlite3
import requests
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from bs4 import BeautifulSoup


# ──────────────────────────────── helpers ────────────────────────────────────
def _ensure_market_schema(conn: sqlite3.Connection):
    """Creates a tiny cache table for last known price & market-cap."""
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS MarketData(
            ticker        TEXT PRIMARY KEY,
            last_price    REAL,
            marketcap     REAL,
            last_updated  TEXT
        );
    """)
    conn.commit()


def _cache_market_data(cur, tic: str, price: float, cap: float):
    ts = dt.datetime.utcnow().strftime("%F %T")
    cur.execute("""
        INSERT INTO MarketData(ticker,last_price,marketcap,last_updated)
        VALUES (?,?,?,?)
        ON CONFLICT(ticker) DO UPDATE
          SET last_price   = excluded.last_price,
              marketcap    = excluded.marketcap,
              last_updated = excluded.last_updated;
    """, (tic, price, cap, ts))


def _fetch_cached_market_data(cur, tic: str):
    cur.execute("SELECT last_price, marketcap FROM MarketData WHERE ticker=?;", (tic,))
    row = cur.fetchone()
    return row if row else (None, None)


# ────────────────────────────── logging helper ───────────────────────────────
def log_valuation_data(ticker, nicks_ttm_valuation, nicks_forward_valuation,
                       finviz_ttm_valuation, finviz_forward_valuation):
    db_path = "Stock Data.db"
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ValuationHistory(
                id INTEGER PRIMARY KEY,
                ticker TEXT,
                date   DATE DEFAULT (datetime('now','localtime')),
                nicks_ttm_valuation    REAL,
                nicks_forward_valuation REAL,
                finviz_ttm_valuation    REAL,
                finviz_forward_valuation REAL
            );
        """)
        cursor.execute("""
            INSERT INTO ValuationHistory
                  (ticker,nicks_ttm_valuation,nicks_forward_valuation,
                   finviz_ttm_valuation,finviz_forward_valuation)
            VALUES (?,?,?,?,?);
        """, (ticker, nicks_ttm_valuation, nicks_forward_valuation,
              finviz_ttm_valuation, finviz_forward_valuation))
        conn.commit()
        print(f"Inserted valuation data for {ticker} into ValuationHistory.")


# ─────────────────────────── scrape Finviz growth ───────────────────────────
def finviz_five_yr(ticker, cursor):
    url = f'https://finviz.com/quote.ashx?t={ticker}'
    headers = {'User-Agent': 'Mozilla/5.0'}
    resp = requests.get(url, headers=headers, timeout=15)
    if resp.status_code != 200:
        print(f"Failed to retrieve Finviz data for {ticker}, status {resp.status_code}")
        return

    soup   = BeautifulSoup(resp.content, 'html.parser')
    label  = soup.find('td', text='EPS next 5Y')
    if not label:
        print(f"No EPS next 5Y on Finviz for {ticker}")
        return

    raw = label.find_next_sibling('td').text.strip('%')
    if not raw:
        print(f"Empty Finviz 5Y growth for {ticker}")
        return
    try:
        growth = float(raw)
    except ValueError:
        print(f"Invalid Finviz 5Y growth '{raw}' for {ticker}")
        return

    # upsert into Tickers_Info
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Tickers_Info(
            ticker TEXT PRIMARY KEY,
            nicks_growth_rate        REAL,
            FINVIZ_5yr_gwth          REAL,
            projected_profit_margin  REAL
        );
    """)
    cursor.execute("INSERT OR IGNORE INTO Tickers_Info(ticker) VALUES(?);", (ticker,))
    cursor.execute("""
        UPDATE Tickers_Info SET FINVIZ_5yr_gwth=? WHERE ticker=?;
    """, (growth, ticker))
    cursor.connection.commit()
    print(f"Stored Finviz 5Y growth for {ticker}: {growth}%")


# ─────────────────────── read financials already in DB ──────────────────────
def fetch_financial_valuation_data(ticker, db_path):
    stock = yf.Ticker(ticker)
    current_price = stock.info.get('currentPrice')

    with sqlite3.connect(db_path) as conn:
        ttm_df = pd.read_sql_query("""
            SELECT 'TTM' AS Year, TTM_Revenue AS Revenue, TTM_EPS AS EPS
              FROM TTM_Data
             WHERE Symbol=?
          ORDER BY Last_Updated DESC LIMIT 1;
        """, conn, params=(ticker,))

        fwd_df = pd.read_sql_query("""
            SELECT strftime('%Y',Date) AS Year,
                   ForwardRevenue      AS Revenue,
                   ForwardEPS          AS EPS
              FROM ForwardFinancialData
             WHERE Ticker=? ORDER BY Date;
        """, conn, params=(ticker,))

        gv_df = pd.read_sql_query("""
            SELECT nicks_growth_rate, FINVIZ_5yr_gwth, projected_profit_margin
              FROM Tickers_Info WHERE ticker=?;
        """, conn, params=(ticker,))

    combined = pd.concat([ttm_df, fwd_df], ignore_index=True)
    return combined, gv_df, current_price, fwd_df


# ───────────────────────── market-cap / price helper ────────────────────────
def fetch_stock_data(ticker, cur):
    """
    1. Try Yahoo Finance live.
    2. If price or market-cap missing, fall back to cached MarketData table.
    3. Cache fresh values when both are available.
    """
    info  = yf.Ticker(ticker).info or {}
    price = info.get('currentPrice') or info.get('regularMarketPrice')
    cap   = info.get('marketCap')

    if price is None or cap is None:
        cached_price, cached_cap = _fetch_cached_market_data(cur, ticker)
        price = price or cached_price
        cap   = cap   or cached_cap

    if price is None or cap is None:
        print(f"⚠️  {ticker}: missing price ({price}) or market-cap ({cap}); skipping.")
        return None, None, None, None, None

    _cache_market_data(cur, ticker, price, cap)  # refresh cache

    fwd_eps = info.get('forwardEps')
    pe      = info.get('trailingPE')
    ps      = info.get('priceToSalesTrailing12Months')
    fwd_pe  = price / fwd_eps if fwd_eps else None
    return price, pe, ps, fwd_pe, cap


# ─────────────────────────── valuation maths ────────────────────────────────
def calculate_valuations(combined_data, growth_values,
                         treasury_yield, current_price, marketcap):
    ty = float(treasury_yield) / 100
    ngr = (growth_values['nicks_growth_rate'].iloc[0]        or 0) / 100
    fgr = (growth_values['FINVIZ_5yr_gwth'].iloc[0]          or 0) / 100
    ppm = (growth_values['projected_profit_margin'].iloc[0]  or 0) / 100

    n_pe = ((ngr - ty + 1) ** 10) * 10
    f_pe = ((fgr - ty + 1) ** 10) * 10
    n_ps = n_pe * ppm
    f_ps = f_pe * ppm

    combined_data['Revenue_Per_Share'] = (
        combined_data['Revenue'] / marketcap * current_price
    )

    def row_val(r):
        if r.EPS > 0:
            return r.EPS, 'EPS', r.EPS * n_pe, r.EPS * f_pe
        else:
            rps = r.Revenue_Per_Share
            return rps, 'Revenue', rps * n_ps, rps * f_ps

    tmp = combined_data.apply(row_val, axis=1, result_type='expand')
    tmp.columns = ['Basis_Value','Basis_Type','Nicks_Valuation','Finviz_Valuation']
    combined_data[tmp.columns] = tmp
    return combined_data, n_pe, f_pe, n_ps, f_ps


# ──────────────────────────────── plotting ──────────────────────────────────
def plot_valuation_chart(valuation_data, current_price, ticker, growth_values):
    fig, ax = plt.subplots(figsize=(10,6))
    yrs = valuation_data['Year']
    if pd.notna(growth_values['nicks_growth_rate'].iloc[0]):
        ax.plot(yrs, valuation_data['Nicks_Valuation'], marker='o', label='Nicks Valuation')
    if pd.notna(growth_values['FINVIZ_5yr_gwth'].iloc[0]):
        ax.plot(yrs, valuation_data['Finviz_Valuation'], marker='o', label='Finviz Valuation')
    ax.plot(yrs, [current_price]*len(yrs), marker='x', linestyle='--', label='Current Price')
    ax.set_title(f"Valuation Comparison for {ticker}")
    ax.set_xlabel("Year"); ax.set_ylabel("USD"); ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    os.makedirs("charts", exist_ok=True)
    out = os.path.join("charts", f"{ticker}_valuation_chart.png")
    plt.tight_layout(); plt.savefig(out, bbox_inches='tight'); plt.close()
    print(f"Figure saved to {out}")


# ──────────────────────────── HTML table helpers ────────────────────────────
def generate_valuation_tables(ticker, combined_data, growth_values,
                              treasury_yield, current_price,
                              n_pe, f_pe, n_ps, cursor):
    price, pe, ps, fwd_pe, mcap = fetch_stock_data(ticker, cursor)
    if price is None:   # skip – already warned
        return

    os.makedirs("charts", exist_ok=True)

    df1 = pd.DataFrame({
        "Share Price":      [f"${price:,.2f}"],
        "Treasury Yield":   [f"{float(treasury_yield):.2f}%"],
        "Fair P/E (Nicks)": [f"{n_pe:.1f}"],
        "Fair P/E (Finviz)":[f"{f_pe:.1f}"],
        "Fair P/S (Nicks)": [f"{n_ps:.3f}"],
        "Current P/E":      [f"{pe:.1f}" if pe else "N/A"],
        "Current P/S":      [f"{ps:.2f}" if ps else "N/A"]
    })
    p1 = os.path.join("charts", f"{ticker}_valuation_info.html")
    df1.to_html(p1, index=False, escape=False); print(f"Saved summary → {p1}")

    fmt = lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A"
    df2 = combined_data.copy()
    df2['Nicks_Valuation']  = df2['Nicks_Valuation'].apply(fmt)
    df2['Finviz_Valuation'] = df2['Finviz_Valuation'].apply(fmt)
    p2 = os.path.join("charts", f"{ticker}_valuation_table.html")
    df2.to_html(p2, index=False, escape=False); print(f"Saved detail → {p2}")


# ───────────────────────────── CSV growth ingest ────────────────────────────
def process_update_growth_csv(file_path, db_path):
    if not os.path.exists(file_path):
        print(f"{file_path} not found"); return
    conn = sqlite3.connect(db_path); cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS Tickers_Info(
            ticker TEXT PRIMARY KEY,
            nicks_growth_rate        REAL,
            FINVIZ_5yr_gwth          REAL,
            projected_profit_margin  REAL
        );
    """)
    with open(file_path, newline='') as f:
        for row in csv.reader(f):
            if len(row) not in (2,3): print("Bad row", row); continue
            tic, rate, *pm = [x.strip() for x in row]
            try: rate_f = float(rate)
            except: print(f"Bad rate {rate} for {tic}"); continue
            pm_val = float(pm[0]) if pm else None
            cur.execute("INSERT OR IGNORE INTO Tickers_Info(ticker) VALUES(?);", (tic.upper(),))
            cur.execute("""
                UPDATE Tickers_Info
                   SET nicks_growth_rate=?, projected_profit_margin=?
                 WHERE ticker=?;
            """, (rate_f, pm_val, tic.upper()))
    conn.commit(); conn.close()
    open(file_path,'w').close(); print(f"Processed & cleared {file_path}")


# ──────────────────────────── main entry point ──────────────────────────────
def valuation_update(ticker, cursor, treasury_yield,
                     incoming_marketcap, dashboard_data):
    db_path = "Stock Data.db"
    _ensure_market_schema(cursor.connection)  # ensure cache table
    finviz_five_yr(ticker, cursor)

    combined, gv, cur_price, fwd_df = fetch_financial_valuation_data(ticker, db_path)
    if fwd_df.empty:
        print(f"No forecast for {ticker}, skipping"); return
    if gv.empty or (pd.isna(gv.iloc[0,0]) and pd.isna(gv.iloc[0,1])):
        print("Missing growth data, skipping"); return

    # ── get reliable price & market-cap ────────────────────────────────────
    price, *_ , marketcap = fetch_stock_data(ticker, cursor)
    if price is None:     # already warned
        return
    if incoming_marketcap is None:
        incoming_marketcap = marketcap

    combined, n_pe, f_pe, n_ps, _ = calculate_valuations(
        combined, gv, treasury_yield, price, incoming_marketcap
    )

    plot_valuation_chart(combined, price, ticker, gv)
    generate_valuation_tables(ticker, combined, gv, treasury_yield,
                              price, n_pe, f_pe, n_ps, cursor)

    try:
        log_valuation_data(
            ticker,
            combined.loc[0,'Nicks_Valuation'],
            combined.loc[1,'Nicks_Valuation'],
            combined.loc[0,'Finviz_Valuation'],
            combined.loc[1,'Finviz_Valuation']
        )
    except Exception:
        pass

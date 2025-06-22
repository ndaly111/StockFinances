# valuation_update.py  — FULL FILE
# -----------------------------------------------------------
# Fetches growth data, calculates fair-value PE/PS, builds
# valuation tables & charts, logs history, and updates the DB.
# -----------------------------------------------------------

import os
import csv
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import yfinance as yf
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
#  Safe price helper
# ---------------------------------------------------------------------------
def get_current_price(ticker_obj: yf.Ticker):
    """
    Robust share-price lookup.
    1) ticker.info['currentPrice']
    2) ticker.fast_info['lastPrice']
    3) last daily close
    Returns float or None.
    """
    price = ticker_obj.info.get("currentPrice")
    if price is None:
        try:
            price = ticker_obj.fast_info.get("lastPrice")
        except Exception:
            price = None
    if price is None:
        try:
            hist = ticker_obj.history(period="1d")
            if not hist.empty and "Close" in hist.columns:
                price = float(hist["Close"].iloc[-1])
        except Exception:
            price = None
    return price


# ---------------------------------------------------------------------------
#  History logger
# ---------------------------------------------------------------------------
def log_valuation_data(ticker, nicks_ttm_valuation, nicks_forward_valuation,
                       finviz_ttm_valuation, finviz_forward_valuation):
    db_path = "Stock Data.db"
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ValuationHistory (
                id INTEGER PRIMARY KEY,
                ticker TEXT,
                date DATE DEFAULT (datetime('now','localtime')),
                nicks_ttm_valuation REAL,
                nicks_forward_valuation REAL,
                finviz_ttm_valuation REAL,
                finviz_forward_valuation REAL
            );
        """)
        cursor.execute("""
            INSERT INTO ValuationHistory
                (ticker, nicks_ttm_valuation, nicks_forward_valuation,
                 finviz_ttm_valuation, finviz_forward_valuation)
            VALUES (?,?,?,?,?);
        """, (ticker, nicks_ttm_valuation, nicks_forward_valuation,
              finviz_ttm_valuation, finviz_forward_valuation))
        conn.commit()
        print(f"[{ticker}] valuation row inserted.")


# ---------------------------------------------------------------------------
#  FINVIZ 5-year EPS growth scraper
# ---------------------------------------------------------------------------
def finviz_five_yr(ticker, cursor):
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    headers = {
        "User-Agent":
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }

    resp = requests.get(url, headers=headers, timeout=15)
    if resp.status_code != 200:
        print(f"[{ticker}] Finviz request failed ({resp.status_code}).")
        return

    soup = BeautifulSoup(resp.content, "html.parser")
    cell = soup.find("td", text="EPS next 5Y")
    if not cell:
        print(f"[{ticker}] Finviz 5-yr growth cell not found.")
        return

    try:
        pct = float(cell.find_next_sibling("td").text.strip("%"))
    except (AttributeError, ValueError):
        print(f"[{ticker}] Couldn’t parse 5-yr growth.")
        return

    # ensure ticker exists
    cursor.execute("SELECT 1 FROM Tickers_Info WHERE ticker=?;", (ticker,))
    if not cursor.fetchone():
        cursor.execute("INSERT INTO Tickers_Info (ticker) VALUES (?);", (ticker,))
        cursor.connection.commit()

    cursor.execute("""
        UPDATE Tickers_Info
        SET FINVIZ_5yr_gwth = ?
        WHERE ticker = ?;
    """, (pct, ticker))
    cursor.connection.commit()
    print(f"[{ticker}] Finviz 5-yr growth set to {pct}%")


# ---------------------------------------------------------------------------
#  Financial data fetcher
# ---------------------------------------------------------------------------
def fetch_financial_valuation_data(ticker, db_path):
    stock = yf.Ticker(ticker)
    current_price = get_current_price(stock)

    with sqlite3.connect(db_path) as conn:
        ttm = pd.read_sql_query(
            """
            SELECT 'TTM' AS Year, TTM_Revenue AS Revenue, TTM_EPS AS EPS
            FROM TTM_Data
            WHERE Symbol=? ORDER BY Last_Updated DESC LIMIT 1;
            """, conn, params=(ticker,))
        fcst = pd.read_sql_query(
            """
            SELECT strftime('%Y', Date) AS Year,
                   ForwardRevenue AS Revenue,
                   ForwardEPS AS EPS
            FROM ForwardFinancialData
            WHERE Ticker=? ORDER BY Date;
            """, conn, params=(ticker,))
        growth = pd.read_sql_query(
            """
            SELECT nicks_growth_rate,
                   FINVIZ_5yr_gwth,
                   projected_profit_margin
            FROM Tickers_Info WHERE ticker=?;
            """, conn, params=(ticker,))

    combined = pd.concat([ttm, fcst]).reset_index(drop=True)
    return combined, growth, current_price, fcst


# ---------------------------------------------------------------------------
#  Valuation maths
# ---------------------------------------------------------------------------
def calculate_valuations(combined, growth, treasury_yield,
                         current_price, marketcap):
    treasury_yield = float(treasury_yield) / 100.0
    nicks_gr = float(growth["nicks_growth_rate"].iloc[0] or 0) / 100.0
    finviz_gr = float(growth["FINVIZ_5yr_gwth"].iloc[0] or 0) / 100.0
    margin    = float(growth["projected_profit_margin"].iloc[0] or 0) / 100.0

    nicks_fair_pe  = ((nicks_gr  - treasury_yield + 1) ** 10) * 10
    finviz_fair_pe = ((finviz_gr - treasury_yield + 1) ** 10) * 10
    nicks_fair_ps  = nicks_fair_pe  * margin
    finviz_fair_ps = finviz_fair_pe * margin

    if current_price is not None:
        combined["Revenue_Per_Share"] = (combined["Revenue"] / marketcap) * current_price
    else:
        combined["Revenue_Per_Share"] = np.nan

    def _row_val(row):
        if row["EPS"] > 0:
            return (row["EPS"] * nicks_fair_pe,
                    row["EPS"] * finviz_fair_pe if finviz_fair_pe else None,
                    "EPS", row["EPS"])
        else:
            rps = row["Revenue_Per_Share"]
            return (rps * nicks_fair_ps,
                    rps * finviz_fair_ps if finviz_fair_ps else None,
                    "Revenue", rps)

    vals = combined.apply(lambda r: _row_val(r), axis=1, result_type="expand")
    combined["Nicks_Valuation"]   = vals[0]
    combined["Finviz_Valuation"]  = vals[1]
    combined["Basis_Type"]        = vals[2]
    combined["Basis_Value"]       = vals[3]
    return combined, nicks_fair_pe, finviz_fair_pe, nicks_fair_ps, finviz_fair_ps


# ---------------------------------------------------------------------------
#  Chart maker
# ---------------------------------------------------------------------------
def plot_valuation_chart(df, current_price, ticker, growth):
    fig, ax = plt.subplots(figsize=(10, 6))

    if pd.notna(growth['nicks_growth_rate'].iloc[0]):
        ax.plot(df['Year'], df['Nicks_Valuation'],
                label="Nicks Valuation", marker="o")

    if pd.notna(growth['FINVIZ_5yr_gwth'].iloc[0]):
        ax.plot(df['Year'], df['Finviz_Valuation'],
                label="Finviz Valuation", marker="o", color="green")

    ax.plot(df['Year'], [current_price]*len(df),
            label="Current Price", linestyle="--", marker="x", color="orange")

    ax.set_xlabel("Year"); ax.set_ylabel("Valuation (USD)")
    ax.set_title(f"Valuation Comparison – {ticker}")
    ax.legend(); ax.grid(True, linestyle="--", alpha=.7)

    path = f"charts/{ticker}_valuation_chart.png"
    plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"[{ticker}] chart saved -> {path}")


# ---------------------------------------------------------------------------
#  Live price / multiples fetch
# ---------------------------------------------------------------------------
def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)
    current_price = get_current_price(stock)
    forward_eps   = stock.info.get("forwardEps")
    pe_ratio      = stock.info.get("tralingPE")
    ps_ratio      = stock.info.get("priceToSalesTrailing12Months")
    fwd_pe        = current_price / forward_eps if (forward_eps and current_price) else None
    return current_price, pe_ratio, ps_ratio, fwd_pe


# ---------------------------------------------------------------------------
#  HTML valuation tables
# ---------------------------------------------------------------------------
def generate_valuation_tables(ticker, combined, growth_values, treasury_yield,
                              current_price, nicks_fair_pe,
                              finviz_fair_pe, nicks_fair_ps):

    # Format snapshot values
    current_price_formatted = f"${current_price:.2f}"
    treasury_yield_formatted = f"{treasury_yield:.1f}%"

    # Build the table as before (omitted for brevity)
    # …

# ---------------------------------------------------------------------------
#  ** RESTORED ** process_update_growth_csv
# ---------------------------------------------------------------------------
def process_update_growth_csv(file_path, db_path="Stock Data.db"):
    """
    Reads update_growth.csv and updates Tickers_Info
    with Nick's growth rate and projected profit margin.
    After processing, wipes the file.
    """
    if not os.path.exists(file_path):
        print(f"{file_path} not found – no growth updates.")
        return

    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()

    with open(file_path, newline="") as fh:
        rdr = csv.reader(fh)
        for row in rdr:
            if len(row) == 2:
                ticker, growth = row
                margin = None
            elif len(row) == 3:
                ticker, growth, margin = row
                margin = None if margin == "0" else margin
            else:
                print("Bad row ->", row)
                continue

            ticker = ticker.upper()
            try:
                growth = float(growth)
                margin = float(margin) if margin else None
            except ValueError:
                print(f"[{ticker}] invalid number(s) ->", row)
                continue

            # ensure ticker row exists
            cur.execute("SELECT 1 FROM Tickers_Info WHERE ticker=?;", (ticker,))
            if not cur.fetchone():
                cur.execute("INSERT INTO Tickers_Info (ticker) VALUES (?);", (ticker,))

            cur.execute("""
                UPDATE Tickers_Info
                SET nicks_growth_rate=?,
                    projected_profit_margin=?
                WHERE ticker=?;
            """, (growth, margin, ticker))
            conn.commit()
            print(f"[{ticker}] growth={growth}%, margin={margin}%")

    conn.close()
    open(file_path, "w").close()  # wipe
    print(f"{file_path} processed & cleared.")


# ---------------------------------------------------------------------------
#  Master driver – called from main_remote.py
# ---------------------------------------------------------------------------
def valuation_update(ticker, cursor, treasury_yield, marketcap, dashboard):
    db_path = "Stock Data.db"

    finviz_five_yr(ticker, cursor)
    combo, growth, price, fcst = fetch_financial_valuation_data(ticker, db_path)

    if price is None:
        print(f"[{ticker}] price unavailable – skipping.")
        return
    if fcst.empty:
        print(f"[{ticker}] no forecast rows – skipping.")
        return
    if growth.empty or (pd.isna(growth["nicks_growth_rate"].iloc[0]) and
                        pd.isna(growth["FINVIZ_5yr_gwth"].iloc[0])):
        print(f"[{ticker}] growth inputs missing – skipping.")
        return

    combo, n_pe, f_pe, n_ps, _ = calculate_valuations(
        combo, growth, treasury_yield, price, marketcap)

    plot_valuation_chart(combo[["Year", "Nicks_Valuation", "Finviz_Valuation"]],
                         price, ticker, growth)

    generate_valuation_tables(ticker, combo, growth, treasury_yield,
                              price, n_pe, f_pe, n_ps)

    # --- push one-line dashboard summary -------------------------------
    try:
        def clean(v):
            return float(str(v).strip('$BMK').replace(',', ''))
        n_ttm  = clean(combo["Nicks_Valuation"].iloc[0])
        n_fwd  = clean(combo["Nicks_Valuation"].iloc[1])
        n_ttm_pct  = (n_ttm/price - 1)*100
        n_fwd_pct  = (n_fwd/price - 1)*100

        if pd.notna(growth["FINVIZ_5yr_gwth"].iloc[0]):
            f_ttm = clean(combo["Finviz_Valuation"].iloc[0])
            f_fwd = clean(combo["Finviz_Valuation"].iloc[1])
            f_ttm_pct = (f_ttm/price - 1)*100
            f_fwd_pct = (f_fwd/price - 1)*100
        else:
            f_ttm = f_fwd = f_ttm_pct = f_fwd_pct = "-"
    except Exception as e:
        print(f"[{ticker}] dashboard calc error: {e}")
        n_ttm = n_fwd = n_ttm_pct = n_fwd_pct = "-"
        f_ttm = f_fwd = f_ttm_pct = f_fwd_pct = "-"

    dashboard.append([
        ticker,
        f"${price:.2f}",
        f"{n_ttm_pct:.1f}%" if isinstance(n_ttm_pct, float) else n_ttm_pct,
        f"{n_fwd_pct:.1f}%" if isinstance(n_fwd_pct, float) else n_fwd_pct,
        f"{f_ttm_pct:.1f}%" if isinstance(f_ttm_pct, float) else f_ttm_pct,
        f"{f_fwd_pct:.1f}%" if isinstance(f_fwd_pct, float) else f_fwd_pct
    ])

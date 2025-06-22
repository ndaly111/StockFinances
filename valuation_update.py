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
    pe_ratio      = stock.info.get("trailingPE")
    ps_ratio      = stock.info.get("priceToSalesTrailing12Months")
    fwd_pe        = current_price / forward_eps if (forward_eps and current_price) else None
    return current_price, pe_ratio, ps_ratio, fwd_pe


# ---------------------------------------------------------------------------
#  HTML valuation tables
# ---------------------------------------------------------------------------
def generate_valuation_tables(
    ticker,
    combined,
    growth_values,
    treasury_yield,
    current_price,
    nicks_fair_pe,
    finviz_fair_pe,
    nicks_fair_ps
):
    import pandas as pd

    # 1) Snapshot metrics formatting
    current_price_formatted = f"${current_price:,.2f}"
    treasury_yield_formatted = f"{float(treasury_yield):.1f}%"

    # raw input values (may be None/NaN)
    n_gr = growth_values['nicks_growth_rate'].iloc[0]
    f_gr = growth_values['FINVIZ_5yr_gwth'].iloc[0]
    m_pm = growth_values['projected_profit_margin'].iloc[0]

    # if NaN or None, show "-" instead of crashing
    nicks_growth_rate_formatted = f"{n_gr:.0f}%" if pd.notna(n_gr) else "-"
    finviz_growth_rate_formatted = f"{f_gr:.0f}%" if pd.notna(f_gr) else "-"
    expected_margin_formatted    = f"{m_pm:.0f}%" if pd.notna(m_pm) else "-"

    estimates_string = (
        f"Nicks&nbsp;Growth:&nbsp;{nicks_growth_rate_formatted}<br>"
        f"Nick's&nbsp;Expected&nbsp;Margin:&nbsp;{expected_margin_formatted}<br>"
        f"FINVIZ&nbsp;Growth:&nbsp;{finviz_growth_rate_formatted}"
    )

    # Fair‐value P/E strings
    fair_pe_string = (
        f"Nicks:&nbsp;{nicks_fair_pe:.0f}<br>"
        f"Finviz:&nbsp;{finviz_fair_pe:.0f}"
        if finviz_fair_pe is not None else
        "Finviz: N/A"
    )

    # Fetch live multiples for current P/E and P/S
    cp, pe_ratio, price_to_sales, forward_pe_ratio = fetch_stock_data(ticker)

    # Build the snapshot table
    table_1_data = {
        "Share Price":      [current_price_formatted],
        "Treasury Yield":   [treasury_yield_formatted],
        "Estimates":        [estimates_string],
        "Fair Value (P/E)": [fair_pe_string],
        "Fair Value (P/S)": [f"Nick's: {nicks_fair_ps:.3f}"],
        "Current P/S":      [f"{price_to_sales:.1f}" if price_to_sales else "N/A"]
    }
    if pe_ratio and pe_ratio > 0:
        table_1_data["Current P/E"] = [f"{pe_ratio:.1f}"]

    table_1_df = pd.DataFrame(table_1_data)
    table_1_df.to_html(
        f"charts/{ticker}_valuation_info.html",
        index=False,
        escape=False,
        classes="table table-striped",
        justify="left"
    )

    # 2) Year-by-year valuations table
    def fmt_val(v):
        return "N/A" if v is None else f"${v:,.2f}"

    combined["Nicks_Valuation_str"]  = combined["Nicks_Valuation"].apply(fmt_val)
    combined["Finviz_Valuation_str"] = combined["Finviz_Valuation"].apply(fmt_val)

    def pct_cell(x):
        try:
            return f"{x:.1f}%"
        except Exception:
            return "-"

    combined["Nicks vs Share Price"] = combined.apply(
        lambda r: pct_cell((r["Nicks_Valuation"] / current_price - 1) * 100)
                  if r["Nicks_Valuation"] not in (None, 0) else "-",
        axis=1
    )

    if pd.notna(f_gr):
        combined["Finviz vs Share Price"] = combined.apply(
            lambda r: pct_cell((r["Finviz_Valuation"] / current_price - 1) * 100)
                      if r["Finviz_Valuation"] not in (None, 0) else "-",
            axis=1
        )

    combined["Basis"] = combined.apply(
        lambda r: f"${format_number(r['EPS'])} EPS"
                  if r["EPS"] > 0 else
                  f"${format_number(r['Revenue_Per_Share'])} RevPS",
        axis=1
    )

    cols = ["Basis", "Year", "Nicks_Valuation_str", "Nicks vs Share Price"]
    if pd.notna(f_gr):
        cols += ["Finviz_Valuation_str", "Finviz vs Share Price"]

    table_2_df = combined[cols].rename(columns={
        "Nicks_Valuation_str":  "Nicks Valuation",
        "Finviz_Valuation_str": "Finviz Valuation"
    })

    table_2_df.to_html(
        f"charts/{ticker}_valuation_table.html",
        index=False,
        escape=False,
        classes="table table-striped",
        justify="left"
    )

    print(f"[{ticker}] valuation tables saved.")


# ---------------------------------------------------------------------------
#  Misc helpers
# ---------------------------------------------------------------------------
def format_number(v):
    if v >= 1_000_000_000:
        return f"{v/1_000_000_000:.2f}B"
    if v >= 1_000_000:
        return f"{v/1_000_000:.2f}M"
    if v >= 1_000:
        return f"{v/1_000:.2f}K"
    return f"{v:.2f}"


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

            cur.execute("SELECT 1 FROM Tickers_Info WHERE ticker=?;", (ticker,))
            if not cur.fetchone():
                cur.execute("INSERT INTO Tickers_Info (ticker) VALUES (?);", (ticker,))

            cur.execute("""
                UPDATE Tickers_Info
                SET n

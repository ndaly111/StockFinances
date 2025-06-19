import requests
from bs4 import BeautifulSoup
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import os
import yfinance as yf
import csv

DB_PATH = "Stock Data.db"
CHART_DIR = "charts"
os.makedirs(CHART_DIR, exist_ok=True)

def log_valuation_data(ticker, nicks_ttm_valuation, nicks_forward_valuation, finviz_ttm_valuation,
                       finviz_forward_valuation):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute('''
            CREATE TABLE IF NOT EXISTS ValuationHistory (
                id INTEGER PRIMARY KEY,
                ticker TEXT,
                date DATE DEFAULT (datetime('now','localtime')),
                nicks_ttm_valuation REAL,
                nicks_forward_valuation REAL,
                finviz_ttm_valuation REAL,
                finviz_forward_valuation REAL
            );
        ''')
        cur.execute('''
            INSERT INTO ValuationHistory
              (ticker, nicks_ttm_valuation, nicks_forward_valuation, finviz_ttm_valuation, finviz_forward_valuation)
            VALUES (?, ?, ?, ?, ?);
        ''', (ticker, nicks_ttm_valuation, nicks_forward_valuation, finviz_ttm_valuation, finviz_forward_valuation))
        conn.commit()
    print(f"Inserted valuation data for {ticker} into ValuationHistory.")


def finviz_five_yr(ticker, cursor):
    """Fetches and stores the 5-year EPS growth percentage from Finviz."""
    url = f'https://finviz.com/quote.ashx?t={ticker}'
    headers = {
        'User-Agent': 'Mozilla/5.0'
    }
    resp = requests.get(url, headers=headers)
    if resp.status_code != 200:
        print(f"Failed to retrieve Finviz for {ticker}: {resp.status_code}")
        return
    soup = BeautifulSoup(resp.content, 'html.parser')
    cell = soup.find('td', text='EPS next 5Y')
    if not cell:
        print(f"No EPS next 5Y on Finviz for {ticker}")
        return
    val = cell.find_next_sibling('td').text.strip('%')
    try:
        gw = float(val)
    except ValueError:
        print(f"Invalid Finviz growth '{val}' for {ticker}")
        return

    # ensure ticker exists
    cursor.execute("SELECT 1 FROM Tickers_Info WHERE ticker=?", (ticker,))
    if not cursor.fetchone():
        cursor.execute("INSERT INTO Tickers_Info(ticker) VALUES(?)", (ticker,))
    cursor.execute("""
        UPDATE Tickers_Info
           SET FINVIZ_5yr_gwth = ?
         WHERE ticker = ?
    """, (gw, ticker))
    cursor.connection.commit()
    print(f"Stored Finviz 5Y growth for {ticker}: {gw}%")


def fetch_financial_valuation_data(ticker, db_path):
    """
    Returns:
      combined_data: TTM + forecast, with no NaNs in Revenue/EPS
      growth   : growth rates table, with no NaNs in margins/growth
      price    : currentPrice
      forecast : the forecast slice
      mcap     : marketCap
    """
    stock = yf.Ticker(ticker)
    info  = stock.info or {}
    price = info.get('currentPrice', 0.0) or 0.0
    mcap  = info.get('marketCap',     0.0) or 0.0

    with sqlite3.connect(db_path) as conn:
        # TTM
        ttm_query = """
          SELECT 'TTM' AS Year, TTM_Revenue AS Revenue, TTM_EPS AS EPS
            FROM TTM_Data
           WHERE Symbol = ?
        ORDER BY Last_Updated DESC
           LIMIT 1;
        """
        ttm = pd.read_sql_query(ttm_query, conn, params=(ticker,))

        # Forecast
        forecast_query = """
          SELECT strftime('%Y', Date) AS Year,
                 ForwardRevenue AS Revenue,
                 ForwardEPS     AS EPS
            FROM ForwardFinancialData
           WHERE Ticker = ?
        ORDER BY Date;
        """
        forecast = pd.read_sql_query(forecast_query, conn, params=(ticker,))

        # Growth rates
        growth_query = """
          SELECT nicks_growth_rate,
                 FINVIZ_5yr_gwth,
                 projected_profit_margin
            FROM Tickers_Info
           WHERE ticker = ?;
        """
        growth = pd.read_sql_query(growth_query, conn, params=(ticker,))

    # ─── Fill NaNs so we never multiply None ───────────────────────
    ttm[['Revenue','EPS']]         = ttm[['Revenue','EPS']].fillna(0.0)
    forecast[['Revenue','EPS']]    = forecast[['Revenue','EPS']].fillna(0.0)
    growth['nicks_growth_rate']    = growth['nicks_growth_rate'].fillna(0.0)
    growth['FINVIZ_5yr_gwth']      = growth['FINVIZ_5yr_gwth'].fillna(0.0)
    growth['projected_profit_margin'] = growth['projected_profit_margin'].fillna(0.0)

    combined = pd.concat([ttm, forecast], ignore_index=True)
    return combined, growth, price, forecast, mcap


def calculate_valuations(df, growth, treasury_yield, price, mcap):
    tyld = float(treasury_yield) / 100
    # growth rates already filled, so no None here:
    n_gw = growth['nicks_growth_rate'].iloc[0] / 100
    f_gw = growth['FINVIZ_5yr_gwth'].iloc[0]   / 100
    pm   = growth['projected_profit_margin'].iloc[0] / 100

    n_pe = ((n_gw - tyld + 1)**10)*10
    f_pe = ((f_gw - tyld + 1)**10)*10
    n_ps = n_pe * pm
    f_ps = f_pe * pm

    # avoid dividing by zero market cap
    if mcap <= 0:
        df['Revenue_Per_Share'] = 0.0
    else:
        df['Revenue_Per_Share'] = (df['Revenue'] / mcap) * price

    def _val(row):
        if row['EPS'] > 0:
            n_val = row['EPS'] * n_pe
            f_val = row['EPS'] * f_pe
        else:
            n_val = row['Revenue_Per_Share'] * n_ps
            f_val = row['Revenue_Per_Share'] * f_ps
        basis = row['EPS'] if row['EPS']>0 else row['Revenue_Per_Share']
        btype = 'EPS' if row['EPS']>0 else 'Revenue'
        return pd.Series({
            'Basis_Value':      basis,
            'Basis_Type':       btype,
            'Nicks_Valuation':  n_val,
            'Finviz_Valuation': f_val
        })

    vals = df.apply(_val, axis=1)
    df = pd.concat([df, vals], axis=1)
    return df, n_pe, f_pe, n_ps, f_ps


def plot_valuation_chart(valuation_data, current_price, ticker, growth):
    fig, ax = plt.subplots(figsize=(10,6))
    years = valuation_data['Year']
    ax.plot(years, valuation_data['Nicks_Valuation'], label='Nicks Valuation', marker='o')
    ax.plot(years, valuation_data['Finviz_Valuation'], label='Finviz Valuation', marker='o')
    ax.plot(years, [current_price]*len(years), '--x', label='Current Price')
    ax.set_xlabel('Year'); ax.set_ylabel('USD')
    ax.set_title(f'Valuation for {ticker}'); ax.legend(); ax.grid(True)
    out = os.path.join(CHART_DIR, f"{ticker}_valuation_chart.png")
    plt.tight_layout(); plt.savefig(out); plt.close()
    print(f"Figure saved to {out}")


def generate_valuation_tables(ticker, df, growth, treasury_yield, current_price,
                              n_pe, f_pe, n_ps):
    # ... your existing table–building code, unchanged ...
    pass


def valuation_update(ticker, cursor, treasury_yield, marketcap, dashboard_data):
    finviz_five_yr(ticker, cursor)
    combined, growth, price, forecast, mcap = fetch_financial_valuation_data(ticker, DB_PATH)

    if forecast.empty or growth.empty:
        print("No forecast or growth data; skipping", ticker)
        return

    df_vals, n_pe, f_pe, n_ps, f_ps = calculate_valuations(
        combined, growth, treasury_yield, price, mcap
    )

    plot_valuation_chart(df_vals, price, ticker, growth)
    generate_valuation_tables(ticker, df_vals, growth, treasury_yield, price, n_pe, f_pe, n_ps)

    # ... dashboard_data appending, unchanged ...

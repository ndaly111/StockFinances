import requests
from bs4 import BeautifulSoup
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import os
import yfinance as yf
import csv


def log_valuation_data(ticker, nicks_ttm_valuation, nicks_forward_valuation,
                       finviz_ttm_valuation, finviz_forward_valuation):
    db_path = "Stock Data.db"
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('''
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
        cursor.execute('''
            INSERT INTO ValuationHistory
                (ticker, nicks_ttm_valuation, nicks_forward_valuation,
                 finviz_ttm_valuation, finviz_forward_valuation)
            VALUES (?, ?, ?, ?, ?);
        ''', (ticker, nicks_ttm_valuation, nicks_forward_valuation,
              finviz_ttm_valuation, finviz_forward_valuation))
        conn.commit()
        print(f"Inserted valuation data for {ticker} into ValuationHistory.")


def finviz_five_yr(ticker, cursor):
    """Fetches and stores the 5-year EPS growth percentage from Finviz."""
    url = f'https://finviz.com/quote.ashx?t={ticker}'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
    }
    resp = requests.get(url, headers=headers)
    if resp.status_code != 200:
        print(f"Failed to retrieve Finviz data for {ticker}, status {resp.status_code}")
        return

    soup = BeautifulSoup(resp.content, 'html.parser')
    label = soup.find('td', text='EPS next 5Y')
    if not label:
        print(f"No EPS next 5Y on Finviz for {ticker}")
        return

    val_td = label.find_next_sibling('td')
    raw = val_td.text.strip('%')
    if not raw:
        print(f"Empty Finviz 5Y growth for {ticker}")
        return

    try:
        growth = float(raw)
    except ValueError:
        print(f"Invalid Finviz 5Y growth '{raw}' for {ticker}")
        return

    # ensure Tickers_Info exists and ticker row present
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Tickers_Info(
            ticker TEXT PRIMARY KEY,
            nicks_growth_rate REAL,
            FINVIZ_5yr_gwth REAL,
            projected_profit_margin REAL
        );
    ''')
    cursor.execute('SELECT 1 FROM Tickers_Info WHERE ticker=?', (ticker,))
    if not cursor.fetchone():
        cursor.execute('INSERT INTO Tickers_Info(ticker) VALUES(?)', (ticker,))
    cursor.execute('''
        UPDATE Tickers_Info
           SET FINVIZ_5yr_gwth=?
         WHERE ticker=?;
    ''', (growth, ticker))
    cursor.connection.commit()
    print(f"Stored Finviz 5Y growth for {ticker}: {growth}%")


def fetch_financial_valuation_data(ticker, db_path):
    """Load TTM and forward Revenue/EPS from the local SQLite DB."""
    stock = yf.Ticker(ticker)
    current_price = stock.info.get('currentPrice')

    with sqlite3.connect(db_path) as conn:
        # TTM
        ttm_df = pd.read_sql_query("""
            SELECT 'TTM' AS Year, TTM_Revenue AS Revenue, TTM_EPS AS EPS
              FROM TTM_Data
             WHERE Symbol=?
          ORDER BY Last_Updated DESC
             LIMIT 1;
        """, conn, params=(ticker,))
        # Forecast
        fwd_df = pd.read_sql_query("""
            SELECT strftime('%Y', Date) AS Year,
                   ForwardRevenue AS Revenue,
                   ForwardEPS     AS EPS
              FROM ForwardFinancialData
             WHERE Ticker=?
          ORDER BY Date ASC;
        """, conn, params=(ticker,))
        # Growth / margins
        gv_df = pd.read_sql_query("""
            SELECT nicks_growth_rate, FINVIZ_5yr_gwth, projected_profit_margin
              FROM Tickers_Info
             WHERE ticker=?;
        """, conn, params=(ticker,))

    combined = pd.concat([ttm_df, fwd_df], ignore_index=True)
    return combined, gv_df, current_price, fwd_df


def calculate_valuations(combined_data, growth_values, treasury_yield, current_price, marketcap):
    ty = float(treasury_yield) / 100
    ngr = (growth_values['nicks_growth_rate'].iloc[0] or 0) / 100
    fgr = (growth_values['FINVIZ_5yr_gwth'].iloc[0] or 0) / 100
    ppm = (growth_values['projected_profit_margin'].iloc[0] or 0) / 100

    n_pe = ((ngr - ty + 1) ** 10) * 10
    f_pe = ((fgr - ty + 1) ** 10) * 10
    n_ps = n_pe * ppm
    f_ps = f_pe * ppm

    combined_data['Revenue_Per_Share'] = combined_data['Revenue'] / marketcap * current_price

    def row_val(r):
        if r.EPS > 0:
            return r.EPS, 'EPS', r.EPS * n_pe, r.EPS * f_pe
        else:
            rps = r.Revenue_Per_Share
            return rps, 'Revenue', rps * n_ps, rps * f_ps

    vals = combined_data.apply(lambda r: row_val(r), axis=1)
    combined_data[['Basis_Value','Basis_Type','Nicks_Valuation','Finviz_Valuation']] = \
        pd.DataFrame(vals.tolist(), index=combined_data.index)

    return combined_data, n_pe, f_pe, n_ps, f_ps


def plot_valuation_chart(valuation_data, current_price, ticker, growth_values):
    fig, ax = plt.subplots(figsize=(10,6))
    years = valuation_data['Year']
    if pd.notna(growth_values['nicks_growth_rate'].iloc[0]):
        ax.plot(years, valuation_data['Nicks_Valuation'],
                marker='o', label='Nicks Valuation')
    if pd.notna(growth_values['FINVIZ_5yr_gwth'].iloc[0]):
        ax.plot(years, valuation_data['Finviz_Valuation'],
                marker='o', label='Finviz Valuation')
    ax.plot(years, [current_price]*len(years),
            linestyle='--', marker='x', label='Current Price')
    ax.set_title(f"Valuation Comparison for {ticker}")
    ax.set_xlabel("Year")
    ax.set_ylabel("USD")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    out = os.path.join("charts", f"{ticker}_valuation_chart.png")
    plt.tight_layout()
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to {out}")


def fetch_stock_data(ticker):
    s = yf.Ticker(ticker)
    cp = s.info.get('currentPrice')
    fwd_eps = s.info.get('forwardEps')
    pe = s.info.get('trailingPE')
    ps = s.info.get('priceToSalesTrailing12Months')
    return cp, pe, ps, (cp / fwd_eps if fwd_eps else None)


def generate_valuation_tables(ticker, combined_data, growth_values,
                              treasury_yield, current_price, n_pe, f_pe, n_ps):
    cp, pe, ps, fwd_pe = fetch_stock_data(ticker)
    # Table 1
    t1 = {
        "Share Price": [f"${cp:,.2f}"],
        "Treasury Yield": [f"{float(treasury_yield):.2f}%"],
        "Fair P/E (Nicks)": [f"{n_pe:.1f}"],
        "Fair P/E (Finviz)": [f"{f_pe:.1f}"],
        "Fair P/S (Nicks)": [f"{n_ps:.3f}"],
        "Current P/E": [f"{pe:.1f}" if pe else "N/A"],
        "Current P/S": [f"{ps:.2f}" if ps else "N/A"]
    }
    df1 = pd.DataFrame(t1)
    path1 = os.path.join("charts", f"{ticker}_valuation_info.html")
    df1.to_html(path1, index=False, escape=False)
    print(f"Saved summary → {path1}")

    # Table 2
    def fmt(x):
        return f"${x:,.2f}" if pd.notna(x) else "N/A"
    df2 = combined_data.copy()
    df2['Nicks_Valuation'] = df2['Nicks_Valuation'].apply(fmt)
    df2['Finviz_Valuation'] = df2['Finviz_Valuation'].apply(fmt)
    path2 = os.path.join("charts", f"{ticker}_valuation_table.html")
    df2.to_html(path2, index=False, escape=False)
    print(f"Saved detail → {path2}")


def process_update_growth_csv(file_path, db_path):
    """Reads update_growth.csv and updates Tickers_Info."""
    if not os.path.exists(file_path):
        print(f"{file_path} not found")
        return
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    with open(file_path, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) not in (2,3):
                print("Skipping invalid row", row); continue
            ticker = row[0].upper()
            rate   = row[1].strip()
            pm     = row[2].strip() if len(row)==3 else None

            try:
                rate_f = float(rate)
            except:
                print(f"Bad rate {rate} for {ticker}"); continue

            cur.execute('''
                CREATE TABLE IF NOT EXISTS Tickers_Info(
                    ticker TEXT PRIMARY KEY,
                    nicks_growth_rate REAL,
                    FINVIZ_5yr_gwth REAL,
                    projected_profit_margin REAL
                );
            ''')
            cur.execute('SELECT 1 FROM Tickers_Info WHERE ticker=?',(ticker,))
            if not cur.fetchone():
                cur.execute('INSERT INTO Tickers_Info(ticker) VALUES(?)',(ticker,))
            cur.execute('''
                UPDATE Tickers_Info
                   SET nicks_growth_rate=?,
                       projected_profit_margin=?
                 WHERE ticker=?;
            ''', (rate_f, pm, ticker))
    conn.commit()
    conn.close()
    open(file_path, 'w').close()
    print(f"Processed and cleared {file_path}")


def valuation_update(ticker, cursor, treasury_yield, marketcap, dashboard_data):
    db = "Stock Data.db"
    finviz_five_yr(ticker, cursor)
    combined, gv, cp, fwd_df = fetch_financial_valuation_data(ticker, db)
    if fwd_df.empty:
        print(f"No forecast for {ticker}, skipping"); return

    if gv.empty or (pd.isna(gv.iloc[0,0]) and pd.isna(gv.iloc[0,1])):
        print("Missing growth data, skipping"); return

    combined, n_pe, f_pe, n_ps, _ = calculate_valuations(
        combined, gv, treasury_yield, cp, marketcap
    )
    plot_valuation_chart(combined, cp, ticker, gv)
    generate_valuation_tables(ticker, combined, gv, treasury_yield, cp, n_pe, f_pe, n_ps)

    # Optionally log to history table
    try:
        ttm_val    = combined.loc[0,'Nicks_Valuation']
        fwd_val    = combined.loc[1,'Nicks_Valuation']
        fv_ttm_val = combined.loc[0,'Finviz_Valuation']
        fv_fwd_val = combined.loc[1,'Finviz_Valuation']
        log_valuation_data(ticker, ttm_val, fwd_val, fv_ttm_val, fv_fwd_val)
    except Exception:
        pass

import yfinance as yf
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
from functools import lru_cache

# ─────────────────────────────────────────────────────────────────────────────
# Database connection setup
# ─────────────────────────────────────────────────────────────────────────────
def get_db_connection(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("CREATE INDEX IF NOT EXISTS idx_symbol ON Annual_Data(Symbol);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_symbol_quarter ON TTM_Data(Symbol, Quarter);")
    conn.commit()
    return conn

def fetch_ticker_data(ticker, cursor):
    try:
        cursor.execute("PRAGMA table_info(Annual_Data)")
        cols = [c[1] for c in cursor.fetchall()]
        cursor.execute("SELECT * FROM Annual_Data WHERE Symbol = ? ORDER BY Date ASC", (ticker,))
        rows = [dict(zip(cols, r)) for r in cursor.fetchall()]
        return rows if rows else None
    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
        return None

def fetch_ttm_data(ticker, cursor):
    try:
        cursor.execute("PRAGMA table_info(TTM_Data)")
        cols = [c[1] for c in cursor.fetchall()]
        cursor.execute("SELECT * FROM TTM_Data WHERE Symbol = ? ORDER BY Quarter DESC", (ticker,))
        rows = [dict(zip(cols, r)) for r in cursor.fetchall()]
        return rows if rows else None
    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
        return None

# ─────────────────────────────────────────────────────────────────────────────
# Date utilities
# ─────────────────────────────────────────────────────────────────────────────

def get_latest_annual_data_date(ticker_data):
    """
    Returns the most recent 'Date' from the list-of-dicts or DataFrame
    in ticker_data, or None if it can't parse any.
    """
    if isinstance(ticker_data, pd.DataFrame):
        if ticker_data.empty:
            return None
        try:
            dates = pd.to_datetime(ticker_data['Date'], format='%Y-%m-%d', errors='coerce')
            return dates.max()
        except Exception as e:
            logging.error(f"Error parsing dates from DataFrame: {e}")
            return None

    if not ticker_data:
        return None

    # assume list of dicts
    parsed = []
    for row in ticker_data:
        d = row.get('Date')
        if isinstance(d, str):
            try:
                parsed.append(datetime.strptime(d, '%Y-%m-%d'))
            except Exception:
                continue
    return max(parsed) if parsed else None

def calculate_next_check_date(latest_date, months):
    return None if latest_date is None else latest_date + timedelta(days=30*months)

def needs_update(latest_date, months):
    if latest_date is None:
        return True
    nxt = calculate_next_check_date(latest_date, months)
    return nxt is None or nxt <= datetime.now()

def check_null_fields(data, fields):
    if not isinstance(data, list):
        return False
    for entry in data:
        if not isinstance(entry, dict):
            continue
        for f in fields:
            if entry.get(f) in (None, ""):
                return True
    return False

# ─────────────────────────────────────────────────────────────────────────────
# Cleaning & fetching from Yahoo
# ─────────────────────────────────────────────────────────────────────────────
def clean_financial_data(df):
    df.dropna(subset=['Revenue','Net_Income','EPS'], how='all', inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df.infer_objects(copy=False)
    return df

@lru_cache(maxsize=32)
def fetch_annual_data_from_yahoo(ticker):
    logging.info("Fetching annual data from Yahoo Finance")
    try:
        stock = yf.Ticker(ticker)
        fin = stock.financials
        if fin.empty:
            return pd.DataFrame()
        fin = fin.T
        fin['Date'] = fin.index
        mapping = {'Total Revenue':'Revenue','Net Income':'Net_Income','Basic EPS':'EPS'}
        rename = {y:db for y,db in mapping.items() if y in fin.columns}
        if len(rename)<len(mapping):
            missing = set(mapping.values())-set(rename.values())
            logging.warning(f"Missing {missing} for {ticker}")
            return pd.DataFrame()
        fin.rename(columns=rename, inplace=True)
        return clean_financial_data(fin)
    except Exception as e:
        logging.error(f"Error fetching annual for {ticker}: {e}")
        return pd.DataFrame()

@lru_cache(maxsize=32)
def fetch_ttm_data_from_yahoo(ticker):
    logging.info("Fetching TTM data from Yahoo Finance")
    try:
        stock = yf.Ticker(ticker)
        q = stock.quarterly_financials
        if q is None or q.empty:
            return None
        data = {}
        try:
            data['TTM_Revenue']    = q.loc['Total Revenue'][:4].sum()
            data['TTM_Net_Income'] = q.loc['Net Income'][:4].sum()
        except KeyError:
            data['TTM_Revenue']=data['TTM_Net_Income']=None
        data['TTM_EPS'] = stock.info.get('trailingEps')
        data['Shares_Outstanding'] = stock.info.get('sharesOutstanding')
        data['Quarter'] = q.columns[0].strftime('%Y-%m-%d')
        return data
    except Exception as e:
        logging.error(f"Error fetching TTM for {ticker}: {e}")
        return None

# ─────────────────────────────────────────────────────────────────────────────
# Storage
# ─────────────────────────────────────────────────────────────────────────────
def store_annual_data(ticker, df, cursor):
    logging.info("Storing annual data")
    for _, row in df.iterrows():
        d = row['Date']
        ds = d.strftime('%Y-%m-%d') if isinstance(d, pd.Timestamp) else d
        cursor.execute("""
            SELECT 1 FROM Annual_Data
             WHERE Symbol=? AND Date=? 
               AND Revenue IS NOT NULL AND Net_Income IS NOT NULL AND EPS IS NOT NULL
        """,(ticker,ds))
        if cursor.fetchone():
            continue
        try:
            cursor.execute("""
            INSERT INTO Annual_Data
              (Symbol,Date,Revenue,Net_Income,EPS,Last_Updated)
            VALUES(?,?,?,?,?,CURRENT_TIMESTAMP)
            ON CONFLICT(Symbol,Date) DO UPDATE
             SET Revenue=EXCLUDED.Revenue,
                 Net_Income=EXCLUDED.Net_Income,
                 EPS=EXCLUDED.EPS,
                 Last_Updated=CURRENT_TIMESTAMP
             WHERE Revenue IS NULL OR Net_Income IS NULL OR EPS IS NULL;
            """,(ticker,ds,row['Revenue'],row['Net_Income'],row['EPS']))
            cursor.connection.commit()
        except sqlite3.Error as e:
            logging.error(f"DB error storing annual {ticker}: {e}")

def store_ttm_data(ticker, data, cursor):
    logging.info("Storing TTM data")
    vals=(
        ticker,
        data.get('TTM_Revenue'),
        data.get('TTM_Net_Income'),
        data.get('TTM_EPS'),
        data.get('Shares_Outstanding'),
        data.get('Quarter'),
    )
    try:
        cursor.execute("""
        INSERT OR REPLACE INTO TTM_Data
          (Symbol,TTM_Revenue,TTM_Net_Income,TTM_EPS,Shares_Outstanding,Quarter,Last_Updated)
        VALUES(?,?,?,?,?,?           ,CURRENT_TIMESTAMP);
        """, vals)
        cursor.connection.commit()
    except sqlite3.Error as e:
        logging.error(f"DB error storing TTM {ticker}: {e}")

def handle_ttm_duplicates(ticker, cursor):
    logging.info("Checking for duplicate TTM entries")
    try:
        cursor.execute("SELECT * FROM TTM_Data WHERE Symbol=? ORDER BY Quarter DESC",(ticker,))
        rows=cursor.fetchall()
        if len(rows)>1:
            keep=rows[0][5]  # Quarter column
            cursor.execute("DELETE FROM TTM_Data WHERE Symbol=? AND Quarter<>?",(ticker,keep))
            cursor.connection.commit()
            logging.info(f"Removed duplicates, kept {keep}")
            return True
    except sqlite3.Error as e:
        logging.error(f"DB error dedup TTM {ticker}: {e}")
    return False

# ─────────────────────────────────────────────────────────────────────────────
# Chart & table helpers
# ─────────────────────────────────────────────────────────────────────────────
def chart_needs_update(path, last_update, *_):
    if not os.path.exists(path):
        return True
    lu = datetime.fromtimestamp(os.path.getmtime(path))
    return last_update > lu

def create_formatted_dataframe(df):
    def fmt(v):
        if pd.isna(v): return "N/A"
        if abs(v)>=1e9: return f"${v/1e9:,.1f}B"
        if abs(v)>=1e6: return f"${v/1e6:,.1f}M"
        return f"${v/1e3:,.1f}K"
    df['Formatted_Revenue']    = df['Revenue'].apply(fmt)
    df['Formatted_Net_Income'] = df['Net_Income'].apply(fmt)
    df['Formatted_EPS']        = df['EPS'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
    return df

def add_eps_value_labels(ax, bars, df):
    for bar in bars:
        h=bar.get_height()
        lbl=df.loc[df['EPS']==h,'Formatted_EPS'].iat[0]
        off=12 if h>=0 else -12
        ax.annotate(lbl, xy=(bar.get_x()+bar.get_width()/2,h),
                    xytext=(0,off), textcoords="offset points",
                    ha='center', va='bottom')

def generate_eps_chart(ticker, out_dir, df):
    if df.empty: return
    path=os.path.join(out_dir,f"{ticker}_eps_chart.png")
    df['EPS']=pd.to_numeric(df['EPS'],errors='coerce')
    pos=np.arange(len(df)); w=0.4
    fig,ax=plt.subplots(figsize=(8,5))
    bars=ax.bar(pos,df['EPS'],w,color='teal')
    ax.grid(True,axis='y',linestyle='--',linewidth=0.5)
    ax.axhline(0,color='black',linewidth=2)
    ax.set_ylabel('EPS'); ax.set_title(f"EPS Chart for {ticker}")
    ax.set_xticks(pos); ax.set_xticklabels(df['Date'],rotation=0)
    add_eps_value_labels(ax,bars,df)
    plt.tight_layout()
    os.makedirs(out_dir,exist_ok=True)
    plt.savefig(path); plt.close()

def add_value_labels(ax,bars,df,colname,sf):
    for bar in bars:
        h=bar.get_height()
        scaled=h*sf
        col='Net_Income' if 'Net_Income' in colname else 'Revenue'
        matches=df.loc[np.isclose(df[col],scaled,atol=1e-2),colname]
        lbl=matches.iat[0] if not matches.empty else "N/A"
        off=3 if h>=0 else -12
        ax.annotate(lbl, xy=(bar.get_x()+bar.get_width

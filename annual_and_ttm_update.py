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
# Chart & table generation helpers
# ─────────────────────────────────────────────────────────────────────────────
def chart_needs_update(path, last_update, *_):
    if not os.path.exists(path):
        return True
    lu = datetime.fromtimestamp(os.path.getmtime(path))
    return last_update > lu

def create_formatted_dataframe(df):
    def fmt(v):
        if pd.isna(v):
            return "N/A"
        if abs(v)>=1e9:
            return f"${v/1e9:,.1f}B"
        if abs(v)>=1e6:
            return f"${v/1e6:,.1f}M"
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
        ax.annotate(lbl, xy=(bar.get_x()+bar.get_width()/2,h),
                    xytext=(0,off), textcoords="offset points",
                    ha='center', va='bottom')

def generate_revenue_net_income_chart(df,ticker,path):
    df2=create_formatted_dataframe(df.copy())
    df2['Revenue']=pd.to_numeric(df2['Revenue'].replace('[\$,]','',regex=True),errors='coerce')
    df2['Net_Income']=pd.to_numeric(df2['Net_Income'].replace('[\$,]','',regex=True),errors='coerce')
    df2['Date']=df2['Date'].astype(str)
    pos=np.arange(len(df2)); w=0.3
    fig,ax=plt.subplots(figsize=(10,6))
    max_net=df2['Net_Income'].max()
    if abs(max_net)>=1e9:
        sf,le,yl=1e9,'B','Amount (Billions $)'
    else:
        sf,le,yl=1e6,'M','Amount (Millions $)'
    combined=pd.concat([df2['Revenue'],df2['Net_Income']])/sf
    buf=combined.abs().max()*0.2
    mn=combined.max(); mi=df2['Net_Income'].min()/sf
    bars1=ax.bar(pos-w/2,df2['Revenue']/sf,w,label=f"Revenue ({le})",color='green')
    bars2=ax.bar(pos+w/2,df2['Net_Income']/sf,w,label=f"Net Income ({le})",color='blue')
    ax.set_ylabel(yl); ax.set_ylim(mi-buf if mi<0 else 0, mn+buf)
    ax.set_title(f"Revenue and Net Income for {ticker}")
    ax.set_xticks(pos); ax.set_xticklabels(df2['Date'],rotation=0)
    ax.legend(); ax.grid(True,axis='y',linestyle='--',linewidth=0.5); ax.axhline(0,color='black',linewidth=1)
    add_value_labels(ax,bars1,df2,'Formatted_Revenue',sf)
    add_value_labels(ax,bars2,df2,'Formatted_Net_Income',sf)
    plt.tight_layout()
    os.makedirs(os.path.dirname(path) or ".",exist_ok=True)
    plt.savefig(path); plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# ←— MISSING FUNCTION RE-ADDED HERE
def generate_financial_charts(ticker, charts_output_dir, financial_data):
    print("Generating financial charts")
    if financial_data.empty:
        print("---chart data empty; skip")
        return

    rev_path = os.path.join(charts_output_dir, f"{ticker}_revenue_net_income_chart.png")
    eps_path = os.path.join(charts_output_dir, f"{ticker}_eps_chart.png")

    if 'Last_Updated' in financial_data.columns:
        lu = pd.to_datetime(financial_data['Last_Updated'], errors='coerce').max()
    else:
        lu = datetime.now()

    if chart_needs_update(rev_path, lu):
        generate_revenue_net_income_chart(financial_data, ticker, rev_path)
        print("---revenue/net chart updated")

    if chart_needs_update(eps_path, lu):
        generate_eps_chart(ticker, charts_output_dir, financial_data)
        print("---EPS chart updated")

    generate_financial_data_table_html(ticker, financial_data, charts_output_dir)

# ─────────────────────────────────────────────────────────────────────────────
def calculate_and_format_changes(df):
    df.sort_values('Date',inplace=True)
    for col in ['Revenue','Net_Income','EPS']:
        if df[col].dtype=='object':
            df[col]=df[col].replace('[\$,MK]','',regex=True).astype(float)*1e3
    for col in ['Revenue','Net_Income','EPS']:
        c2=f"{col}_Change"
        df[c2]=df[col].pct_change()*100
        df[c2]=df[c2].apply(lambda x:f"{x:.1f}%" if pd.notna(x) else "N/A")
    return df

def style_changes(val):
    if isinstance(val,str) and '%' in val:
        return f"color:{'red' if '-' in val else 'green'};"
    return ''

def generate_financial_data_table_html(ticker, df, charts_output_dir):
    df2=calculate_and_format_changes(df.copy())
    keep=['Date','Formatted_Revenue','Formatted_Net_Income','Formatted_EPS',
          'Revenue_Change','Net_Income_Change','EPS_Change']
    df2=df2[keep]
    df2.columns=['Date','Revenue','Net Income','EPS',
                 'Revenue Change','Net Income Change','EPS Change']
    avg=df2[['Revenue Change','Net Income Change','EPS Change']].replace('N/A',np.nan)\
           .apply(lambda x:pd.to_numeric(x.str.replace('%','')),axis=0).mean()\
           .apply(lambda x:f"{x:.1f}%" if pd.notna(x) else "N/A")
    avg_row=pd.Series(['Average']+['']*3+avg.tolist(),index=df2.columns)
    df2=pd.concat([df2,avg_row.to_frame().T],ignore_index=True)
    styled=df2.style.applymap(style_changes,subset=['Revenue Change','Net Income Change','EPS Change'])
    html=styled.to_html()
    path=os.path.join(charts_output_dir,f"{ticker}_rev_net_table.html")
    os.makedirs(charts_output_dir,exist_ok=True)
    with open(path,'w',encoding='utf-8') as f:
        f.write(html)
    print(f"Financial data table for {ticker} saved to {path}")

def prepare_data_for_charts(ticker, cursor):
    print("Preparing data for charts")
    cursor.execute("SELECT Date,Revenue,Net_Income,EPS,Last_Updated FROM Annual_Data WHERE Symbol=? ORDER BY Date",(ticker,))
    ann=cursor.fetchall()
    print("---annual data",ann)

    cursor.execute("SELECT 'TTM' AS Date,TTM_Revenue,TTM_Net_Income,TTM_EPS,Last_Updated FROM TTM_Data WHERE Symbol=?",(ticker,))
    ttm=cursor.fetchall()
    print("---ttm data",ttm)

    cursor.execute("SELECT 'TTM' AS Date,TTM_Revenue,TTM_Net_Income,TTM_EPS,Quarter,Last_Updated FROM TTM_Data WHERE Symbol=?",(ticker,))
    ttm2=cursor.fetchall()
    ann_df=pd.DataFrame(ann,columns=['Date','Revenue','Net_Income','EPS','Last_Updated'])
    ttm_df=pd.DataFrame(ttm,columns=['Date','Revenue','Net_Income','EPS','Last_Updated'])
    ttm2_df=pd.DataFrame(ttm2,columns=['Date','Revenue','Net_Income','EPS','Quarter','Last_Updated'])
    ann_df['Last_Updated']=pd.to_datetime(ann_df['Last_Updated'])
    ttm_df['Last_Updated']=pd.to_datetime(ttm_df['Last_Updated'])

    if not ttm_df.empty:
        q=ttm2_df.loc[0,'Quarter']
        if isinstance(q,str) and q.strip():
            ttm_df.at[0,'Date']=f"TTM {q}"

    if not ann and not ttm:
        return pd.DataFrame()

    ann_df.dropna(axis=1,how='all',inplace=True)
    ttm_df.dropna(axis=1,how='all',inplace=True)
    df=pd.concat([ann_df,ttm_df],ignore_index=True)
    df['Revenue']=pd.to_numeric(df['Revenue'],errors='coerce')
    df['Net_Income']=pd.to_numeric(df['Net_Income'],errors='coerce')
    df['EPS']=pd.to_numeric(df['EPS'],errors='coerce')
    df=clean_financial_data(df)
    df['Date']=df['Date'].astype(str)
    df.sort_values('Date',inplace=True)
    return create_formatted_dataframe(df)

def annual_and_ttm_update(ticker, db_path):
    conn=get_db_connection(db_path); cur=conn.cursor()
    ann=fetch_ticker_data(ticker,cur)
    if not ann:
        nya=fetch_annual_data_from_yahoo(ticker)
        if not nya.empty:
            store_annual_data(ticker, nya, cur)
            ann=nya

    ttm=fetch_ttm_data(ticker,cur)
    if not ttm:
        nyt=fetch_ttm_data_from_yahoo(ticker)
        if nyt:
            store_ttm_data(ticker, nyt, cur)
            ttm=[nyt]

    handle_ttm_duplicates(ticker,cur)

    au=False; tu=False

    if ann:
        lad=get_latest_annual_data_date(ann)
        au=needs_update(lad,13) or check_null_fields(ann,['Revenue','Net_Income','EPS'])

    if ttm:
        valid=[r['Quarter'] for r in ttm if isinstance(r.get('Quarter'),str) and r['Quarter'].strip()]
        if valid:
            lttm=max(datetime.strptime(q,'%Y-%m-%d') for q in valid)
        else:
            lttm=None
        tu=(lttm is None) or needs_update(lttm,4) or check_null_fields(ttm,['TTM_Revenue','TTM_Net_Income','TTM_EPS'])

    if au:
        nya=fetch_annual_data_from_yahoo(ticker)
        if not nya.empty:
            store_annual_data(ticker, nya, cur)
    if tu:
        nyt=fetch_ttm_data_from_yahoo(ticker)
        if nyt:
            store_ttm_data(ticker, nyt, cur)

    df=prepare_data_for_charts(ticker,cur)
    annual_and_ttm_update_charts="charts"
    generate_financial_charts(ticker, annual_and_ttm_update_charts, df)

    conn.close()
    logging.debug(f"Update for {ticker} completed")

if __name__ == "__main__":
    annual_and_ttm_update("PG", "Stock Data.db")

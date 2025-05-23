import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from datetime import datetime
import yfinance as yf

DB_PATH = "Stock Data.db"

def clean_value(val):
    if pd.isna(val):
        return None
    if isinstance(val, pd.Timestamp) or isinstance(val, datetime):
        return val.isoformat()
    if isinstance(val, (int, float, str, type(None))):
        return val
    return str(val)

def fetch_and_store_income_statement(ticker: str) -> pd.DataFrame:
    stock = yf.Ticker(ticker)
    df = stock.quarterly_financials.transpose()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS IncomeStatement (
        ticker              TEXT,
        period_ending       TEXT,
        total_revenue       REAL,
        cost_of_revenue     REAL,
        research_and_development REAL,
        selling_general_admin    REAL,
        operating_income         REAL,
        PRIMARY KEY (ticker, period_ending)
    );
    """)

    for idx, row in df.iterrows():
        period_ending   = idx.to_pydatetime() if isinstance(idx, pd.Timestamp) else idx
        cursor.execute("""
        INSERT OR REPLACE INTO IncomeStatement
          (ticker, period_ending, total_revenue, cost_of_revenue,
           research_and_development, selling_general_admin, operating_income)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            clean_value(ticker),
            clean_value(period_ending),
            clean_value(row.get('Total Revenue')),
            clean_value(row.get('Cost Of Revenue')),
            clean_value(row.get('Research Development')),
            clean_value(row.get('Selling General Administrative')),
            clean_value(row.get('Operating Income')),
        ))

    conn.commit()
    conn.close()
    return df

def plot_revenue_vs_expenses(ticker: str):
    # 1. Load from DB
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(f"""
        SELECT * FROM IncomeStatement
        WHERE ticker = '{ticker}'
    """, conn)
    conn.close()

    # 2. Prepare year and numeric-only grouping
    df['period_ending'] = pd.to_datetime(df['period_ending'])
    df['year'] = df['period_ending'].dt.year

    # only sum the numeric cols so we don't hit datetime64 sum errors
    df_yearly = df.groupby('year', as_index=False)[[
        'total_revenue',
        'cost_of_revenue',
        'research_and_development',
        'selling_general_admin'
    ]].sum()

    years   = df_yearly['year'].astype(str)
    cost    = df_yearly['cost_of_revenue']
    rnd     = df_yearly['research_and_development']
    sga     = df_yearly['selling_general_admin']
    # example fines: $0 except $5M in 2019
    fines   = [0 if y != 2019 else 5e6 for y in df_yearly['year']]
    revenue = df_yearly['total_revenue']

    # 3. Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(years, cost,  label='Cost of Revenue',                  color='dimgray')
    ax.bar(years, rnd,   label='Research and Development',        bottom=cost,               color='blue')
    ax.bar(years, sga,   label='Sales and Marketing',             bottom=cost + rnd,         color='mediumpurple')
    ax.bar(years, fines, label='European Commission Fines',       bottom=cost + rnd + sga,   color='red')
    ax.bar(years, revenue, label='Revenue', color='darkgreen', alpha=0.8)

    ax.set_ylabel("Amount ($M)")
    ax.set_title("Revenue vs Expenses")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${int(x/1e6)}M"))
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Fetching income statement for AAPL…")
    fetch_and_store_income_statement("AAPL")
    print("Plotting revenue vs. expenses…")
    plot_revenue_vs_expenses("AAPL")

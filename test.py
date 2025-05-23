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
        total_revenue   = row.get('Total Revenue')
        cost_of_revenue = row.get('Cost Of Revenue')
        r_and_d         = row.get('Research Development')
        sga             = row.get('Selling General Administrative')
        op_income       = row.get('Operating Income')

        cursor.execute("""
        INSERT OR REPLACE INTO IncomeStatement
          (ticker, period_ending, total_revenue, cost_of_revenue,
           research_and_development, selling_general_admin, operating_income)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            clean_value(ticker),
            clean_value(period_ending),
            clean_value(total_revenue),
            clean_value(cost_of_revenue),
            clean_value(r_and_d),
            clean_value(sga),
            clean_value(op_income),
        ))

    conn.commit()
    conn.close()
    print(df.columns.tolist())
    return df

def plot_revenue_vs_expenses(ticker: str):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(f"""
        SELECT * FROM IncomeStatement
        WHERE ticker = '{ticker}'
    """, conn)
    conn.close()

    df['period_ending'] = pd.to_datetime(df['period_ending'])
    df['year'] = df['period_ending'].dt.year
    df_yearly = df.groupby('year', as_index=False).sum()

    years = df_yearly['year'].astype(str)
    cost = df_yearly['cost_of_revenue']
    rnd = df_yearly['research_and_development']
    sga = df_yearly['selling_general_admin']
    fines = [0 if y != 2019 else 5e6 for y in df_yearly['year']]  # Example: $5M in 2019
    revenue = df_yearly['total_revenue']

    fig, ax = plt.subplots(figsize=(10, 6))
    bar1 = ax.bar(years, cost, label='Cost of Revenue', color='dimgray')
    bar2 = ax.bar(years, rnd, label='Research and Development', bottom=cost, color='blue')
    bar3 = ax.bar(years, sga, label='Sales and Marketing', bottom=cost + rnd, color='mediumpurple')
    bar4 = ax.bar(years, fines, label='European Commission Fines', bottom=cost + rnd + sga, color='red')
    ax.bar(years, revenue, label='Revenue', color='darkgreen', alpha=0.8)

    ax.set_ylabel("Amount ($M)")
    ax.set_title("Revenue vs Expenses")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${int(x/1e6)}M"))
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

# Example mini-main
if __name__ == "__main__":
    print("Fetching income statement for AAPL")
    df_aapl = fetch_and_store_income_statement("AAPL")
    print(df_aapl)

    print("Plotting revenue vs. expenses...")
    plot_revenue_vs_expenses("AAPL")

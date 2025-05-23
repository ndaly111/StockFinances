import sqlite3
import pandas as pd
from datetime import datetime
import yfinance as yf

DB_PATH = "Stock Data.db"

def clean_value(val):
    """
    Convert any unsupported types into sqlite-friendly types:
    - NaN / NaT → None
    - pandas Timestamp → ISO string
    - Other non-primitive → str()
    """
    if pd.isna(val):
        return None
    # pandas Timestamp
    if isinstance(val, pd.Timestamp):
        return val.isoformat()
    # datetime.datetime
    if isinstance(val, datetime):
        return val.isoformat()
    # everything else (int, float, str) passes through
    if isinstance(val, (int, float, str, type(None))):
        return val
    # fallback for any other type
    return str(val)

def fetch_and_store_income_statement(ticker: str) -> pd.DataFrame:
    """
    Fetches quarterly income statement data for `ticker` via yfinance,
    stores it into the IncomeStatement table, and returns a DataFrame.
    """
    # 1. Fetch the raw Income Statement
    stock = yf.Ticker(ticker)
    df = stock.quarterly_financials.transpose()  # rows = quarters

    # 2. Prepare DB & table
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

    # 3. Insert each quarter
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
    # right after you fetch df:
    print(df.columns.tolist())

    # 4. Return the raw DataFrame for any downstream use
    return df

# Example mini-main for testing
if __name__ == "__main__":
    print("Fetching income statement for AAPL")
    df_aapl = fetch_and_store_income_statement("AAPL")
    print(df_aapl)

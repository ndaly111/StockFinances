# expense_reports.py

import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from datetime import datetime
import yfinance as yf

DB_PATH = "Stock Data.db"
OUTPUT_DIR = "charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def clean_value(val):
    if pd.isna(val):
        return None
    if isinstance(val, (pd.Timestamp, datetime)):
        return val.isoformat()
    if isinstance(val, (int, float, str)) or val is None:
        return val
    return str(val)


def fetch_and_store_income_statement(ticker: str) -> pd.DataFrame:
    print(f"\n--- Fetching financials for {ticker} ---")
    stock = yf.Ticker(ticker)
    df = stock.quarterly_financials.transpose()
    print("Fetched columns:", list(df.columns))

    # Ensure required columns exist
    expected = {
        'Total Revenue': 0.0,
        'Cost Of Revenue': 0.0,
        'Research Development': 0.0,
        'Selling General Administrative': 0.0,
        'Operating Income': 0.0
    }
    for col, default in expected.items():
        if col not in df.columns:
            df[col] = default
            print(f"⚠️  Missing: {col} → defaulting to 0.0")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS IncomeStatement (
        ticker TEXT,
        period_ending TEXT,
        total_revenue REAL,
        cost_of_revenue REAL,
        research_and_development REAL,
        selling_general_admin REAL,
        operating_income REAL,
        PRIMARY KEY (ticker, period_ending)
    );
    """)
    for idx, row in df.iterrows():
        pe = idx.to_pydatetime() if isinstance(idx, pd.Timestamp) else idx
        cursor.execute("""
        INSERT OR REPLACE INTO IncomeStatement
        (ticker, period_ending, total_revenue, cost_of_revenue,
         research_and_development, selling_general_admin, operating_income)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            ticker, clean_value(pe),
            clean_value(row['Total Revenue']),
            clean_value(row['Cost Of Revenue']),
            clean_value(row['Research Development']),
            clean_value(row['Selling General Administrative']),
            clean_value(row['Operating Income']),
        ))
    conn.commit()
    conn.close()
    print("✅ Income statement stored in DB.")
    return df


def load_yearly_data(ticker: str) -> pd.DataFrame:
    print("--- Loading & aggregating yearly data ---")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT period_ending, total_revenue, cost_of_revenue,
               research_and_development, selling_general_admin
        FROM IncomeStatement
        WHERE ticker = ?
    """, conn, params=(ticker,))
    conn.close()

    df['period_ending'] = pd.to_datetime(df['period_ending'])
    df['year'] = df['period_ending'].dt.year

    # Only aggregate numeric columns
    numeric_cols = ['total_revenue', 'cost_of_revenue',
                    'research_and_development', 'selling_general_admin']

    grouped = df.groupby('year', as_index=False)[numeric_cols].sum()
    print(grouped)
    return grouped


def save_yearly_table(df_yearly: pd.DataFrame, ticker: str):
    print("--- Saving yearly summary table ---")
    df = df_yearly.copy()
    for col in ['total_revenue', 'cost_of_revenue', 'research_and_development', 'selling_general_admin']:
        df[col] = df[col].map(lambda x: f"${x/1e6:,.0f}M")

    df = df.rename(columns={
        'year': 'Year',
        'total_revenue': 'Revenue',
        'cost_of_revenue': 'Cost of Revenue',
        'research_and_development': 'R&D',
        'selling_general_admin': 'SG&A'
    })

    path = os.path.join(OUTPUT_DIR, f"{ticker}_yearly_financials.csv")
    df.to_csv(path, index=False)
    print(df.to_string(index=False))
    print(f"✅ Saved → {path}")


def plot_absolute_vs_revenue(df_yearly: pd.DataFrame, ticker: str):
    print("--- Plotting absolute revenue vs. expenses ---")
    yrs  = df_yearly['year'].astype(str)
    cost = df_yearly['cost_of_revenue']
    rnd  = df_yearly['research_and_development']
    sga  = df_yearly['selling_general_admin']
    rev  = df_yearly['total_revenue']

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(yrs, cost, label='Cost of Revenue', color='dimgray')
    ax.bar(yrs, rnd, label='R&D', bottom=cost, color='blue')
    ax.bar(yrs, sga, label='SG&A', bottom=cost + rnd, color='mediumpurple')
    ax.plot(yrs, rev, label='Revenue', color='darkgreen', marker='o')

    ax.set_ylabel("Amount ($)")
    ax.set_title("Revenue vs Expenses")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${int(x/1e6)}M"))
    ax.legend(loc="upper right")
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, f"{ticker}_rev_expense_chart.png")
    plt.savefig(path, dpi=300)
    plt.close(fig)
    print(f"✅ Saved → {path}")


def plot_expense_percent(df_yearly: pd.DataFrame, ticker: str):
    print("--- Plotting expenses as % of revenue ---")
    yrs = df_yearly['year'].astype(str)
    rev = df_yearly['total_revenue']
    cats = ['cost_of_revenue', 'research_and_development', 'selling_general_admin']
    lbls = ['Cost of Revenue', 'R&D', 'SG&A']
    cols = ['dimgray', 'blue', 'mediumpurple']
    pct = df_yearly[cats].div(rev, axis=0) * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = [0] * len(df_yearly)
    for c, l, clr in zip(cats, lbls, cols):
        ax.bar(yrs, pct[c], bottom=bottom, label=l, color=clr)
        bottom = (pd.Series(bottom) + pct[c]).tolist()

    ax.set_ylabel("Percent of Revenue")
    ax.set_title("Expenses as % of Revenue")
    ax.legend(loc="upper right")
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, f"{ticker}_expense_percent_chart.png")
    plt.savefig(path, dpi=300)
    plt.close(fig)
    print(f"✅ Saved → {path}")


def save_yoy_table(df_yearly: pd.DataFrame, ticker: str):
    print("--- Calculating YoY % change ---")
    df = df_yearly[['year', 'cost_of_revenue', 'research_and_development', 'selling_general_admin']].copy()
    for col in ['cost_of_revenue', 'research_and_development', 'selling_general_admin']:
        df[col] = (df[col].pct_change() * 100).round(2)

    df = df.rename(columns={
        'year': 'Year',
        'cost_of_revenue': 'Cost of Revenue %Δ',
        'research_and_development': 'R&D %Δ',
        'selling_general_admin': 'SG&A %Δ'
    })

    path = os.path.join(OUTPUT_DIR, f"{ticker}_yoy_expense_change.csv")
    df.to_csv(path, index=False)
    print(df.to_string(index=False))
    print(f"✅ Saved → {path}")
    
def generate_expense_reports(ticker: str):
    print(f"\n=== Generating expense reports for {ticker} ===")
    try:
        df = fetch_and_store_income_statement(ticker)

        # ✅ Print the full raw income statement DataFrame
        print(f"\n📄 Full income statement for {ticker}:\n")
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 150)
        print(df.to_string())

        df_yearly = load_yearly_data(ticker)
        save_yearly_table(df_yearly, ticker)
        plot_absolute_vs_revenue(df_yearly, ticker)
        plot_expense_percent(df_yearly, ticker)
        save_yoy_table(df_yearly, ticker)
        print(f"\n=== Done for {ticker} ===\n")
    except Exception as e:
        import traceback
        print(f"\n❌ Error generating expense reports for {ticker}:\n")
        traceback.print_exc()


if __name__ == "__main__":
    generate_expense_reports("AAPL")

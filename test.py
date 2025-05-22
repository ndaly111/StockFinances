import yfinance as yf
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import os

# Constants
DB_PATH = "Stock Data.db"
CHARTS_DIR = "charts"
os.makedirs(CHARTS_DIR, exist_ok=True)

def fetch_and_store_income_statement(ticker):
    print(f"Fetching income statement for {ticker}")
    stock = yf.Ticker(ticker)
    df = stock.income_stmt
    if df is None or df.empty:
        print(f"No income statement data found for {ticker}")
        return None

    df = df.T  # Transpose so dates are rows
    df.reset_index(inplace=True)
    df.rename(columns={"index": "Date"}, inplace=True)
    df["Ticker"] = ticker

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS IncomeStatementData (
            Ticker TEXT,
            Date TEXT,
            TotalRevenue REAL,
            CostOfRevenue REAL,
            SellingGeneralAdministrative REAL,
            ResearchDevelopment REAL,
            SellingAndMarketing REAL
        );
    """)

    for _, row in df.iterrows():
        cursor.execute("""
            INSERT OR REPLACE INTO IncomeStatementData (
                Ticker, Date, TotalRevenue, CostOfRevenue, 
                SellingGeneralAdministrative, ResearchDevelopment, 
                SellingAndMarketing
            ) VALUES (?, ?, ?, ?, ?, ?, ?);
        """, (
            row["Ticker"],
            row["Date"],
            row.get("TotalRevenue"),
            row.get("CostOfRevenue"),
            row.get("SellingGeneralAdministrative"),
            row.get("ResearchDevelopment"),
            row.get("SellingAndMarketing")
        ))
    conn.commit()
    conn.close()
    print(f"Stored income statement data for {ticker}")
    return df

def plot_income_statement_chart(df, ticker):
    print(f"Generating chart for {ticker}")
    df['Year'] = pd.to_datetime(df['Date']).dt.year

    categories = [
        ("CostOfRevenue", "#3498db"),
        ("SellingGeneralAdministrative", "#ff7f50"),
        ("ResearchDevelopment", "#77dd77"),
        ("SellingAndMarketing", "#f4c542")
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    bottom = pd.Series([0] * len(df), index=df.index)
    for col, color in categories:
        if col in df.columns:
            values = df[col].fillna(0)
            ax.bar(df["Year"], values, bottom=bottom, label=col, color=color)
            bottom += values

    ax.set_title(f"{ticker} - Operational Expenses Breakdown")
    ax.set_xlabel("Year")
    ax.set_ylabel("Amount ($)")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    filename = os.path.join(CHARTS_DIR, f"{ticker}_income_statement_chart.png")
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
    print(f"Chart saved to {filename}")

def run_income_stmt_pipeline(ticker):
    df = fetch_and_store_income_statement(ticker)
    if df is not None:
        plot_income_statement_chart(df, ticker)

# Example standalone run
if __name__ == "__main__":
    run_income_stmt_pipeline("AAPL")

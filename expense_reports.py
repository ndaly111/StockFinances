import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from datetime import datetime, timedelta
import yfinance as yf

DB_PATH = "Stock Data.db"
OUTPUT_DIR = "charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_value(val):
    if pd.isna(val):
        return None
    if isinstance(val, (pd.Timestamp, datetime)):
        return val.isoformat()
    return val

def extract_expenses(row: pd.Series):
    def first(cols):
        for c in cols:
            if c in row and pd.notna(row[c]):
                return row[c]
        return 0.0

    cost_rev = first(["Cost Of Revenue", "Reconciled Cost Of Revenue"])
    rnd = first(["Research And Development", "Research Development"])
    mkt = row.get("Selling And Marketing Expense", np.nan)
    adm = row.get("General And Administrative Expense", np.nan)
    sga_comb = first([
        "Selling General & Administrative",
        "Selling General And Administration",
        "Selling General Administrative"
    ])

    mkt = 0.0 if pd.isna(mkt) else mkt
    adm = 0.0 if pd.isna(adm) else adm
    if mkt > 0 and adm > 0:
        sga_comb = 0.0  # Prefer separate values

    return cost_rev, rnd, mkt, adm, sga_comb

def store_annual_data(ticker: str):
    print(f"\n--- Fetching ANNUAL financials for {ticker} ---")
    df = yf.Ticker(ticker).financials.transpose()

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS IncomeStatement (
            ticker TEXT,
            period_ending TEXT,
            total_revenue REAL,
            cost_of_revenue REAL,
            research_and_development REAL,
            selling_and_marketing REAL,
            general_and_admin REAL,
            sga_combined REAL,
            PRIMARY KEY (ticker, period_ending)
        );
    """)

    for idx, row in df.iterrows():
        pe = idx.to_pydatetime() if isinstance(idx, pd.Timestamp) else idx
        tot_rev, cost, rnd, mkt, adm, sga = (row.get("Total Revenue", 0.0), *extract_expenses(row))
        cur.execute("""
            INSERT OR REPLACE INTO IncomeStatement
            (ticker, period_ending, total_revenue, cost_of_revenue,
             research_and_development, selling_and_marketing,
             general_and_admin, sga_combined)
            VALUES (?,?,?,?,?,?,?,?)
        """, (ticker, clean_value(pe), clean_value(tot_rev),
              clean_value(cost), clean_value(rnd),
              clean_value(mkt), clean_value(adm),
              clean_value(sga)))
    conn.commit()
    conn.close()
    print("✅ Annual data stored.")

def store_quarterly_data(ticker: str):
    print(f"\n--- Fetching QUARTERLY financials for {ticker} ---")
    df = yf.Ticker(ticker).quarterly_financials.transpose()

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS QuarterlyIncomeStatement (
            ticker TEXT,
            period_ending TEXT,
            total_revenue REAL,
            cost_of_revenue REAL,
            research_and_development REAL,
            selling_and_marketing REAL,
            general_and_admin REAL,
            sga_combined REAL,
            PRIMARY KEY (ticker, period_ending)
        );
    """)

    for idx, row in df.iterrows():
        pe = idx.to_pydatetime() if isinstance(idx, pd.Timestamp) else idx
        tot_rev, cost, rnd, mkt, adm, sga = (row.get("Total Revenue", 0.0), *extract_expenses(row))
        cur.execute("""
            INSERT OR REPLACE INTO QuarterlyIncomeStatement
            (ticker, period_ending, total_revenue, cost_of_revenue,
             research_and_development, selling_and_marketing,
             general_and_admin, sga_combined)
            VALUES (?,?,?,?,?,?,?,?)
        """, (ticker, clean_value(pe), clean_value(tot_rev),
              clean_value(cost), clean_value(rnd),
              clean_value(mkt), clean_value(adm),
              clean_value(sga)))
    conn.commit()
    conn.close()
    print("✅ Quarterly data stored.")

def fetch_ttm_data(ticker: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT * FROM QuarterlyIncomeStatement
        WHERE ticker = ?
        ORDER BY period_ending DESC
    """, conn, params=(ticker,))
    conn.close()

    df["period_ending"] = pd.to_datetime(df["period_ending"])
    df = df.sort_values("period_ending", ascending=False)
    recent_df = df[df["period_ending"] > (datetime.today() - timedelta(days=150))]

    if len(recent_df) < 4:
        print(f"⛔ Not enough recent data for TTM: {len(recent_df)} quarters")
        return pd.DataFrame()

    recent_df = recent_df.head(4).sort_values("period_ending")
    expected_quarters = pd.date_range(end=recent_df["period_ending"].max(), periods=4, freq="Q")
    actual_quarters = list(recent_df["period_ending"].dt.to_period("Q"))

    if list(expected_quarters.to_period("Q")) != actual_quarters:
        print("⛔ Quarters are not consecutive — TTM invalid")
        return pd.DataFrame()

    agg = recent_df.drop(columns=["ticker", "period_ending"]).sum().to_frame().T
    agg.insert(0, "year", "TTM")
    return agg

def format_short(x, decimal=1):
    if pd.isna(x): return "$0"
    abs_x = abs(x)
    if abs_x >= 1e12: return f"${x / 1e12:.{decimal}f} T"
    elif abs_x >= 1e9: return f"${x / 1e9:.{decimal}f} B"
    elif abs_x >= 1e6: return f"${x / 1e6:.{decimal}f} M"
    elif abs_x >= 1e3: return f"${x / 1e3:.{decimal}f} K"
    else: return f"${x:.{decimal}f}"

def load_yearly_data(ticker: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT period_ending, total_revenue, cost_of_revenue,
               research_and_development, selling_and_marketing,
               general_and_admin, sga_combined
        FROM IncomeStatement
        WHERE ticker = ?
    """, conn, params=(ticker,))
    conn.close()

    if df.empty:
        print(f"⛔ No data found in IncomeStatement for {ticker}")
        return pd.DataFrame()

    df["period_ending"] = pd.to_datetime(df["period_ending"], errors="coerce")
    df = df.dropna(subset=["period_ending"])
    df["year"] = df["period_ending"].dt.year

    numeric_cols = ["year", "total_revenue", "cost_of_revenue",
                    "research_and_development", "selling_and_marketing",
                    "general_and_admin", "sga_combined"]
    df = df[numeric_cols]

    if df[numeric_cols[1:]].replace(0, np.nan).dropna(how="all").empty:
        print(f"⛔ All numeric data for {ticker} is zero or NaN")
        return pd.DataFrame()

    return df.groupby("year", as_index=False).sum()

def plot_chart(df: pd.DataFrame, ticker: str):
    print("--- Plotting Revenue vs Expenses ---")
    yrs = df["year"].astype(str)
    n = len(df)
    positions = np.arange(n)
    width = 0.35
    exp_pos = positions - width / 2
    rev_pos = positions + width / 2

    rev = df["total_revenue"]
    cost = df["cost_of_revenue"]
    rnd = df["research_and_development"]
    mkt = df["selling_and_marketing"]
    adm = df["general_and_admin"]
    sga = df["sga_combined"]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bottom = np.zeros(n)

    def add_stack(values, label, color):
        if np.sum(values) == 0:
            return bottom
        bars = ax.bar(exp_pos, values, width, bottom=bottom, label=label, color=color)
        for i, bar in enumerate(bars):
            val = bar.get_height()
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bottom[i] + val/2,
                        format_short(val, 0),
                        ha='center', va='center', fontsize=7, color='white')
        return bottom + values

    bottom = add_stack(cost, "Cost of Revenue", "dimgray")
    bottom = add_stack(rnd, "R&D", "blue")
    if np.sum(mkt) > 0 or np.sum(adm) > 0:
        bottom = add_stack(mkt, "Sales and Marketing", "mediumpurple")
        bottom = add_stack(adm, "General and Administrative", "pink")
    else:
        bottom = add_stack(sga, "SG&A", "mediumpurple")

    bars = ax.bar(rev_pos, rev, width, label="Revenue", color="green")
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height, format_short(height, 0),
                ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xticks(positions)
    ax.set_xticklabels(yrs)
    ax.set_ylabel("Amount")
    ax.set_title(f"Revenue vs Expenses — {ticker}")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: format_short(x, 0)))

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"{ticker}_rev_expense_chart.png")
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"✅ Saved chart → {path}")

def save_expense_table_html(df: pd.DataFrame, ticker: str):
    print("--- Saving expense table HTML ---")
    df = df.rename(columns={
        "year": "Year", "total_revenue": "Revenue",
        "cost_of_revenue": "Cost of Revenue",
        "research_and_development": "R&D",
        "selling_and_marketing": "Sales and Marketing",
        "general_and_admin": "General and Administrative",
        "sga_combined": "SG&A"
    }).copy()

    for col in df.columns[1:]:
        df[col] = df[col].apply(lambda x: format_short(x, 1))

    html = df.to_html(index=False, border=0, classes="table table-striped")
    path = os.path.join(OUTPUT_DIR, f"{ticker}_expense_table.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ Saved table → {path}")

def generate_expense_reports(ticker: str):
    print(f"\n=== Generating expense reports for {ticker} ===")

    store_annual_data(ticker)
    store_quarterly_data(ticker)

    yearly = load_yearly_data(ticker)
    if yearly.empty:
        print(f"⚠️ No annual data found in IncomeStatement for {ticker}")

    ttm = fetch_ttm_data(ticker)
    if ttm.empty:
        print(f"⚠️ No valid TTM data from QuarterlyIncomeStatement for {ticker}")

    if yearly.empty and ttm.empty:
        print(f"⛔ No yearly or TTM data available for {ticker} — skipping report")
        return

    combined = pd.concat([df for df in [yearly, ttm] if not df.empty], ignore_index=True)
    save_expense_table_html(combined, ticker)
    plot_chart(combined, ticker)

    print(f"=== Done for {ticker} ===\n")

if __name__ == "__main__":
    generate_expense_reports("AAPL")

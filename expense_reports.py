import os
import sqlite3
import pandas as pd
import numpy as np
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
        sga_comb = 0.0  # Prefer separate if both are present

    return cost_rev, rnd, mkt, adm, sga_comb

def fetch_and_store_quarterly(ticker: str):
    print(f"\n--- Fetching QUARTERLY financials for {ticker} ---")
    tkr = yf.Ticker(ticker)
    df = tkr.quarterly_financials.transpose()
    print("Fetched columns:", list(df.columns))

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS IncomeStatement_Quarterly (
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

    cur.execute("DELETE FROM IncomeStatement_Quarterly WHERE ticker = ?", (ticker,))

    for idx, row in df.iterrows():
        pe = idx.to_pydatetime() if isinstance(idx, pd.Timestamp) else idx
        tot_rev, cost, rnd, mkt, adm, sga = (
            row.get("Total Revenue", 0.0),
            *extract_expenses(row)
        )
        cur.execute("""
            INSERT OR REPLACE INTO IncomeStatement_Quarterly
              (ticker, period_ending, total_revenue, cost_of_revenue,
               research_and_development, selling_and_marketing,
               general_and_admin, sga_combined)
            VALUES (?,?,?,?,?,?,?,?)
        """, (
            ticker, clean_value(pe), clean_value(tot_rev),
            clean_value(cost), clean_value(rnd),
            clean_value(mkt), clean_value(adm),
            clean_value(sga)
        ))
    conn.commit()
    conn.close()
    print("✅ Quarterly income statement stored in DB.")

def load_ttm_data(ticker: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT * FROM IncomeStatement_Quarterly
        WHERE ticker = ?
    """, conn, params=(ticker,))
    conn.close()

    df["period_ending"] = pd.to_datetime(df["period_ending"])
    df = df.sort_values("period_ending", ascending=False)
    recent_cutoff = pd.Timestamp.today() - pd.DateOffset(months=5)
    df = df[df["period_ending"] >= recent_cutoff]

    if len(df) < 4:
        print("❌ Not enough recent quarters for TTM.")
        return pd.DataFrame()

    # Check for consecutive quarters (3-month intervals)
    diffs = df["period_ending"].diff(-1).dt.days.dropna()
    if not all((85 <= d <= 100) for d in diffs[:3]):
        print("❌ Quarters not consecutive.")
        return pd.DataFrame()

    latest_year = df["period_ending"].max().year
    agg = df.head(4).drop(columns=["ticker", "period_ending"]).sum().to_frame().T
    agg["year"] = f"{latest_year} TTM"
    return agg[["year"] + [c for c in agg.columns if c != "year"]]

def format_short(x, decimal=1):
    if pd.isna(x):
        return "$0"
    abs_x = abs(x)
    if abs_x >= 1e12:
        return f"${x / 1e12:.{decimal}f} T"
    elif abs_x >= 1e9:
        return f"${x / 1e9:.{decimal}f} B"
    elif abs_x >= 1e6:
        return f"${x / 1e6:.{decimal}f} M"
    elif abs_x >= 1e3:
        return f"${x / 1e3:.{decimal}f} K"
    else:
        return f"${x:.{decimal}f}"

def load_yearly_data(ticker: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT * FROM IncomeStatement
        WHERE ticker = ?
    """, conn, params=(ticker,))
    conn.close()

    df["period_ending"] = pd.to_datetime(df["period_ending"])
    df["year"] = df["period_ending"].dt.year

    df_annual = df.drop(columns=["ticker", "period_ending"])
    df_annual = df_annual.groupby("year", as_index=False).sum()
    return df_annual

def plot_revenue_vs_expenses(df: pd.DataFrame, ticker: str):
    yrs = df["year"].astype(str)
    n = len(df)
    pos = np.arange(n)
    width = 0.35
    exp_pos = pos - width / 2
    rev_pos = pos + width / 2

    rev = df["total_revenue"]
    cost = df["cost_of_revenue"]
    rnd = df["research_and_development"]
    mkt = df["selling_and_marketing"]
    adm = df["general_and_admin"]
    sga = df["sga_combined"]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bottom = np.zeros(n)

    def stack(values, label, color):
        if np.sum(values) == 0:
            return bottom
        bars = ax.bar(exp_pos, values, width, bottom=bottom, label=label, color=color)
        for i, bar in enumerate(bars):
            if bar.get_height() > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bottom[i] + bar.get_height()/2,
                        format_short(bar.get_height(), 0), ha='center', va='center', fontsize=7, color='white')
        return bottom + values

    bottom = stack(cost, "Cost of Revenue", "dimgray")
    bottom = stack(rnd, "R&D", "blue")
    if np.sum(mkt) > 0 or np.sum(adm) > 0:
        bottom = stack(mkt, "Sales and Marketing", "mediumpurple")
        bottom = stack(adm, "General and Administrative", "pink")
    else:
        bottom = stack(sga, "SG&A", "mediumpurple")

    bars = ax.bar(rev_pos, rev, width, label="Revenue", color="green")
    for i, bar in enumerate(bars):
        if bar.get_height() > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), format_short(bar.get_height(), 0),
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xticks(pos)
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
    fetch_and_store_quarterly(ticker)
    annual_df = load_yearly_data(ticker)
    ttm_df = load_ttm_data(ticker)
    final_df = pd.concat([annual_df, ttm_df], ignore_index=True)
    save_expense_table_html(final_df, ticker)
    plot_revenue_vs_expenses(final_df, ticker)
    print(f"\n=== Done for {ticker} ===\n")

if __name__ == "__main__":
    generate_expense_reports("AAPL")

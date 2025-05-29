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
        sga_comb = 0.0

    return cost_rev, rnd, mkt, adm, sga_comb


def fetch_and_store_income_statement(ticker: str) -> pd.DataFrame:
    print(f"\n--- Fetching ANNUAL financials for {ticker} ---")
    tkr = yf.Ticker(ticker)
    df = tkr.financials.transpose()
    print("Fetched columns:", list(df.columns))

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS IncomeStatement (
            ticker TEXT,
            period_ending TEXT PRIMARY KEY,
            total_revenue REAL,
            cost_of_revenue REAL,
            research_and_development REAL,
            selling_and_marketing REAL,
            general_and_admin REAL,
            sga_combined REAL
        );
    """)

    for idx, row in df.iterrows():
        pe = idx.to_pydatetime() if isinstance(idx, pd.Timestamp) else idx
        tot_rev, cost, rnd, mkt, adm, sga = (
            row.get("Total Revenue", 0.0),
            *extract_expenses(row)
        )
        cur.execute("""
            INSERT OR REPLACE INTO IncomeStatement
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
    print("✅ Income statement stored in DB.")
    return df


def format_short(x, decimal=0):
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
    print("--- Loading & aggregating yearly data ---")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT period_ending, total_revenue, cost_of_revenue,
               research_and_development, selling_and_marketing,
               general_and_admin, sga_combined
        FROM IncomeStatement
        WHERE ticker = ?
    """, conn, params=(ticker,))
    conn.close()

    df["period_ending"] = pd.to_datetime(df["period_ending"])
    df["year"] = df["period_ending"].dt.year

    # Only keep numeric columns + 'year'
    numeric_cols = ["year", "total_revenue", "cost_of_revenue",
                    "research_and_development", "selling_and_marketing",
                    "general_and_admin", "sga_combined"]
    df = df[numeric_cols]

    grouped = df.groupby("year", as_index=False).sum()
    return grouped

def plot_revenue_vs_expenses(df_yearly: pd.DataFrame, ticker: str):
    print("--- Plotting Revenue vs Expenses ---")
    yrs = df_yearly["year"].astype(str)
    cost = df_yearly["cost_of_revenue"]
    rnd = df_yearly["research_and_development"]
    mkt = df_yearly["selling_and_marketing"]
    adm = df_yearly["general_and_admin"]
    sga = df_yearly["sga_combined"]
    rev = df_yearly["total_revenue"]

    n = len(df_yearly)
    positions = np.arange(n)
    width = 0.4
    exp_pos = positions - width / 2
    rev_pos = positions + width / 2

    fig, ax = plt.subplots(figsize=(8, 4))
    bottom = np.zeros(n)

    legend_labels = []
    def stack_bar(series, label, color):
        nonlocal bottom
        if series.sum() > 0:
            ax.bar(exp_pos, series, width, bottom=bottom, label=label, color=color)
            bottom += series
            legend_labels.append(label)

    stack_bar(cost, "Cost of Revenue", "dimgray")
    stack_bar(rnd, "R&D", "blue")
    stack_bar(mkt, "Sales and Marketing", "mediumpurple")
    stack_bar(adm, "General and Administrative", "pink")
    stack_bar(sga, "SG&A", "violet")

    # Add total expense labels
    for i, total in enumerate(bottom):
        ax.text(exp_pos[i], total, format_short(total, 0),
                ha='center', va='bottom', fontsize=8)

    # Revenue bars and labels
    ax.bar(rev_pos, rev, width, label="Revenue", color="green")
    for i, r in enumerate(rev):
        ax.text(rev_pos[i], r, format_short(r, 0),
                ha='center', va='bottom', fontsize=8)

    ax.set_xticks(positions)
    ax.set_xticklabels(yrs)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: format_short(x, 0)))
    ax.set_ylabel("Amount")
    ax.set_title(f"Revenue vs Expenses — {ticker}")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, f"{ticker}_rev_expense_chart.png")
    plt.savefig(path, dpi=300)
    plt.close(fig)
    print(f"✅ Saved chart → {path}")


def save_expense_table_html(df_yearly: pd.DataFrame, ticker: str):
    print("--- Saving expense table HTML ---")
    df = df_yearly.copy()
    df = df.rename(columns={
        "year": "Year", "total_revenue": "Revenue",
        "cost_of_revenue": "Cost of Revenue",
        "research_and_development": "R&D",
        "selling_and_marketing": "Sales and Marketing",
        "general_and_admin": "General and Administrative",
        "sga_combined": "SG&A"
    })

    for col in df.columns[1:]:
        df[col] = df[col].apply(lambda x: format_short(x, 1))

    html = df.to_html(index=False, border=0, classes="table table-striped")
    path = os.path.join(OUTPUT_DIR, f"{ticker}_expense_table.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ Saved table → {path}")


def generate_expense_reports(ticker: str):
    print(f"\n=== Generating expense reports for {ticker} ===")
    fetch_and_store_income_statement(ticker)
    df = load_yearly_data(ticker)
    save_expense_table_html(df, ticker)
    plot_revenue_vs_expenses(df, ticker)
    print(f"\n=== Done for {ticker} ===\n")


if __name__ == "__main__":
    generate_expense_reports("AAPL")

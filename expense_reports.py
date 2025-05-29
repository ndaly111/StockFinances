import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from datetime import datetime
import yfinance as yf

DB_PATH    = "Stock Data.db"
OUTPUT_DIR = "charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def clean_value(val):
    if pd.isna(val):
        return None
    if isinstance(val, (pd.Timestamp, datetime)):
        return val.isoformat()
    return val


def extract_expenses(row: pd.Series):
    """Return (cost_rev, rnd, mkt, adm, sga_combined)."""
    def first(cols):
        for c in cols:
            if c in row and pd.notna(row[c]):
                return row[c]
        return 0.0

    cost_rev = first(["Cost Of Revenue", "Reconciled Cost Of Revenue"])
    rnd      = first(["Research And Development", "Research Development"])
    mkt      = row.get("Selling And Marketing Expense", np.nan)
    adm      = row.get("General And Administrative Expense", np.nan)
    sga_comb = first([
        "Selling General & Administrative",
        "Selling General And Administration",
        "Selling General Administrative"
    ])

    mkt = 0.0 if pd.isna(mkt) else mkt
    adm = 0.0 if pd.isna(adm) else adm
    # if both granular pieces exist, ignore combined
    if mkt > 0 and adm > 0:
        sga_comb = 0.0

    return cost_rev, rnd, mkt, adm, sga_comb


def fetch_and_store_income_statement(ticker: str) -> pd.DataFrame:
    print(f"\n--- Fetching ANNUAL financials for {ticker} ---")
    tkr = yf.Ticker(ticker)
    df  = tkr.financials.transpose()
    print("Fetched columns:", list(df.columns))

    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
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
    """Format x into K/M/B/T with `decimal` decimals."""
    if pd.isna(x) or x == 0:
        return "$0"
    abs_x = abs(x)
    for factor, suffix in ((1e12,"T"), (1e9,"B"), (1e6,"M"), (1e3,"K")):
        if abs_x >= factor:
            return f"${x/factor:.{decimal}f}{suffix}"
    return f"${x:.{decimal}f}"


def load_yearly_data(ticker: str) -> pd.DataFrame:
    print("--- Loading & aggregating yearly data ---")
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql_query("""
        SELECT period_ending, total_revenue, cost_of_revenue,
               research_and_development, selling_and_marketing,
               general_and_admin, sga_combined
        FROM IncomeStatement
        WHERE ticker = ?
    """, conn, params=(ticker,))
    conn.close()

    df["period_ending"] = pd.to_datetime(df["period_ending"])
    df["year"]          = df["period_ending"].dt.year

    cols = ["year","total_revenue","cost_of_revenue",
            "research_and_development","selling_and_marketing",
            "general_and_admin","sga_combined"]
    df = df[cols]

    return df.groupby("year", as_index=False).sum()


def plot_revenue_vs_expenses(df_yearly: pd.DataFrame, ticker: str):
    print("--- Plotting Revenue vs Expenses ---")
    yrs       = df_yearly["year"].astype(str)
    n         = len(df_yearly)
    positions = np.arange(n)
    width     = 0.4
    exp_pos   = positions - width/2
    rev_pos   = positions + width/2

    rev = df_yearly["total_revenue"]
    cost = df_yearly["cost_of_revenue"]
    rnd  = df_yearly["research_and_development"]
    mkt  = df_yearly["selling_and_marketing"]
    adm  = df_yearly["general_and_admin"]
    sga  = df_yearly["sga_combined"]

    fig, ax = plt.subplots(figsize=(9, 4))
    bottom = np.zeros(n)
    handles = []

    def add_stack(vals, label, color):
        nonlocal bottom
        if vals.sum() == 0:
            return
        bars = ax.bar(exp_pos, vals, width, bottom=bottom, label=label, color=color)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(
                    bar.get_x()+bar.get_width()/2,
                    bottom[bar.get_x()==bar.get_x()]+val/2,
                    format_short(val,0),
                    ha='center', va='center', color='white', fontsize=7
                )
        bottom += vals
        handles.append(label)

    # stack in order
    add_stack(cost, "Cost of Revenue", "dimgray")
    add_stack(rnd,  "R&D",             "blue")
    # if granular SM+GA exist use them, else fallback to SG&A
    if mkt.sum()>0 or adm.sum()>0:
        add_stack(mkt, "Sales and Marketing", "mediumpurple")
        add_stack(adm, "General and Administrative", "pink")
    else:
        add_stack(sga, "SG&A", "mediumslateblue")

    # revenue bars & labels
    rev_bars = ax.bar(rev_pos, rev, width, label="Revenue", color="green")
    for bar, val in zip(rev_bars, rev):
        ax.text(
            bar.get_x()+bar.get_width()/2,
            val + rev.max()*0.01,
            format_short(val,0),
            ha='center', va='bottom', fontsize=9, fontweight='bold', color='black'
        )

    # axes & legend
    ax.set_xticks(positions)
    ax.set_xticklabels(yrs)
    ax.set_ylabel("Amount")
    ax.set_title(f"Revenue vs Expenses — {ticker}")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x,_: format_short(x,0)))
    ax.legend(handles=handles+["Revenue"], loc="upper left", bbox_to_anchor=(1.01,1.0))
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"{ticker}_rev_expense_chart.png")
    plt.savefig(out, dpi=300)
    plt.close(fig)
    print(f"✅ Saved chart → {out}")


def save_expense_table_html(df_yearly: pd.DataFrame, ticker: str):
    print("--- Saving expense table HTML ---")
    df = df_yearly.copy().rename(columns={
        "year": "Year", "total_revenue": "Revenue",
        "cost_of_revenue": "Cost of Revenue",
        "research_and_development": "R&D",
        "selling_and_marketing": "Sales and Marketing",
        "general_and_admin": "General and Administrative",
        "sga_combined": "SG&A"
    })

    # one decimal in table
    for col in df.columns[1:]:
        df[col] = df[col].apply(lambda x: format_short(x,1))

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

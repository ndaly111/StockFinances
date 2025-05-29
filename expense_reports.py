# expense_reports.py

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


# ────────────────────────────────────────────────
# Utility
# ────────────────────────────────────────────────
def clean_value(val):
    if pd.isna(val):
        return None
    if isinstance(val, (pd.Timestamp, datetime)):
        return val.isoformat()
    return val


# ────────────────────────────────────────────────
# Flexible field extractor
# ────────────────────────────────────────────────
def extract_expenses(row: pd.Series):
    """Return (cost_of_revenue, rnd, marketing, admin, sga_combined)."""
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

    # normalize nan → 0
    mkt = 0.0 if pd.isna(mkt) else mkt
    adm = 0.0 if pd.isna(adm) else adm

    # if they gave us both marketing AND admin, ignore the combined fallback
    if mkt > 0 and adm > 0:
        sga_comb = 0.0

    return cost_rev, rnd, mkt, adm, sga_comb


# ────────────────────────────────────────────────
# Fetch & store (annual)
# ────────────────────────────────────────────────
def fetch_and_store_income_statement(ticker: str) -> pd.DataFrame:
    print(f"\n--- Fetching ANNUAL financials for {ticker} ---")
    tkr = yf.Ticker(ticker)
    df  = tkr.financials.transpose()  # annual, not quarterly
    print("Fetched columns:", list(df.columns))

    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS IncomeStatement (
            ticker                   TEXT,
            period_ending            TEXT PRIMARY KEY,
            total_revenue            REAL,
            cost_of_revenue          REAL,
            research_and_development REAL,
            selling_and_marketing    REAL,
            general_and_admin        REAL,
            sga_combined             REAL
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


# ────────────────────────────────────────────────
# Load & aggregate
# ────────────────────────────────────────────────
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

    grouped = df.groupby("year", as_index=False)[[
        "total_revenue", "cost_of_revenue",
        "research_and_development", "selling_and_marketing",
        "general_and_admin", "sga_combined"
    ]].sum()
    print(grouped)
    return grouped


# ────────────────────────────────────────────────
# Expense table → HTML fragment
# ────────────────────────────────────────────────
def save_expense_table_html(df_yearly: pd.DataFrame, ticker: str):
    print("--- Saving expense table HTML ---")
    tbl = df_yearly.copy().rename(columns={
        "year": "Year",
        "total_revenue": "Revenue",
        "cost_of_revenue": "Cost of Revenue",
        "research_and_development": "Research and Development",
        "selling_and_marketing": "Sales and Marketing",
        "general_and_admin": "General and Administrative",
        "sga_combined": "SG&A"
    })

    # decide columns: if they broke out Mkt/Admin use those, else use SG&A
    if tbl[["Sales and Marketing","General and Administrative"]].sum().sum() > 0:
        out_cols = ["Year","Revenue","Cost of Revenue",
                    "Research and Development",
                    "Sales and Marketing","General and Administrative"]
    else:
        out_cols = ["Year","Revenue","Cost of Revenue",
                    "Research and Development","SG&A"]

    # format $ in millions
    for c in out_cols[1:]:
        tbl[c] = tbl[c].map(lambda x: f"${x/1e6:,.0f}M")

    html = tbl[out_cols].to_html(
        index=False, border=0, classes="table table-striped"
    )
    path = os.path.join(OUTPUT_DIR, f"{ticker}_expense_table.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ Saved expense table → {path}")


# ────────────────────────────────────────────────
# Plot: Revenue vs Expenses (side-by-side, stacked)
# ────────────────────────────────────────────────
def plot_revenue_vs_expenses(df_yearly: pd.DataFrame, ticker: str):
    print("--- Plotting Revenue vs Expenses ---")
    yrs   = df_yearly["year"].astype(str)
    cost  = df_yearly["cost_of_revenue"]
    rnd   = df_yearly["research_and_development"]
    mkt   = df_yearly["selling_and_marketing"]
    adm   = df_yearly["general_and_admin"]
    sga   = df_yearly["sga_combined"]
    rev   = df_yearly["total_revenue"]

    n         = len(df_yearly)
    positions = np.arange(n)
    width     = 0.4
    exp_pos   = positions - width / 2
    rev_pos   = positions + width / 2

    fig, ax = plt.subplots(figsize=(10, 4))

    # Custom formatter
    def format_money(val):
        abs_val = abs(val)
        if abs_val >= 1e12:
            return f"${val / 1e12:.1f} T"
        elif abs_val >= 1e9:
            return f"${val / 1e9:.1f} B"
        elif abs_val >= 1e6:
            return f"${val / 1e6:.1f} M"
        elif abs_val >= 1e3:
            return f"${val / 1e3:.1f} K"
        elif abs_val > 0:
            return f"${val:,.0f}"
        else:
            return "$0"

    # Expense bar stack
    expense_components = [
        (cost, "Cost of Revenue", "dimgray"),
        (rnd,  "R&D",             "blue"),
        (mkt,  "Sales and Marketing", "mediumpurple"),
        (adm,  "General and Administrative", "pink"),
        (sga,  "SG&A", "mediumpurple")
    ]

    bottom = np.zeros(n)
    for vals, label, color in expense_components:
        if np.nansum(vals) == 0:
            continue  # skip label if all-zero
        ax.bar(exp_pos, vals, width, bottom=bottom, label=label, color=color)
        bottom += vals

    # Revenue bars
    if np.nansum(rev) > 0:
        ax.bar(rev_pos, rev, width, label="Revenue", color="green")

    # Axis formatting
    ax.set_xticks(positions)
    ax.set_xticklabels(yrs)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: format_money(x)))
    ax.set_ylabel("Amount")
    ax.set_title(f"Revenue vs Expenses — {ticker}")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    plt.tight_layout()

    # Table prep
    tbl = df_yearly.copy()
    tbl["Revenue"] = rev
    tbl["Cost of Revenue"] = cost
    tbl["R&D"] = rnd
    tbl["SG&A"] = mkt.fillna(0) + adm.fillna(0) + sga.fillna(0)
    tbl["Year"] = yrs

    table_data = tbl[["Year", "Revenue", "Cost of Revenue", "R&D", "SG&A"]].copy()
    for col in ["Revenue", "Cost of Revenue", "R&D", "SG&A"]:
        table_data[col] = table_data[col].apply(format_money)

    # Add table below
    cell_data = table_data.values
    col_labels = table_data.columns.tolist()
    plt.table(cellText=cell_data,
              colLabels=col_labels,
              cellLoc='center',
              loc='bottom',
              bbox=[0.0, -0.48, 1, 0.33])
    plt.subplots_adjust(bottom=0.35)

    # Save
    path = os.path.join(OUTPUT_DIR, f"{ticker}_rev_expense_chart.png")
    plt.savefig(path, dpi=300)
    plt.close(fig)
    print(f"✅ Saved chart → {path}")

# ────────────────────────────────────────────────
# YoY % Δ for expense categories
# ────────────────────────────────────────────────
def save_yoy_table(df_yearly: pd.DataFrame, ticker: str):
    print("--- Calculating YoY % change ---")
    df = df_yearly.copy().set_index("year")
    df["Cost of Revenue %Δ"] = df["cost_of_revenue"].pct_change() * 100
    df["R&D %Δ"]             = df["research_and_development"].pct_change() * 100

    # combine all expense categories for a total‐expense YoY
    df["Expenses %Δ"] = (
        df[["selling_and_marketing","general_and_admin","sga_combined"]]
        .sum(axis=1).pct_change() * 100
    )

    out = df[["Cost of Revenue %Δ","R&D %Δ","Expenses %Δ"]].round(2)
    out = out.rename_axis("Year").reset_index()

    path = os.path.join(OUTPUT_DIR, f"{ticker}_yoy_expense_change.csv")
    out.to_csv(path, index=False)
    print(out.to_string(index=False))
    print(f"✅ Saved YoY → {path}")


# ────────────────────────────────────────────────
# Orchestrator
# ────────────────────────────────────────────────
def generate_expense_reports(ticker: str):
    print(f"\n=== Generating expense reports for {ticker} ===")
    df_raw    = fetch_and_store_income_statement(ticker)
    pd.set_option("display.max_columns", None, "display.width", 160)
    print(f"\nRaw income statement:\n{df_raw.to_string()}\n")

    df_yearly = load_yearly_data(ticker)
    save_expense_table_html(df_yearly, ticker)
    plot_revenue_vs_expenses(df_yearly, ticker)
    save_yoy_table(df_yearly, ticker)

    print(f"\n=== Done for {ticker} ===\n")


if __name__ == "__main__":
    generate_expense_reports("AAPL")

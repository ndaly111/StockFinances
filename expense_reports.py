# expense_reports.py
import os
import sqlite3
import pandas as pd
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
    """Return tuple:
       (cost_of_revenue, research_dev, marketing, admin, sga_combined)
       Defaults to 0.0 when a field is absent / NaN.
    """
    def first(fields):
        for f in fields:
            if f in row and pd.notna(row[f]):
                return row[f]
        return 0.0

    # Cost of revenue
    cost_rev = first(["Cost Of Revenue", "Reconciled Cost Of Revenue"])

    # Research & development
    rnd = first(["Research And Development",
                 "Research Development", "R&D"])

    # Separate marketing / admin
    marketing = row.get("Selling And Marketing Expense")
    admin     = row.get("General And Administrative Expense")

    # Combined SG&A fallback
    sga_comb  = first(["Selling General And Administration",
                       "Selling General Administrative"])

    # Resolve overlap
    if pd.notna(marketing) and pd.notna(admin):
        sga_comb = 0.0     # we already have granular pieces
    else:
        marketing = 0.0 if pd.isna(marketing) else marketing
        admin     = 0.0 if pd.isna(admin)     else admin

    return cost_rev, rnd, marketing, admin, sga_comb


# ────────────────────────────────────────────────
# Fetch + store
# ────────────────────────────────────────────────
def fetch_and_store_income_statement(ticker: str) -> pd.DataFrame:
    print(f"\n--- Fetching financials for {ticker} ---")
    yf_tkr = yf.Ticker(ticker)
    df = yf_tkr.quarterly_financials.transpose()
    print("Fetched columns:", list(df.columns))

    # open DB + ensure table
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

    # insert / update rows
    for idx, row in df.iterrows():
        pe = idx.to_pydatetime() if isinstance(idx, pd.Timestamp) else idx
        tot_rev   = row.get("Total Revenue", 0.0)
        cost_rev, rnd, mkt, adm, sga_comb = extract_expenses(row)

        cur.execute("""
            INSERT OR REPLACE INTO IncomeStatement
              (ticker, period_ending, total_revenue, cost_of_revenue,
               research_and_development, selling_and_marketing,
               general_and_admin, sga_combined)
            VALUES (?,?,?,?,?,?,?,?)
        """, (
            ticker, clean_value(pe), clean_value(tot_rev), clean_value(cost_rev),
            clean_value(rnd), clean_value(mkt), clean_value(adm),
            clean_value(sga_comb)
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
    df = pd.read_sql_query("""
        SELECT period_ending, total_revenue, cost_of_revenue,
               research_and_development, selling_and_marketing,
               general_and_admin, sga_combined
        FROM IncomeStatement
        WHERE ticker = ?
    """, conn, params=(ticker,))
    conn.close()

    df["period_ending"] = pd.to_datetime(df["period_ending"])
    df["year"]          = df["period_ending"].dt.year

    # compute SG&A total (marketing + admin + combined)
    df["sga_total"] = (
        df["selling_and_marketing"].fillna(0) +
        df["general_and_admin"].fillna(0)     +
        df["sga_combined"].fillna(0)
    )

    grouped = df.groupby("year", as_index=False)[[
        "total_revenue", "cost_of_revenue",
        "research_and_development", "sga_total"
    ]].sum()

    print(grouped)
    return grouped


# ────────────────────────────────────────────────
# Output helpers
# ────────────────────────────────────────────────
def save_yearly_table(df_yearly, ticker):
    print("--- Saving yearly summary table ---")
    df = df_yearly.copy()
    for col in ["total_revenue", "cost_of_revenue",
                "research_and_development", "sga_total"]:
        df[col] = df[col].map(lambda x: f"${x/1e6:,.0f}M")

    df = df.rename(columns={
        "year": "Year", "total_revenue": "Revenue",
        "cost_of_revenue": "Cost of Revenue",
        "research_and_development": "R&D",
        "sga_total": "SG&A"
    })
    path = os.path.join(OUTPUT_DIR, f"{ticker}_yearly_financials.csv")
    df.to_csv(path, index=False)
    print(df.to_string(index=False))
    print("✅ Saved →", path)


def plot_absolute_vs_revenue(df_yearly, ticker):
    print("--- Plotting absolute revenue vs. expenses ---")
    yrs   = df_yearly["year"].astype(str)
    cost  = df_yearly["cost_of_revenue"]
    rnd   = df_yearly["research_and_development"]
    sga   = df_yearly["sga_total"]
    rev   = df_yearly["total_revenue"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(yrs, cost, label="Cost of Revenue", color="dimgray")
    ax.bar(yrs, rnd,  label="R&D",            bottom=cost, color="blue")
    ax.bar(yrs, sga,  label="SG&A",           bottom=cost + rnd, color="mediumpurple")
    ax.plot(yrs, rev, label="Revenue", color="darkgreen", marker="o")

    ax.set_ylabel("Amount ($)")
    ax.set_title("Revenue vs Expenses")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${int(x/1e6)}M"))
    ax.legend(loc="upper right")
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, f"{ticker}_rev_expense_chart.png")
    plt.savefig(out, dpi=300)
    plt.close(fig)
    print("✅ Saved →", out)


def plot_expense_percent(df_yearly, ticker):
    print("--- Plotting expenses as % of revenue ---")
    yrs = df_yearly["year"].astype(str)
    rev = df_yearly["total_revenue"]
    cats = ["cost_of_revenue", "research_and_development", "sga_total"]
    lbls = ["Cost of Revenue", "R&D", "SG&A"]
    cols = ["dimgray", "blue", "mediumpurple"]
    pct  = df_yearly[cats].div(rev, axis=0) * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = [0]*len(df_yearly)
    for c, l, clr in zip(cats, lbls, cols):
        ax.bar(yrs, pct[c], bottom=bottom, label=l, color=clr)
        bottom = (pd.Series(bottom) + pct[c]).tolist()

    ax.set_ylabel("Percent of Revenue")
    ax.set_title("Expenses as % of Revenue")
    ax.legend(loc="upper right")
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, f"{ticker}_expense_percent_chart.png")
    plt.savefig(out, dpi=300)
    plt.close(fig)
    print("✅ Saved →", out)


def save_yoy_table(df_yearly, ticker):
    print("--- Calculating YoY % change ---")
    df = df_yearly[["year", "cost_of_revenue",
                    "research_and_development", "sga_total"]].copy()
    for col in ["cost_of_revenue", "research_and_development", "sga_total"]:
        df[col] = (df[col].pct_change() * 100).round(2)

    df = df.rename(columns={
        "year": "Year",
        "cost_of_revenue": "Cost of Revenue %Δ",
        "research_and_development": "R&D %Δ",
        "sga_total": "SG&A %Δ"
    })
    out = os.path.join(OUTPUT_DIR, f"{ticker}_yoy_expense_change.csv")
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print("✅ Saved →", out)


# ────────────────────────────────────────────────
# Orchestrator
# ────────────────────────────────────────────────
def generate_expense_reports(ticker: str):
    print(f"\n=== Generating expense reports for {ticker} ===")
    try:
        raw = fetch_and_store_income_statement(ticker)

        # Show raw DF for debugging
        pd.set_option("display.max_columns", None, "display.width", 160)
        print(f"\n📄 Raw income statement ({ticker}):\n", raw.to_string(), "\n")

        yearly = load_yearly_data(ticker)
        save_yearly_table(yearly, ticker)
        plot_absolute_vs_revenue(yearly, ticker)
        plot_expense_percent(yearly, ticker)
        save_yoy_table(yearly, ticker)
        print(f"\n=== Done for {ticker} ===\n")

    except Exception:
        import traceback, sys
        print(f"\n❌ Error generating expense reports for {ticker}\n")
        traceback.print_exc(file=sys.stdout)


if __name__ == "__main__":
    generate_expense_reports("AAPL")

# expense_reports.py
# -----------------------------------------------------------------------------
# Builds annual / quarterly operating-expense tables, stores them in SQLite,
# and generates two charts per ticker:
#
#   1) Revenue vs. stacked expenses (absolute $)
#   2) Expenses as % of revenue
#   3) YoY expense-change HTML table
#   4) Absolute expense-dollar HTML table
# -----------------------------------------------------------------------------

import os, sqlite3
import numpy as np
import pandas as pd

from expense_labels import (
    COST_OF_REVENUE, RESEARCH_AND_DEVELOPMENT, SELLING_AND_MARKETING,
    GENERAL_AND_ADMIN, SGA_COMBINED, FACILITIES_DA, PERSONNEL_COSTS,
    INSURANCE_CLAIMS, OTHER_OPERATING,
)

DB_PATH = "Stock Data.db"
OUTPUT_DIR = "charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def _cats():
    return {
        "Cost of Revenue": COST_OF_REVENUE,
        "R&D": RESEARCH_AND_DEVELOPMENT,
        "Selling & Marketing": SELLING_AND_MARKETING,
        "G&A": GENERAL_AND_ADMIN,
        "SG&A Combined": SGA_COMBINED,
        "Facilities / D&A": FACILITIES_DA,
        "Personnel Costs": PERSONNEL_COSTS,
        "Insurance / Claims": INSURANCE_CLAIMS,
        "Other Operating": OTHER_OPERATING,
    }

def _prepare_expense_df(df: pd.DataFrame, cats: dict) -> pd.DataFrame:
    df = df.copy()
    df = df[df["Revenue"].replace(0, np.nan).notna()]  # Drop zero/NaN revenue rows
    df = df.reset_index(drop=True)

    for col in ["Revenue"] + sum(cats.values(), []):
        if col not in df.columns:
            df[col] = np.nan

    cat_data = {}
    for cat_name, fields in cats.items():
        valid = [f for f in fields if f in df.columns]
        if not valid:
            continue
        df[cat_name] = df[valid].sum(axis=1, min_count=1)
        if df[cat_name].replace(0, np.nan).notna().any():
            cat_data[cat_name] = df[cat_name]

    out = pd.DataFrame({"Date": df["Date"], "Revenue": df["Revenue"]})
    for cat in cat_data:
        out[cat] = cat_data[cat]
    return out

def _fetch_annual_and_ttm(ticker: str, conn) -> pd.DataFrame:
    df = pd.read_sql("SELECT * FROM Income_Statements WHERE Ticker = ?", conn, params=(ticker,))
    df = df[df["Period"] == "annual"].sort_values("Date")

    ttm = pd.read_sql("SELECT * FROM Income_Statements_TTM WHERE Ticker = ?", conn, params=(ticker,))
    if not ttm.empty:
        row = ttm.iloc[-1]
        row["Date"] = "TTM " + row["Quarter"]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    return df

def _store_html_table(ticker: str, df: pd.DataFrame):
    path = os.path.join(OUTPUT_DIR, f"{ticker}_yoy_expense_change.html")
    fmt = df.copy()
    for c in fmt.columns[1:]:
        fmt[c] = fmt[c].apply(lambda x: f"${x/1e9:.1f}B" if pd.notna(x) else "-")
    fmt.to_html(path, index=False)

def generate_expense_reports(ticker: str, conn=None):
    if conn is None:
        conn = sqlite3.connect(DB_PATH)

    df = _fetch_annual_and_ttm(ticker, conn)
    if df.empty:
        return

    cats = _cats()
    df_clean = _prepare_expense_df(df, cats)
    _store_html_table(ticker, df_clean)

    conn.close()

if __name__ == "__main__":
    generate_expense_reports("AAPL")

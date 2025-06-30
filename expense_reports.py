# expense_reports.py
# ------------------------------------------------------------------
# Generates charts and tables for operating expenses (Annual + TTM)
# ------------------------------------------------------------------

import os, sqlite3
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.ticker import FuncFormatter
from matplotlib import colors as mcolors

from expense_labels import (
    COST_OF_REVENUE, RESEARCH_AND_DEVELOPMENT, SELLING_AND_MARKETING,
    GENERAL_AND_ADMIN, SGA_COMBINED, FACILITIES_DA, PERSONNEL_COSTS,
    INSURANCE_CLAIMS, OTHER_OPERATING,
)

DB_PATH = "Stock Data.db"
OUTPUT_DIR = "charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def _fmt_short(x: float, d: int = 1) -> str:
    if pd.isna(x): return ""
    for div, suf in [(1e12,"T"),(1e9,"B"),(1e6,"M"),(1e3,"K")]:
        if abs(x) >= div: return f"${x/div:.{d}f}{suf}"
    return f"${x:.{d}f}"

def _all_nan_or_zero(col: pd.Series) -> bool:
    return (col.replace(0, np.nan).notna().sum() == 0)

def clean(v):
    if pd.isna(v): return None
    return v.isoformat() if isinstance(v, (pd.Timestamp, datetime)) else v

def pick_any(row: pd.Series, labels):
    for k in row.index:
        if pd.notna(row[k]) and any(lbl.lower() in k.lower() for lbl in labels):
            return row[k]
    return None

def extract_expenses(r: pd.Series):
    return (
        pick_any(r, COST_OF_REVENUE),
        pick_any(r, RESEARCH_AND_DEVELOPMENT),
        pick_any(r, SELLING_AND_MARKETING),
        pick_any(r, GENERAL_AND_ADMIN),
        pick_any(r, SGA_COMBINED),
        pick_any(r, FACILITIES_DA),
        pick_any(r, PERSONNEL_COSTS),
        pick_any(r, INSURANCE_CLAIMS),
        pick_any(r, OTHER_OPERATING),
    )

def ensure_schema(conn):
    schema = """
    CREATE TABLE IF NOT EXISTS IncomeStatement (
      ticker TEXT, period_ending TEXT,
      total_revenue REAL, cost_of_revenue REAL, research_and_development REAL,
      selling_and_marketing REAL, general_and_admin REAL, sga_combined REAL,
      facilities_da REAL, personnel_costs REAL, insurance_claims REAL,
      other_operating REAL, PRIMARY KEY(ticker, period_ending));
    """
    cur = conn.cursor()
    cur.execute(schema)
    conn.commit()

def store(ticker, conn=None):
    own = conn is None
    if own: conn = sqlite3.connect(DB_PATH)
    ensure_schema(conn)
    cur = conn.cursor()
    df = yf.Ticker(ticker).financials.transpose()
    if df.empty: return
    for idx, row in df.iterrows():
        pe = idx.to_pydatetime() if isinstance(idx, pd.Timestamp) else idx
        vals = extract_expenses(row)
        cur.execute(
            """INSERT OR REPLACE INTO IncomeStatement VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (ticker, clean(pe), clean(row.get("Total Revenue")), *map(clean, vals))
        )
    conn.commit(); cur.close()
    if own: conn.close()

def load_yearly(ticker):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM IncomeStatement WHERE ticker = ?", conn, params=(ticker,))
    conn.close()
    if df.empty: return df
    df["period_ending"] = pd.to_datetime(df["period_ending"])
    df["year_int"] = df["period_ending"].dt.year
    g = df.groupby("year_int", as_index=False).sum(numeric_only=True)
    g["year_label"] = g["year_int"].astype(str)
    return g

def load_ttm(ticker):
    q = yf.Ticker(ticker).quarterly_financials.transpose()
    if len(q) < 4: return pd.DataFrame()
    q = q.head(4)
    ttm_row = {
        "year_label": "TTM",
        "year_int": np.nan,
        "total_revenue": q["Total Revenue"].sum() if "Total Revenue" in q else np.nan,
        "cost_of_revenue": q.apply(lambda r: pick_any(r, COST_OF_REVENUE), axis=1).sum(),
        "research_and_development": q.apply(lambda r: pick_any(r, RESEARCH_AND_DEVELOPMENT), axis=1).sum(),
        "selling_and_marketing": q.apply(lambda r: pick_any(r, SELLING_AND_MARKETING), axis=1).sum(),
        "general_and_admin": q.apply(lambda r: pick_any(r, GENERAL_AND_ADMIN), axis=1).sum(),
        "sga_combined": q.apply(lambda r: pick_any(r, SGA_COMBINED), axis=1).sum(),
        "facilities_da": q.apply(lambda r: pick_any(r, FACILITIES_DA), axis=1).sum(),
        "personnel_costs": q.apply(lambda r: pick_any(r, PERSONNEL_COSTS), axis=1).sum(),
        "insurance_claims": q.apply(lambda r: pick_any(r, INSURANCE_CLAIMS), axis=1).sum(),
        "other_operating": q.apply(lambda r: pick_any(r, OTHER_OPERATING), axis=1).sum(),
    }
    return pd.DataFrame([ttm_row])

def write_html(df, path):
    df.to_html(path, index=False, border=0)
    print(f"Saved → {path}")

def generate_expense_reports(ticker):
    store(ticker)
    annual = load_yearly(ticker)
    if annual.empty:
        print(f"⛔ No annual data found for {ticker}")
        return
    ttm = load_ttm(ticker)
    df = pd.concat([annual, ttm], ignore_index=True)

    df = df[df["total_revenue"].notna() & (df["total_revenue"] != 0)]

    drop_cols = [col for col in df.columns if col not in ["year_label", "year_int", "total_revenue"]
                 and _all_nan_or_zero(df[col])]
    df = df.drop(columns=drop_cols)

    abs_df = df.copy().sort_values("year_label")
    abs_fmt = abs_df.copy()
    for c in abs_fmt.columns[2:]:
        abs_fmt[c] = abs_fmt[c].apply(_fmt_short)
    rename_abs = {
        "year_label": "Year", "total_revenue": "Revenue ($)",
        "cost_of_revenue": "Cost of Revenue ($)", "research_and_development": "R&D ($)",
        "selling_and_marketing": "Sales & Marketing ($)", "general_and_admin": "G&A ($)",
        "sga_combined": "SG&A ($)", "facilities_da": "Facilities / D&A ($)",
        "personnel_costs": "Personnel ($)", "insurance_claims": "Insurance ($)",
        "other_operating": "Other Operating ($)",
    }
    abs_fmt = abs_fmt.rename(columns={k: v for k, v in rename_abs.items() if k in abs_fmt.columns})
    write_html(abs_fmt, os.path.join(OUTPUT_DIR, f"{ticker}_expense_absolute.html"))

    yoy = abs_df.copy()
    for c in yoy.columns[2:]:
        yoy[c] = yoy[c].pct_change().replace([np.inf, -np.inf], np.nan) * 100
    yoy = yoy.dropna(axis=1, how='all', subset=yoy.columns[2:])
    yoy = yoy[yoy.iloc[:, 2:].notna().any(axis=1)]
    rename_yoy = {
        k: v.replace("($)", "Change (%)") for k, v in rename_abs.items() if k in yoy.columns
    }
    yoy = yoy.rename(columns=rename_yoy)
    write_html(yoy, os.path.join(OUTPUT_DIR, f"{ticker}_yoy_expense_change.html"))

    print(f"[{ticker}] ✔ Expense tables generated.")

if __name__ == "__main__":
    generate_expense_reports("AAPL")

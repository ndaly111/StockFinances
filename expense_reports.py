"""
expense_reports.py
Builds annual / quarterly operating-expense tables, stores them in SQLite,
and generates two charts per ticker:

  1) Revenue vs. stacked expenses (absolute $)
  2) Expenses as % of revenue

The public entry-point `generate_expense_reports()` is unchanged, but the
implementation now avoids “database is locked” errors by:

  • using WAL + busy-timeout connections everywhere;
  • running the schema check only once per Python process.
"""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib.ticker import FuncFormatter

from expense_labels import (
    COST_OF_REVENUE,
    RESEARCH_AND_DEVELOPMENT,
    SELLING_AND_MARKETING,
    GENERAL_AND_ADMIN,
    SGA_COMBINED,
    FACILITIES_DA,
    PERSONNEL_COSTS,
    INSURANCE_CLAIMS,
    OTHER_OPERATING,
)

# --------------------------------------------------------------------------- #
#  Constants & one-time set-up                                                #
# --------------------------------------------------------------------------- #
DB_PATH     = "Stock Data.db"
DB_TIMEOUT  = 30                 # seconds – let SQLite wait for the lock
OUTPUT_DIR  = "charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

__all__ = ["generate_expense_reports"]

# internal flag – after the schema is ensured once, we never touch it again
_SCHEMA_READY = False


def _open_conn() -> sqlite3.Connection:
    """
    Unified helper to open a connection **with**:

      • WAL mode   → readers & writers don’t block each other;
      • busy timer → wait ´DB_TIMEOUT´ seconds before raising ‘locked’.
    """
    conn = sqlite3.connect(
        DB_PATH,
        timeout=DB_TIMEOUT,
        check_same_thread=False,      # safe – we never share the object
    )
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute(f"PRAGMA busy_timeout={DB_TIMEOUT * 1000};")
    return conn


# --------------------------------------------------------------------------- #
#  Utility helpers                                                            #
# --------------------------------------------------------------------------- #
def clean_value(val):
    """Convert NaNs → None and datetimes → ISO-8601 strings (for SQLite)."""
    if pd.isna(val):
        return None
    if isinstance(val, (pd.Timestamp, datetime)):
        return val.isoformat()
    return val


# --------------------------------------------------------------------------- #
#  Expense extraction                                                         #
# --------------------------------------------------------------------------- #
def extract_expenses(row: pd.Series):
    def match_any(label_list):
        for key in row.index:
            for lbl in label_list:
                if lbl.lower() in key.lower() and pd.notna(row[key]):
                    return row[key]
        return None

    return (
        match_any(COST_OF_REVENUE),
        match_any(RESEARCH_AND_DEVELOPMENT),
        match_any(SELLING_AND_MARKETING),
        match_any(GENERAL_AND_ADMIN),
        match_any(SGA_COMBINED),
        match_any(FACILITIES_DA),
        match_any(PERSONNEL_COSTS),
        match_any(INSURANCE_CLAIMS),
        match_any(OTHER_OPERATING),
    )


# --------------------------------------------------------------------------- #
#  Schema enforcement                                                         #
# --------------------------------------------------------------------------- #
TABLES = ("IncomeStatement", "QuarterlyIncomeStatement")
_TABLE_SCHEMA = """
    CREATE TABLE IF NOT EXISTS {name} (
        ticker TEXT,
        period_ending TEXT,
        total_revenue REAL,
        cost_of_revenue REAL,
        research_and_development REAL,
        selling_and_marketing REAL,
        general_and_admin REAL,
        sga_combined REAL,
        facilities_da REAL,
        personnel_costs REAL,
        insurance_claims REAL,
        other_operating REAL,
        PRIMARY KEY (ticker, period_ending)
    );
"""


def _ensure_tables(*, drop: bool = False) -> None:
    """Create or (optionally) rebuild the two income-statement tables."""
    conn = _open_conn()
    cur = conn.cursor()

    for tbl in TABLES:
        if drop:
            cur.execute(f"DROP TABLE IF EXISTS {tbl};")
        cur.execute(_TABLE_SCHEMA.format(name=tbl))

    conn.commit()
    conn.close()
    msg = "dropped & recreated" if drop else "ensured (created if missing)"
    print(f"✅ Tables {msg}")


def _ensure_schema_once(*, drop: bool = False) -> None:
    """Run `_ensure_tables()` only the first time in this interpreter."""
    global _SCHEMA_READY
    if _SCHEMA_READY and not drop:
        return
    _ensure_tables(drop=drop)
    _SCHEMA_READY = True


# --------------------------------------------------------------------------- #
#  Storage helpers                                                            #
# --------------------------------------------------------------------------- #
def store_data(ticker: str, *, mode: str = "annual") -> None:
    """
    Fetch *annual* or *quarterly* financials via yfinance and upsert rows
    into the correct table.
    """
    yf_tkr = yf.Ticker(ticker)
    raw_df = (
        yf_tkr.financials.transpose()
        if mode == "annual"
        else yf_tkr.quarterly_financials.transpose()
    )

    if raw_df.empty:
        print(f"⚠️  No {mode} financials found for {ticker}")
        return

    table = "IncomeStatement" if mode == "annual" else "QuarterlyIncomeStatement"
    conn = _open_conn()
    cur = conn.cursor()

    for idx, row in raw_df.iterrows():
        pe = idx.to_pydatetime() if isinstance(idx, pd.Timestamp) else idx
        total_rev = row.get("Total Revenue")

        cost_rev, rnd, mkt, adm, sga, fda, ppl, ins, oth = extract_expenses(row)

        cur.execute(
            f"""
            INSERT OR REPLACE INTO {table}
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """,
            (
                ticker,
                clean_value(pe),
                clean_value(total_rev),
                clean_value(cost_rev),
                clean_value(rnd),
                clean_value(mkt),
                clean_value(adm),
                clean_value(sga),
                clean_value(fda),
                clean_value(ppl),
                clean_value(ins),
                clean_value(oth),
            ),
        )

    conn.commit()
    conn.close()
    print(f"✅ {mode.capitalize()} data stored for {ticker}")


# --------------------------------------------------------------------------- #
#  Data fetchers                                                              #
# --------------------------------------------------------------------------- #
def fetch_yearly_data(ticker: str) -> pd.DataFrame:
    """Aggregate annual rows by calendar year (sum of columns)."""
    conn = _open_conn()
    df = pd.read_sql_query(
        """
        SELECT * FROM IncomeStatement
        WHERE ticker = ?
    """,
        conn,
        params=(ticker,),
    )
    conn.close()

    if df.empty:
        return pd.DataFrame()

    df["period_ending"] = pd.to_datetime(df["period_ending"])
    df["year"] = df["period_ending"].dt.year
    agg_cols = df.columns.difference(["ticker", "period_ending", "year"])

    return df.groupby("year", as_index=False)[agg_cols].sum()


def fetch_ttm_data(ticker: str) -> pd.DataFrame:
    """
    Sum the *last four* quarterly rows and label the result “TTM”.
    Returns an empty DF if we don’t have four aligned quarters yet.
    """
    conn = _open_conn()
    df = pd.read_sql_query(
        """
        SELECT * FROM QuarterlyIncomeStatement
        WHERE ticker = ?
        ORDER BY period_ending DESC
    """,
        conn,
        params=(ticker,),
    )
    conn.close()

    if df.empty:
        return pd.DataFrame()

    df["period_ending"] = pd.to_datetime(df["period_ending"])
    recent = df.head(4).sort_values("period_ending")

    if len(recent) < 4:
        return pd.DataFrame()

    # Verify the quarters are consecutive (Q1..Q4)
    expected = pd.date_range(
        end=recent["period_ending"].max(), periods=4, freq="Q"
    )
    if list(expected.to_period("Q")) != list(
        recent["period_ending"].dt.to_period("Q")
    ):
        return pd.DataFrame()

    ttm = recent.drop(columns=["ticker", "period_ending"]).sum().to_frame().T
    ttm.insert(0, "year", "TTM")
    return ttm


# --------------------------------------------------------------------------- #
#  Chart helpers                                                              #
# --------------------------------------------------------------------------- #
def format_short(x, dec=1):
    if pd.isna(x):
        return "$0"
    absx = abs(x)
    if absx >= 1e12:
        return f"${x/1e12:.{dec}f} T"
    if absx >= 1e9:
        return f"${x/1e9:.{dec}f} B"
    if absx >= 1e6:
        return f"${x/1e6:.{dec}f} M"
    if absx >= 1e3:
        return f"${x/1e3:.{dec}f} K"
    return f"${x:.{dec}f}"


# ---- 1) Revenue vs absolute expenses -------------------------------------- #
def plot_expense_charts(df: pd.DataFrame, ticker: str):
    """Stacked bar: absolute expenses vs revenue."""
    numeric_cols = df.columns.difference(["year"])
    df_plot = df.copy()
    df_plot[numeric_cols] = df_plot[numeric_cols].fillna(0)

    # Remove years where every numeric column is zero
    df_plot = df_plot.loc[~(df_plot[numeric_cols] == 0).all(axis=1)]
    if df_plot.empty:
        print("⛔ Nothing to plot after zero-row filter — skipping chart")
        return

    years = df_plot["year"].astype(str)
    pos = np.arange(len(df_plot))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bottom = np.zeros(len(df_plot))

    categories = [
        ("cost_of_revenue", "Cost of Revenue", "dimgray"),
        ("research_and_development", "R&D", "blue"),
        ("selling_and_marketing", "Sales & Marketing", "purple"),
        ("general_and_admin", "G&A", "pink"),
        ("sga_combined", "SG&A", "mediumpurple"),
        ("facilities_da", "Facilities / D&A", "orange"),
        ("personnel_costs", "Personnel", "brown"),
        ("insurance_claims", "Insurance", "teal"),
        ("other_operating", "Other Op.", "gold"),
    ]

    for col, lbl, color in categories:
        if col in df_plot.columns and df_plot[col].sum() > 0:
            vals = df_plot[col].to_numpy()
            prev_bottom = bottom.copy()
            bars = ax.bar(
                pos - width / 2, vals, width, bottom=bottom, label=lbl, color=color
            )
            for i, b in enumerate(bars):
                h = vals[i]
                if h > 0:
                    ax.text(
                        b.get_x() + b.get_width() / 2,
                        prev_bottom[i] + h / 2,
                        format_short(h, 0),
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=7,
                    )
            bottom += vals

    # Revenue bars
    revs = df_plot["total_revenue"].to_numpy()
    bars = ax.bar(pos + width / 2, revs, width, label="Revenue", color="green")
    for b in bars:
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            format_short(b.get_height(), 0),
            ha="center",
            va="bottom",
            fontsize=8,
            weight="bold",
        )

    ax.set_xticks(pos)
    ax.set_xticklabels(years)
    ax.set_ylabel("Amount")
    ax.set_title(f"Revenue vs Expenses — {ticker}")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: format_short(x, 0)))
    plt.tight_layout()

    outfile = os.path.join(OUTPUT_DIR, f"{ticker}_rev_expense_chart.png")
    plt.savefig(outfile, dpi=100)
    plt.close()
    print(f"✅ Saved chart → {outfile}")


# ---- 2) Expenses as % of revenue ------------------------------------------ #
def plot_expense_percent_chart(df: pd.DataFrame, ticker: str):
    """Stacked bar: expenses as % of revenue."""
    df_plot = df.copy()

    # Coerce to numeric and fill NaN with 0
    df_plot["total_revenue"] = pd.to_numeric(
        df_plot["total_revenue"], errors="coerce"
    ).fillna(0)
    numeric_cols = df_plot.columns.difference(["year", "total_revenue"])
    df_plot[numeric_cols] = df_plot[numeric_cols].apply(
        lambda s: pd.to_numeric(s, errors="coerce").fillna(0)
    )

    # Drop rows where revenue is zero
    df_plot = df_plot.loc[df_plot["total_revenue"] != 0]
    if df_plot.empty:
        print("⛔ No valid revenue years — skipping percent-of-revenue chart")
        return

    # Compute percent-of-revenue safely
    revenue = df_plot["total_revenue"]
    revenue_nonzero = revenue.mask(revenue == 0, np.nan)

    pct_df = pd.DataFrame({"year": df_plot["year"]})
    for col in numeric_cols:
        pct = df_plot[col].divide(revenue_nonzero) * 100
        pct_df[col] = pct.fillna(0)

    # Drop categories that are all zero
    percent_cols = pct_df.columns.difference(["year"])
    nonzero_cats = [col for col in percent_cols if pct_df[col].sum() > 0]
    if not nonzero_cats:
        print("⛔ All expense categories are zero percent — skipping chart")
        return

    years = pct_df["year"].astype(str)
    pos = np.arange(len(pct_df))
    width = 0.6
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bottom = np.zeros(len(pct_df))

    category_info = [
        ("cost_of_revenue", "Cost of Revenue", "dimgray"),
        ("research_and_development", "R&D", "blue"),
        ("selling_and_marketing", "Sales & Marketing", "purple"),
        ("general_and_admin", "G&A", "pink"),
        ("sga_combined", "SG&A", "mediumpurple"),
        ("facilities_da", "Facilities / D&A", "orange"),
        ("personnel_costs", "Personnel", "brown"),
        ("insurance_claims", "Insurance", "teal"),
        ("other_operating", "Other Op.", "gold"),
    ]

    for col, lbl, color in category_info:
        if col in nonzero_cats:
            vals = pct_df[col].to_numpy()
            prev_bottom = bottom.copy()
            bars = ax.bar(
                pos, vals, width, bottom=bottom, label=lbl, color=color
            )
            for i, b in enumerate(bars):
                p = vals[i]
                if p > 2:  # label only segments >2%
                    ax.text(
                        b.get_x() + width / 2,
                        prev_bottom[i] + p / 2,
                        f"{p:.1f}%",
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=7,
                    )
            bottom += vals

    ax.set_xticks(pos)
    ax.set_xticklabels(years)
    ax.set_ylabel("Percent of Revenue")
    ax.set_title(f"Expenses as % of Revenue — {ticker}")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    plt.tight_layout()

    outfile = os.path.join(OUTPUT_DIR, f"{ticker}_expense_percent_chart.png")
    plt.savefig(outfile, dpi=100)
    plt.close()
    print(f"✅ Saved percent-of-revenue chart → {outfile}")


# --------------------------------------------------------------------------- #
#  Public orchestrator                                                        #
# --------------------------------------------------------------------------- #
def generate_expense_reports(
    ticker: str, *, rebuild_schema: bool = False
) -> None:
    """
    High-level helper used by main_remote.py (legacy signature).

    Parameters
    ----------
    ticker : str
        The stock symbol to process.

    rebuild_schema : bool, default False
        True  → DROP + recreate both tables (use with care!)
        False → ensure tables exist, keep existing data.
    """
    _ensure_schema_once(drop=rebuild_schema)

    # Refresh raw data in the DB
    store_data(ticker, mode="annual")
    store_data(ticker, mode="quarterly")

    # Build charts and HTML tables
    yearly = fetch_yearly_data(ticker)
    if yearly.empty:
        print("⛔ No data found — skipping charts")
        return

    ttm = fetch_ttm_data(ticker)
    full = pd.concat([yearly, ttm], ignore_index=True)

    plot_expense_charts(full, ticker)
    plot_expense_percent_chart(full, ticker)


# --------------------------------------------------------------------------- #
#  CLI usage                                                                  #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    generate_expense_reports("AAPL")      # quick manual test

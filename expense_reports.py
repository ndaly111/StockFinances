"""
expense_reports.py
───────────────────────────────────────────────────────────────────────────────
Per-ticker outputs
    1) Revenue-vs-stacked-expense chart        ($)
    2) Expenses-as-%-of-revenue chart          (%)
    3) YoY expense-change HTML table           (%)
    4) Absolute expense-dollar HTML table      ($)
"""

from __future__ import annotations
import os
import sqlite3
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib import colors as mcolors
import numpy as np
import pandas as pd
import yfinance as yf

# ─────────────────────────────────────────────────────────────────────────────
# bring in the alias lists that already live in expense_labels.py
# ─────────────────────────────────────────────────────────────────────────────
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

# master list: (alias_list, pretty label, colour)
_CATEGORY_META: list[tuple[list[str], str, str]] = [
    (COST_OF_REVENUE,           "Cost of Revenue",     "#6d6d6d"),
    (RESEARCH_AND_DEVELOPMENT,  "R&D",                 "#3b5bff"),
    (SELLING_AND_MARKETING,     "Selling & Marketing", "#ffc6e2"),
    (GENERAL_AND_ADMIN,         "G&A",                 "#ffb3c6"),
    (SGA_COMBINED,              "SG&A",                "#c2a5ff"),
    (FACILITIES_DA,             "Facilities / D&A",    "#ffa600"),
    (PERSONNEL_COSTS,           "Personnel",           "#65c51f"),
    (INSURANCE_CLAIMS,          "Insurance / Claims",  "#ffd92f"),
    (OTHER_OPERATING,           "Other Operating",     "#a6a6a6"),
]

DB_PATH, OUTPUT_DIR = "Stock Data.db", "charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)
__all__ = ["generate_expense_reports"]

# ───────────────────────── helpers ────────────────────────────
_SUFFIXES = [(1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")]


def _fmt_short(x: float | None, d: int = 1) -> str:
    if pd.isna(x):
        return ""
    for div, suf in _SUFFIXES:
        if abs(x) >= div:
            return f"${x/div:.{d}f}{suf}"
    return f"${x:.{d}f}"


def _all_nan_or_zero(col: pd.Series) -> bool:
    return (col.replace(0, np.nan).notna().sum() == 0)


def _clean_scalar(v):
    if pd.isna(v):
        return None
    if isinstance(v, (pd.Timestamp, datetime)):
        return v.isoformat()
    return float(v)


# ────────────────────── DB schema / IO ───────────────────────
TABLES = ("IncomeStatement", "QuarterlyIncomeStatement")
_SCHEMA = """
CREATE TABLE IF NOT EXISTS {n}(
  ticker TEXT, period_ending TEXT,
  total_revenue REAL, cost_of_revenue REAL, research_and_development REAL,
  selling_and_marketing REAL, general_and_admin REAL, sga_combined REAL,
  facilities_da REAL, personnel_costs REAL, insurance_claims REAL,
  other_operating REAL,
  PRIMARY KEY(ticker,period_ending)
);
"""


def ensure(drop: bool = False, *, conn: sqlite3.Connection | None = None) -> None:
    own = conn is None
    if own:
        conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    for t in TABLES:
        if drop:
            cur.execute(f"DROP TABLE IF EXISTS {t}")
        cur.execute(_SCHEMA.format(n=t))
    conn.commit()
    cur.close()
    if own:
        conn.close()


def _pick_any(row: pd.Series, aliases: list[str]):
    for col in row.index:
        if pd.notna(row[col]) and any(a.lower() in col.lower() for a in aliases):
            return row[col]
    return np.nan


def _extract_expenses(row: pd.Series) -> list:
    """Return values in the order of _CATEGORY_META."""
    return [_pick_any(row, aliases) for aliases, *_ in _CATEGORY_META]


def store(tkr: str, mode: str = "annual", *, conn: sqlite3.Connection | None = None) -> None:
    data = yf.Ticker(tkr)
    df = data.financials.T if mode == "annual" else data.quarterly_financials.T
    if df.empty:
        return

    own = conn is None
    if own:
        conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    table = "IncomeStatement" if mode == "annual" else "QuarterlyIncomeStatement"

    for idx, row in df.iterrows():
        period_ending = idx.to_pydatetime() if isinstance(idx, pd.Timestamp) else idx
        values = _extract_expenses(row)
        cur.execute(
            f"""INSERT OR REPLACE INTO {table}
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (tkr,
             _clean_scalar(period_ending),
             _clean_scalar(row.get("Total Revenue")),
             *[_clean_scalar(v) for v in values])
        )
    conn.commit()
    cur.close()
    if own:
        conn.close()


# ───────────────────── pull yearly / TTM ─────────────────────
def _yearly(tkr: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM IncomeStatement WHERE ticker=?", conn, params=(tkr,))
    conn.close()
    if df.empty:
        return df
    df["period_ending"] = pd.to_datetime(df["period_ending"])
    df["year_int"] = df["period_ending"].dt.year
    g = df.groupby("year_int", as_index=False).sum(numeric_only=True)
    g["year_label"] = g["year_int"].astype(str)
    return g


def _ttm(tkr: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    q = pd.read_sql_query(
        "SELECT * FROM QuarterlyIncomeStatement WHERE ticker=? ORDER BY period_ending DESC",
        conn, params=(tkr,))
    conn.close()
    if q.empty():
        return q
    q["period_ending"] = pd.to_datetime(q["period_ending"])
    recent = q.head(4).sort_values("period_ending")
    if len(recent) < 4:
        return pd.DataFrame()
    expect = pd.date_range(end=recent["period_ending"].max(), periods=4, freq="Q")
    if list(expect.to_period("Q")) != list(recent["period_ending"].dt.to_period("Q")):
        return pd.DataFrame()
    ttm_df = recent.drop(columns=["ticker", "period_ending"]).sum().to_frame().T
    ttm_df.insert(0, "year_label", "TTM")
    ttm_df["year_int"] = np.nan
    return ttm_df


# ────────────────────── chart helpers ────────────────────────
def _categories(columns: list[str], has_sga_combined: bool) -> list[tuple[str, str, str]]:
    cats: list[tuple[str, str, str]] = []
    for aliases, pretty, colour in _CATEGORY_META:
        key = aliases[0]  # canonical name
        col_key = _alias_to_col(key)
        if col_key not in columns:
            continue
        # if SG&A combined exists, drop its child categories
        if has_sga_combined and pretty in {"G&A", "Selling & Marketing"}:
            continue
        cats.append((pretty, col_key, colour))
    return cats


def _alias_to_col(alias: str) -> str:
    return (alias
            .lower()
            .replace(" ", "_")
            .replace("&", "and")
            .replace("/", "")
            .replace(",", "")
            .replace("__", "_"))


def _register_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename/ensure every category has its own clean column name."""
    out = df.copy()
    for aliases, pretty, _ in _CATEGORY_META:
        col_name = _alias_to_col(aliases[0])
        if col_name in out.columns:
            continue
        out[col_name] = np.nan
        for a in aliases:
            if a in df.columns:
                out[col_name] = out[col_name].fillna(df[a])
    return out


def _to_rgb_float(colour: str) -> tuple[float, float, float]:
    return mcolors.to_rgb(mcolors.to_hex(colour))  # guarantees tuple of floats


def _text_colour(bg_hex: str) -> str:
    r, g, b = _to_rgb_float(bg_hex)
    return "white" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.6 else "black"


def _chart_abs(df: pd.DataFrame, tkr: str) -> None:
    f = df.sort_values("year_int")
    xl = f["year_label"].tolist()
    cats = _categories(f.columns, f["sga_combined"].notna().any())
    fig, ax = plt.subplots(figsize=(11, 6))
    bottom = np.zeros(len(f))
    for label, col, clr in cats:
        v = pd.to_numeric(f[col], errors="coerce").fillna(0).values
        ax.bar(xl, v, bottom=bottom, color=clr, width=.6, label=label)
        bottom += v
    rev = pd.to_numeric(f["total_revenue"], errors="coerce").values
    ax.plot(xl, rev, "k-o", lw=2, label="Revenue")
    ax.set_ylim(0, max(bottom.max(), rev.max()) * 1.1)
    ax.set_title(f"Revenue vs Operating Expenses — {tkr}")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: _fmt_short(x)))
    ax.legend(frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{tkr}_expenses_vs_revenue.png"))
    plt.close(fig)


def _chart_pct(df: pd.DataFrame, ticker: str) -> None:
    f = df.sort_values("year_int").loc[lambda d: d["total_revenue"] != 0]
    x_labels = f["year_label"].tolist()
    cats = _categories(f.columns, f["sga_combined"].notna().any())

    for _, col, _ in cats:
        f[col] = pd.to_numeric(f[col], errors="coerce")
        f[col + "_pct"] = f[col] / pd.to_numeric(f["total_revenue"], errors="coerce") * 100

    fig, ax = plt.subplots(figsize=(11, 4))
    bottom = np.zeros(len(f))
    for label, col, colour in cats:
        vals = f[col + "_pct"].fillna(0).values
        ax.bar(x_labels, vals, bottom=bottom, color=colour, width=.6, zorder=2)
        for x, y0, v in zip(x_labels, bottom, vals):
            if v > 4:
                ax.text(
                    x,
                    y0 + v / 2,
                    f"{v:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=_text_colour(colour),
                )
        bottom += vals

    ax.axhline(100, ls="--", lw=1, color="black", zorder=5)
    ylim = np.ceil((bottom.max() * 1.1) / 10) * 10
    ax.set_ylim(0, ylim)
    ax.set_yticks(np.arange(0, ylim + 1, 10))
    ax.set_ylabel("Percent of Revenue")
    ax.set_title(f"Expenses as % of Revenue — {ticker}")
    ax.legend([c[0] for c in cats], bbox_to_anchor=(1.01, 0.5), loc="center left", frameon=False)
    fig.subplots_adjust(right=0.78, top=0.88)
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, f"{ticker}_expenses_pct_of_rev.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ───────────────────────── tables ─────────────────────────────
def _write_html(df: pd.DataFrame, out_path: str) -> None:
    df.to_html(out_path, index=False, border=0, justify="center")


def generate_expense_reports(tkr: str, *, rebuild_schema: bool = False, conn: sqlite3.Connection | None = None) -> None:
    ensure(rebuild_schema, conn=conn)
    store(tkr, "annual",   conn=conn)
    store(tkr, "quarterly", conn=conn)

    yearly_df = _yearly(tkr)
    if yearly_df.empty:
        print(f"⛔ No data for {tkr}")
        return

    full = _register_columns(yearly_df)
    full = pd.concat([full, _ttm(tkr)], ignore_index=True)

    # valid rows only
    full = full.loc[full["total_revenue"].notna() & (full["total_revenue"] != 0)]

    # ────── charts ──────
    _chart_abs(full, tkr)
    _chart_pct(full, tkr)

    # ────── tables ──────
    base_cols = ["total_revenue"] + [
        _alias_to_col(meta[0][0]) for meta in _CATEGORY_META
    ]
    cols = ["year_label"] + [c for c in base_cols if c in full.columns]

    # absolute $
    abs_df = (
        full[cols]
        .copy()
        .sort_values("year_label")
        .astype({c: "float64" for c in cols[1:]})
    )
    abs_df = abs_df.drop(columns=[c for c in abs_df.columns[1:] if _all_nan_or_zero(abs_df[c])])
    for c in abs_df.columns[1:]:
        abs_df[c] = abs_df[c].apply(_fmt_short)
    # rename nicely
    ren_abs = {"year_label": "Year", "total_revenue": "Revenue ($)"}
    for _, pretty, _ in _CATEGORY_META:
        col_key = _alias_to_col(pretty.replace(" & ", " and "))
        ren_abs[col_key] = f"{pretty} ($)"
    abs_df.rename(columns=ren_abs, inplace=True)
    _write_html(abs_df, os.path.join(OUTPUT_DIR, f"{tkr}_expense_absolute.html"))

    # YoY %
    yoy = full[cols].copy().sort_values("year_label").astype({c: "float64" for c in cols[1:]})
    for c in cols[1:]:
        yoy[c] = (
            yoy[c]
            .pct_change()
            .replace([np.inf, -np.inf], np.nan)
            .mul(100)
            .round(2)
        )
    yoy = yoy.drop(columns=[c for c in yoy.columns[1:] if yoy[c].notna().sum() == 0])
    yoy = yoy[yoy.iloc[:, 1:].notna().any(axis=1)]
    ren_pct = {"year_label": "Year", "total_revenue": "Revenue Change (%)"}
    for _, pretty, _ in _CATEGORY_META:
        col_key = _alias_to_col(pretty.replace(" & ", " and "))
        ren_pct[col_key] = f"{pretty} Change (%)"
    yoy.rename(columns=ren_pct, inplace=True)
    _write_html(yoy, os.path.join(OUTPUT_DIR, f"{tkr}_yoy_expense_change.html"))

    print(f"[{tkr}] ✔ charts & tables generated")


if __name__ == "__main__":
    generate_expense_reports("AAPL")

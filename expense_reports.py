# expense_reports.py
# -------------------------------------------------------------------------------
# Per-ticker outputs
#   1) Revenue-vs-stacked-expense chart        ($)
#   2) Expenses-as-%-of-revenue chart          (%)
#   3) YoY expense-change HTML table           (%)
#   4) Absolute expense-dollar HTML table      ($)
# -------------------------------------------------------------------------------

from __future__ import annotations

import os, sqlite3
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib import colors as mcolors
from matplotlib.ticker import FuncFormatter

# ──────────────────────────────────────────────────────────────
# label aliases live in expense_labels.py
# ──────────────────────────────────────────────────────────────
from expense_labels import (
    COST_OF_REVENUE,
    FACILITIES_DA,
    GENERAL_AND_ADMIN,
    INSURANCE_CLAIMS,
    OTHER_OPERATING,
    PERSONNEL_COSTS,
    RESEARCH_AND_DEVELOPMENT,
    SELLING_AND_MARKETING,
    SGA_COMBINED,
    IGNORE_CATEGORIES,
)

DB_PATH    = "Stock Data.db"
OUTPUT_DIR = "charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

__all__ = ["generate_expense_reports"]

# ─────────────────────────── helpers ──────────────────────────
_SUFFIXES = [(1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")]


def _fmt_short(x: float, d: int = 1) -> str:
    """Pretty-print numbers with K/M/B/T suffixes."""
    if pd.isna(x):
        return ""
    for div, suf in _SUFFIXES:
        if abs(x) >= div:
            return f"${x / div:.{d}f}{suf}"
    return f"${x:.{d}f}"


def _all_nan_or_zero(col: pd.Series) -> bool:
    return (col.replace(0, np.nan).notna().sum() == 0)


def _clean(v):
    if pd.isna(v):
        return None
    return v.isoformat() if isinstance(v, (pd.Timestamp, datetime)) else v


def _pick_any(row: pd.Series, labels: list[str]):
    """Return the first non-null column whose name contains ≥1 label substring.

    Columns matching any ignore pattern are skipped to avoid double-counting
    aggregate totals such as "Operating Expense".
    """
    for k in row.index:
        kl = k.lower()
        if any(ig.lower() in kl for ig in IGNORE_CATEGORIES):
            continue
        if pd.notna(row[k]) and any(lbl.lower() in kl for lbl in labels):
            return row[k]
    return None


def _extract_expenses(r: pd.Series):
    """Tuple of nine expense buckets (may include None)."""
    return (
        _pick_any(r, COST_OF_REVENUE),
        _pick_any(r, RESEARCH_AND_DEVELOPMENT),
        _pick_any(r, SELLING_AND_MARKETING),
        _pick_any(r, GENERAL_AND_ADMIN),
        _pick_any(r, SGA_COMBINED),
        _pick_any(r, FACILITIES_DA),
        _pick_any(r, PERSONNEL_COSTS),
        _pick_any(r, INSURANCE_CLAIMS),
        _pick_any(r, OTHER_OPERATING),
    )


# ───────────────────────── category meta ──────────────────────
_CATEGORIES = [
    ("Cost of Revenue",     "cost_of_revenue",          "#6d6d6d"),
    ("R&D",                 "research_and_development", "#4287f5"),
    ("Selling & Marketing", "selling_and_marketing",    "#ffc6e2"),
    ("G&A",                 "general_and_admin",        "#ffb3c6"),
    ("SG&A",                "sga_combined",             "#c2a5ff"),
    ("Facilities / D&A",    "facilities_da",            "#ffa600"),
    ("Personnel",           "personnel_costs",          "#8dd3c7"),
    ("Insurance / Claims",  "insurance_claims",         "#b3de69"),
    ("Other Operating",     "other_operating",          "#bc80bd"),
]

# ────────────────────── database schema / IO ─────────────────────
TABLES  = ("IncomeStatement", "QuarterlyIncomeStatement")
_SCHEMA = """
CREATE TABLE IF NOT EXISTS {n}(
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


def _ensure(drop: bool = False, *, conn=None):
    own = conn is None
    conn = conn or sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    for t in TABLES:
        if drop:
            cur.execute(f"DROP TABLE IF EXISTS {t}")
        cur.execute(_SCHEMA.format(n=t))
    conn.commit()
    cur.close()
    if own:
        conn.close()


def _store(tkr: str, mode: str = "annual", *, conn=None):
    yf_tkr = yf.Ticker(tkr)
    # Safely access financials (may be None)
    raw_df = yf_tkr.financials if mode == "annual" else yf_tkr.quarterly_financials
    if raw_df is None or raw_df.empty:
        return
    df = raw_df.transpose()
    if df.empty:
        return

    own = conn is None
    conn = conn or sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    table = "IncomeStatement" if mode == "annual" else "QuarterlyIncomeStatement"

    for idx, row in df.iterrows():
        pe   = idx.to_pydatetime() if isinstance(idx, pd.Timestamp) else idx
        vals = _extract_expenses(row)
        cur.execute(
            f"INSERT OR REPLACE INTO {table} VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (tkr, _clean(pe), _clean(row.get("Total Revenue")),
             *[_clean(v) for v in vals]),
        )
    conn.commit()
    cur.close()
    if own:
        conn.close()


# ─────────────────────── pull yearly / TTM ──────────────────────
def _yearly(tkr: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql_query(
        "SELECT * FROM IncomeStatement WHERE ticker=?", conn, params=(tkr,))
    conn.close()

    if df.empty:
        return df

    df["period_ending"] = pd.to_datetime(df["period_ending"])
    df["year_int"]      = df["period_ending"].dt.year
    g = df.groupby("year_int", as_index=False).sum(numeric_only=True)
    g["year_label"] = g["year_int"].astype(str)
    return g


def _ttm(tkr: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    q    = pd.read_sql_query(
        "SELECT * FROM QuarterlyIncomeStatement WHERE ticker=? "
        "ORDER BY period_ending DESC",
        conn, params=(tkr,))
    conn.close()

    if q.empty:
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


# ───────────────────────── chart helpers ─────────────────────────
def _cats(df: pd.DataFrame, combo_present: bool):
    cats = _CATEGORIES.copy()
    if combo_present:  # hide S&M + G&A when combined SG&A exists
        cats = [c for c in cats
                if c[1] not in ("general_and_admin", "selling_and_marketing")]
    return [c for c in cats if c[1] in df.columns]


def _nonzero_categories(df: pd.DataFrame, cats: list[tuple[str, str, str]], *, tol: float = 1e-9):
    """Keep only categories that have a non-trivial value in the frame."""

    keep = []
    for label, col, colour in cats:
        series = pd.to_numeric(df[col], errors="coerce").fillna(0)
        if series.abs().sum() > tol:
            keep.append((label, col, colour))
    return keep


def _chart_abs(df: pd.DataFrame, tkr: str):
    f      = df.sort_values("year_int")
    x_lab  = f["year_label"].tolist()
    idx    = np.arange(len(f))
    width  = 0.38
    cats_all = _cats(f, f["sga_combined"].notna().any())
    cats     = _nonzero_categories(f, cats_all)
    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    revenue_vals = f["total_revenue"].fillna(0).astype(float).values
    revenue_bars = ax.bar(
        idx - width / 2,
        revenue_vals,
        width=width,
        color="#155724",
        label="Revenue",
    )

    bottom = np.zeros(len(f), dtype=float)
    seg_labels = []
    for lbl, col, colour in cats:
        vals = f[col].fillna(0).astype(float).values
        bars = ax.bar(
            idx + width / 2,
            vals,
            bottom=bottom,
            color=colour,
            width=width,
            label=lbl,
        )
        for bar, val in zip(bars, vals):
            seg_labels.append((bar, float(val), colour))
        bottom += vals

    ymax = max(bottom.max(), revenue_vals.max()) * 1.1 if len(f) else 1
    ax.set_ylim(0, ymax)
    ax.set_xticks(idx)
    ax.set_xticklabels(x_lab)
    ax.set_title(f"Revenue vs Operating Expenses — {tkr}")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda val, _: _fmt_short(val)))
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, frameon=False, ncol=2)
    for bar, val, colour in seg_labels:
        if val <= 0:
            continue
        height = bar.get_height()
        x = bar.get_x() + bar.get_width() / 2
        base = bar.get_y()
        if height < ymax * 0.03:
            y = base + height + ymax * 0.01
            va = "bottom"
            clr = "black"
        else:
            y = base + height / 2
            va = "center"
            clr = _txt_colour(colour)
        ax.text(
            x,
            y,
            _fmt_short(val),
            ha="center",
            va=va,
            fontsize=8,
            color=clr,
        )

    for bar, val in zip(revenue_bars, revenue_vals):
        if val <= 0:
            continue
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_height() + ymax * 0.01
        ax.text(
            x,
            y,
            _fmt_short(val),
            ha="center",
            va="bottom",
            fontsize=8,
            color="#155724",
        )

    plt.tight_layout()
    for name in (
        f"{tkr}_expenses_vs_revenue.png",
        f"{tkr}_rev_expense_chart.png",
    ):
        plt.savefig(os.path.join(OUTPUT_DIR, name))
    plt.close()


def _txt_colour(clr: str) -> str:
    r, g, b = mcolors.to_rgb(clr)
    return "white" if (0.299*r + 0.587*g + 0.114*b) < 0.6 else "black"


def _chart_pct(df: pd.DataFrame, tkr: str):
    f      = df.sort_values("year_int").loc[
                 lambda d: d["total_revenue"] != 0].copy()
    x      = f["year_label"].tolist()
    cats_all = _cats(f, f["sga_combined"].notna().any())
    cats     = _nonzero_categories(f, cats_all)

    # add percent columns
    for _, col, _ in cats:
        f[col + "_pct"] = pd.to_numeric(f[col], errors="coerce") \
                          / f["total_revenue"].astype(float) * 100

    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    bottom  = np.zeros(len(f), dtype=float)

    for label, col, colour in cats:
        vals = f[col + "_pct"].fillna(0).values.astype(float)
        ax.bar(x, vals, bottom=bottom, color=colour, width=0.6, zorder=2)

        for xi, y0, v in zip(x, bottom, vals):
            if v > 4:
                ax.text(xi, y0 + v/2, f"{v:.1f}%",
                        ha="center", va="center", fontsize=8,
                        color=_txt_colour(colour))
        bottom += vals

    ylim = np.ceil((bottom.max()*1.1) / 10) * 10
    ax.set_ylim(0, ylim)
    ax.set_yticks(np.arange(0, ylim+1, 10))
    ax.set_ylabel("Percent of Revenue")
    ax.set_title(f"Expenses as % of Revenue — {tkr}")
    if cats:
        ax.legend([c[0] for c in cats],
                  bbox_to_anchor=(1.01, 0.5),
                  loc="center left", frameon=False)
    fig.subplots_adjust(right=0.78, top=0.88)
    plt.tight_layout()

    for name in (
        f"{tkr}_expenses_pct_of_rev.png",
        f"{tkr}_expense_percent_chart.png",
    ):
        fig.savefig(
            os.path.join(OUTPUT_DIR, name), dpi=120, bbox_inches="tight"
        )
    plt.close()
    print(
        f"[{tkr}] expense-% chart saved → "
        f"{os.path.join(OUTPUT_DIR, f'{tkr}_expenses_pct_of_rev.png')}"
    )


# ─────────────────────────── tables ────────────────────────────
def _write_html(df: pd.DataFrame, path: str):
    df.to_html(path, index=False, border=0, justify="center")


_TABLE_COLS = [
    "total_revenue", "cost_of_revenue", "research_and_development",
    "selling_and_marketing", "general_and_admin", "sga_combined",
    "facilities_da", "personnel_costs", "insurance_claims",
    "other_operating",
]


# ───────────────────── cost-of-revenue adjustment ─────────────────────
def _dedupe_d_and_a(df: pd.DataFrame):
    """
    Remove stand-alone D&A from Cost-of-Revenue so it isn't counted twice.
    Does **not** let CoR go negative.
    """
    if "cost_of_revenue" not in df.columns or "facilities_da" not in df.columns:
        return df

    adj = df["cost_of_revenue"] - df["facilities_da"]
    df["cost_of_revenue"] = adj.clip(lower=0)
    return df


# ─────────────────────────── main API ───────────────────────────
def generate_expense_reports(tkr: str, *, rebuild_schema: bool = False, conn=None):
    """Main entry-point – called once per ticker."""

    _ensure(drop=rebuild_schema, conn=conn)
    _store(tkr, "annual",     conn=conn)
    _store(tkr, "quarterly",  conn=conn)

    yearly_df = _yearly(tkr)
    if yearly_df.empty:
        print(f"⛔  No data for {tkr}")
        return

    full = pd.concat([yearly_df, _ttm(tkr)], ignore_index=True)

    # keep only rows with real revenue
    full = full.loc[full["total_revenue"].notna() & (full["total_revenue"] != 0)].copy()

    # numeric conversion
    for col in _TABLE_COLS:
        if col in full.columns:
            full[col] = pd.to_numeric(full[col], errors="coerce")

    # ─── **FIX** double-counting: subtract D&A from Cost-of-Revenue ───
    full = _dedupe_d_and_a(full)

    # ─── Charts ───────────────────────────────────────────────
    _chart_abs(full, tkr)
    _chart_pct(full, tkr)

    # ─── Absolute $ table ─────────────────────────────────────
    abs_cols = ["year_label"] + [c for c in _TABLE_COLS if c in full.columns]
    abs_df   = full[abs_cols].sort_values("year_label")
    abs_df   = abs_df.drop(columns=[c for c in abs_df.columns[1:]
                                    if _all_nan_or_zero(abs_df[c])])

    abs_fmt = abs_df.copy()
    for c in abs_fmt.columns[1:]:
        abs_fmt[c] = abs_fmt[c].apply(_fmt_short)

    rename_abs = {
        "year_label":          "Year",
        "total_revenue":       "Revenue ($)",
        "cost_of_revenue":     "Cost of Revenue ($)",
        "research_and_development": "R&D ($)",
        "selling_and_marketing":     "Sales & Marketing ($)",
        "general_and_admin":   "G&A ($)",
        "sga_combined":        "SG&A ($)",
        "facilities_da":       "Facilities / D&A ($)",
        "personnel_costs":     "Personnel ($)",
        "insurance_claims":    "Insurance / Claims ($)",
        "other_operating":     "Other Operating ($)",
    }
    abs_fmt.rename(columns=rename_abs, inplace=True)
    _write_html(abs_fmt,
                os.path.join(OUTPUT_DIR, f"{tkr}_expense_absolute.html"))

    # ─── YoY-% table ─────────────────────────────────────────
    yoy = full[abs_cols].sort_values("year_label").copy()
    for c in abs_cols[1:]:
        yoy[c] = (yoy[c].pct_change() * 100).round(2)

    yoy.drop(columns=[c for c in yoy.columns[1:]
                      if yoy[c].notna().sum() == 0], inplace=True)
    yoy = yoy[yoy.iloc[:, 1:].notna().any(axis=1)]

    rename_pct = {k: v.replace("($)", "Change (%)")
                  for k, v in rename_abs.items()}
    yoy.rename(columns=rename_pct, inplace=True)
    _write_html(yoy,
                os.path.join(OUTPUT_DIR, f"{tkr}_yoy_expense_change.html"))

    print(f"[{tkr}] ✔ charts & tables generated")


# ────────────────────────── CLI helper ─────────────────────────
if __name__ == "__main__":
    generate_expense_reports("AAPL")

# expense_reports.py
# -----------------------------------------------------------------------
# Per-ticker outputs
#   1) Revenue-vs-stacked-expense chart        ($)
#   2) Expenses-as-%-of-revenue chart          (%)
#   3) YoY expense-change HTML table           (%)
#   4) Absolute expense-dollar HTML table      ($)
# -----------------------------------------------------------------------

from __future__ import annotations
import os, re, sqlite3
from datetime import datetime
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib import colors as mcolors
import numpy as np
import pandas as pd
import yfinance as yf

# -----------------------------------------------------------------------
#   CATEGORY DEFINITIONS (single source of truth)
# -----------------------------------------------------------------------
from expense_labels import CATEGORY_META   #  [(alias_list, colour), …]

# helper ────────────────────────────────────────────────────────────────
def _slug(text: str) -> str:
    """Canonical snake-case key →  'Cost Of Revenue' → 'cost_of_revenue'."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")

# build look-ups once
_ALIASES: Dict[str, str] = {}               # any alias  → canonical slug
_META: Dict[str, Tuple[str, str]] = {}      # canonical  → (pretty, colour)
_CANON_ORDER: List[str] = []                # canonical order for charts

for alias_list, colour in CATEGORY_META:
    pretty = alias_list[0]                  # first alias = human-readable
    canon  = _slug(pretty)
    _META[canon] = (pretty, colour)
    _CANON_ORDER.append(canon)
    for a in alias_list:
        _ALIASES[_slug(a)] = canon

# -----------------------------------------------------------------------
DB_PATH, OUTPUT_DIR = "Stock Data.db", "charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)
__all__ = ["generate_expense_reports"]

# number formatter ───────────────────────────────────────────────────────
_SUFFIXES = [(1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")]
def _fmt_short(x: float, d: int = 1) -> str:
    if pd.isna(x):
        return ""
    for div, suf in _SUFFIXES:
        if abs(x) >= div:
            return f"${x/div:.{d}f}{suf}"
    return f"${x:.{d}f}"

def _all_nan_or_zero(col: pd.Series) -> bool:
    return (pd.to_numeric(col, errors="coerce")
              .replace(0, np.nan)
              .notna()
              .sum() == 0)

# -----------------------------------------------------------------------
#   SQLite schema & helpers
# -----------------------------------------------------------------------
TABLES = ("IncomeStatement", "QuarterlyIncomeStatement")
_SCHEMA_COLS = (
    "total_revenue, " +
    ", ".join(_CANON_ORDER)      # dynamic columns
)
SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {{n}}(
  ticker TEXT,
  period_ending TEXT,
  {_SCHEMA_COLS},
  PRIMARY KEY(ticker, period_ending)
);"""

def ensure(drop: bool = False, *, conn=None):
    own = conn is None
    if own:
        conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    for t in TABLES:
        if drop:
            cur.execute(f"DROP TABLE IF EXISTS {t}")
        cur.execute(SCHEMA.format(n=t))
    conn.commit(); cur.close()
    if own: conn.close()

# -----------------------------------------------------------------------
#   Yahoo → DB
# -----------------------------------------------------------------------
def _pick_value(row: pd.Series, aliases: List[str]):
    """Return first numeric hit for any alias in row, else None."""
    for k, v in row.items():
        if pd.notna(v) and _slug(k) in (_slug(a) for a in aliases):
            try:
                return float(v)
            except Exception:    # pragma: no cover
                continue
    return None

def _extract_expense_dict(row: pd.Series) -> Dict[str, float | None]:
    d: Dict[str, float | None] = {}
    for alias_list, _ in CATEGORY_META:
        canon = _slug(alias_list[0])
        d[canon] = _pick_value(row, alias_list)
    return d

def store(tkr: str, mode: str = "annual", *, conn=None):
    fin = yf.Ticker(tkr)
    df = fin.financials.T if mode == "annual" else fin.quarterly_financials.T
    if df.empty:
        return
    own = conn is None
    if own:
        conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    table = "IncomeStatement" if mode == "annual" else "QuarterlyIncomeStatement"

    for idx, row in df.iterrows():
        pe = idx.to_pydatetime() if isinstance(idx, pd.Timestamp) else idx
        vals = _extract_expense_dict(row)
        sql = f"""INSERT OR REPLACE INTO {table} (
                    ticker, period_ending, total_revenue, {_SCHEMA_COLS})
                  VALUES ({",".join("?"*(3+len(_CANON_ORDER)))})"""
        data = [
            tkr,
            pe if isinstance(pe, str) else pe.strftime("%Y-%m-%d"),
            float(row.get("Total Revenue", np.nan)),
        ] + [vals[c] for c in _CANON_ORDER]
        cur.execute(sql, data)

    conn.commit(); cur.close()
    if own: conn.close()

# -----------------------------------------------------------------------
#   Pull yearly & TTM
# -----------------------------------------------------------------------
def _read_table(query: str, tkr: str):
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(query, conn, params=(tkr,))

def yearly(tkr: str):
    df = _read_table("SELECT * FROM IncomeStatement WHERE ticker=?", tkr)
    if df.empty:
        return df
    df["period_ending"] = pd.to_datetime(df["period_ending"])
    df["year_int"]      = df["period_ending"].dt.year
    g = df.groupby("year_int", as_index=False).sum(numeric_only=True)
    g["year_label"] = g["year_int"].astype(str)
    return g

def ttm(tkr: str):
    q = _read_table(
        "SELECT * FROM QuarterlyIncomeStatement WHERE ticker=? ORDER BY period_ending DESC",
        tkr,
    )
    if q.empty:
        return q
    q["period_ending"] = pd.to_datetime(q["period_ending"])
    recent = q.head(4).sort_values("period_ending")
    if len(recent) < 4:
        return pd.DataFrame()
    expect = pd.date_range(end=recent["period_ending"].max(), periods=4, freq="Q")  # noqa: E501
    if list(expect.to_period("Q")) != list(recent["period_ending"].dt.to_period("Q")):
        return pd.DataFrame()
    ttm_df = recent.drop(columns=["ticker", "period_ending"]).sum().to_frame().T
    ttm_df.insert(0, "year_label", "TTM")
    ttm_df["year_int"] = np.nan
    return ttm_df

# -----------------------------------------------------------------------
#   Chart helpers
# -----------------------------------------------------------------------
def _cats(df: pd.DataFrame, combo: bool):
    cols = _CANON_ORDER.copy()
    if combo:
        for c in ("general_and_administrative_expense",
                  "selling_and_marketing_expense"):
            if c in cols:
                cols.remove(c)
    cols = [c for c in cols if c in df.columns]
    return [( _META[c][0], c, _META[c][1]) for c in cols]

def _text_colour(hex_colour: str) -> str:
    r, g, b = mcolors.to_rgb(hex_colour)
    return "white" if (0.299*r + 0.587*g + 0.114*b) < 0.6 else "black"

# -----------------------------------------------------------------------
#   Charts
# -----------------------------------------------------------------------
def chart_abs(df: pd.DataFrame, tkr: str):
    f  = df.sort_values("year_int")
    xl = f["year_label"].tolist()
    cats = _cats(f, f["sga_combined"].notna().any())

    fig, ax = plt.subplots(figsize=(11, 6))
    bot = np.zeros(len(f), dtype=float)
    for label, col, colour in cats:
        v = pd.to_numeric(f[col], errors="coerce").fillna(0).to_numpy(float)
        ax.bar(xl, v, bottom=bot, color=colour, width=.6, label=label)
        bot += v

    ax.plot(xl, f["total_revenue"], "k-o", lw=2, label="Revenue")
    ax.set_ylim(0, max(bot.max(), f["total_revenue"].max()) * 1.1)
    ax.set_title(f"Revenue vs Operating Expenses — {tkr}")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: _fmt_short(x)))
    ax.legend(frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{tkr}_expenses_vs_revenue.png"))
    plt.close()

def chart_pct(df: pd.DataFrame, ticker: str):
    f = df.sort_values("year_int").loc[lambda d: d["total_revenue"] != 0].copy()
    x_labels = f["year_label"].tolist()
    cats = _cats(f, f["sga_combined"].notna().any())

    # compute %
    for _, col, _ in cats:
        f[col] = pd.to_numeric(f[col], errors="coerce")
        f[col + "_pct"] = f[col] / pd.to_numeric(f["total_revenue"], errors="coerce") * 100

    fig, ax = plt.subplots(figsize=(11, 4))
    bottoms = np.zeros(len(f), dtype=float)

    for label, col, colour in cats:
        vals = f[col + "_pct"].fillna(0).to_numpy(float)
        ax.bar(x_labels, vals, bottom=bottoms, color=colour, width=.6)
        for x, y0, v in zip(x_labels, bottoms, vals):
            if v > 4:
                ax.text(x, y0 + v/2, f"{v:.1f}%", ha="center", va="center",
                        fontsize=8, color=_text_colour(colour))
        bottoms += vals

    ax.axhline(100, ls="--", lw=1, color="black")
    ylim = np.ceil((bottoms.max()*1.08) / 10) * 10
    ax.set_ylim(0, ylim)
    ax.set_yticks(np.arange(0, ylim+1, 10))
    ax.set_ylabel("Percent of Revenue")
    ax.set_title(f"Expenses as % of Revenue — {ticker}")
    ax.legend([c[0] for c in cats], bbox_to_anchor=(1.01, 0.5),
              loc="center left", frameon=False)
    fig.subplots_adjust(right=0.78, top=0.88)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"{ticker}_expenses_pct_of_rev.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[{ticker}] expense-% chart saved → {out}")

# -----------------------------------------------------------------------
#   Tables
# -----------------------------------------------------------------------
def _write_html(df: pd.DataFrame, path: str):
    df.to_html(path, index=False, border=0, justify="center")

# -----------------------------------------------------------------------
#   ENTRY-POINT
# -----------------------------------------------------------------------
def generate_expense_reports(tkr: str, *, rebuild_schema: bool = False, conn=None):
    ensure(drop=rebuild_schema, conn=conn)
    store(tkr, "annual",   conn=conn)
    store(tkr, "quarterly", conn=conn)

    yearly_df = yearly(tkr)
    if yearly_df.empty:
        print(f"⛔ No data for {tkr}")
        return

    full = pd.concat([yearly_df, ttm(tkr)], ignore_index=True)
    full = full.loc[full["total_revenue"].notna() & (full["total_revenue"] != 0)]

    # charts
    chart_abs(full, tkr)
    chart_pct(full, tkr)

    # ---- HTML tables --------------------------------------------------
    base_cols = ["total_revenue"] + [c for c in _CANON_ORDER if c in full.columns]
    cols      = ["year_label"] + base_cols

    # Absolute-$ table
    abs_df = full[cols].sort_values("year_label")
    abs_df = abs_df.drop(columns=[c for c in abs_df.columns[1:] if _all_nan_or_zero(abs_df[c])])
    fmt    = abs_df.copy()
    for c in fmt.columns[1:]:
        fmt[c] = fmt[c].apply(_fmt_short)
    rename_abs = {"year_label": "Year", "total_revenue": "Revenue ($)"}
    for canon in _CANON_ORDER:
        if canon in fmt.columns:
            rename_abs[canon] = f"{_META[canon][0]} ($)"
    fmt.rename(columns=rename_abs, inplace=True)
    _write_html(fmt, os.path.join(OUTPUT_DIR, f"{tkr}_expense_absolute.html"))

    # YoY-% table
    yoy = full[cols].sort_values("year_label").copy()
    for c in cols[1:]:
        yoy[c] = pd.to_numeric(yoy[c], errors="coerce")
        yoy[c] = yoy[c].pct_change().replace([np.inf, -np.inf], np.nan) * 100
        yoy[c] = yoy[c].round(2)
    yoy = yoy.drop(columns=[c for c in yoy.columns[1:] if yoy[c].notna().sum() == 0])
    yoy  = yoy[yoy.iloc[:, 1:].notna().any(axis=1)]
    rename_pct = {"year_label": "Year", "total_revenue": "Revenue Change (%)"}
    for canon in _CANON_ORDER:
        if canon in yoy.columns:
            rename_pct[canon] = f"{_META[canon][0]} Change (%)"
    yoy.rename(columns=rename_pct, inplace=True)
    _write_html(yoy, os.path.join(OUTPUT_DIR, f"{tkr}_yoy_expense_change.html"))

    print(f"[{tkr}] ✔ charts & tables generated")

# -----------------------------------------------------------------------
if __name__ == "__main__":
    generate_expense_reports("AAPL")

#!/usr/bin/env python3
# index_growth_table.py – 2025-08-05 + “auto-repair” patch
# --------------------------------------------------------
# • Logs implied growth + P/E for SPY & QQQ
# • Filters out obviously bad growth (>100 % or <–50 %)
# • When run, cleans any existing out-of-range rows for
#   the last 7 days and recomputes them on the spot
# • Generates PNG charts + summary tables
# • Returns SPY-vs-QQQ overview HTML for the homepage
# --------------------------------------------------------

import os, sqlite3, numpy as np, pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, FuncFormatter

DB_PATH, CHART_DIR = "Stock Data.db", "charts"
IDXES = ["SPY", "QQQ"]
os.makedirs(CHART_DIR, exist_ok=True)

# ───── SANITY LIMITS (edit if needed) ────────────────────
GROWTH_LO = -0.50      # -50 %
GROWTH_HI =  1.00      # +100 %

# ─── CSS + table helpers (unchanged) ─────────────────────
SUMMARY_CSS = """
<style>
.summary-table{width:100%;border-collapse:collapse;
  font-family:Verdana,Arial,sans-serif;font-size:12px;
  border:3px solid #003366;}
.summary-table th{background:#f2f2f2;padding:4px 6px;
  border:1px solid #B0B0B0;text-align:center;}
.summary-table td{padding:4px 6px;border:1px solid #B0B0B0;text-align:center;}
</style>
"""
def _pct_color(v):
    try:
        v = float(v)
        if v <= 30: return "color:#008800;font-weight:bold"
        if v >= 70: return "color:#CC0000;font-weight:bold"
    except: pass
    return ""
def _build_html(df):
    return (df.style.hide(axis="index")
                 .map(_pct_color, subset="%ctile")
                 .set_table_attributes('class="summary-table"')
            ).to_html()

# ─── DB schema (unchanged) ───────────────────────────────
def _ensure_tables(conn):
    conn.executescript("""
      CREATE TABLE IF NOT EXISTS Index_Growth_History (
        Date TEXT, Ticker TEXT, Growth_Type TEXT, Implied_Growth REAL,
        PRIMARY KEY (Date,Ticker,Growth_Type));
      CREATE TABLE IF NOT EXISTS Index_PE_History (
        Date TEXT, Ticker TEXT, PE_Type TEXT, PE_Ratio REAL,
        PRIMARY KEY (Date,Ticker,PE_Type));
      CREATE TABLE IF NOT EXISTS Treasury_Yield_History (
        Date TEXT PRIMARY KEY, TenYr REAL);
    """)

# ─── Yield helpers (unchanged) ───────────────────────────
def _latest_yield():
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT TenYr FROM Treasury_Yield_History "
            "ORDER BY Date DESC LIMIT 1").fetchone()
    return row[0] if row else None
def _resolve_yield(passed):
    if passed is not None:
        return float(passed)
    y = _latest_yield()
    if y is None:
        raise RuntimeError("No treasury_yield supplied and DB is empty.")
    return float(y)

# ─── P/E helpers (unchanged) ─────────────────────────────
def _latest_pe(tk, pe_type="TTM"):
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT PE_Ratio FROM Index_PE_History "
            "WHERE Ticker=? AND PE_Type=? ORDER BY Date DESC LIMIT 1",
            (tk, pe_type)
        ).fetchone()
    return row[0] if row else None
def _fetch_pe(tk):
    info = yf.Ticker(tk).info or {}
    ttm = info.get("trailingPE") or _latest_pe(tk, "TTM")
    fwd = info.get("forwardPE")  or _latest_pe(tk, "Forward")
    return ttm, fwd

# ─── Growth calculation + filter  ───────────────────────
def _growth(pe, y):
    if pe is None or pd.isna(pe): return None
    try:
        g = (pe / 10) ** 0.1 + y - 1
        if g < GROWTH_LO or g > GROWTH_HI:
            return None        # filter out unreasonable values
        return g
    except (ValueError, ZeroDivisionError):
        return None

# ─── Patch bad rows from last week before today’s log ───
def _repair_last_week():
    cutoff = (datetime.today() - timedelta(days=7)).strftime("%Y-%m-%d")
    with sqlite3.connect(DB_PATH) as conn:
        for tk in IDXES:
            rows = conn.execute(
                "SELECT Date, Implied_Growth FROM Index_Growth_History "
                "WHERE Ticker=? AND Growth_Type='TTM' AND Date>=?",
                (tk, cutoff)
            ).fetchall()
            for d, g in rows:
                if g is None or g < GROWTH_LO or g > GROWTH_HI:
                    print(f"[repair] Removing out-of-range growth "
                          f"{g:.2% if g else g} for {tk} on {d}")
                    conn.execute(
                        "DELETE FROM Index_Growth_History "
                        "WHERE Date=? AND Ticker=? AND Growth_Type='TTM'",
                        (d, tk)
                    )
        conn.commit()

# ─── Daily logging (unchanged except filter is in _growth) ─
def _log_today(y):
    today = datetime.today().strftime("%Y-%m-%d")
    with sqlite3.connect(DB_PATH) as conn:
        _ensure_tables(conn); cur = conn.cursor()
        cur.execute("INSERT OR REPLACE INTO Treasury_Yield_History VALUES (?,?)",
                    (today, y))
        for tk in IDXES:
            ttm_pe, fwd_pe = _fetch_pe(tk)
            ttm_g, fwd_g = _growth(ttm_pe, y), _growth(fwd_pe, y)

            if ttm_g is not None:
                cur.execute("INSERT OR REPLACE INTO Index_Growth_History VALUES (?,?, 'TTM', ?)",
                            (today, tk, ttm_g))
            if fwd_g is not None:
                cur.execute("INSERT OR REPLACE INTO Index_Growth_History VALUES (?,?, 'Forward', ?)",
                            (today, tk, fwd_g))

            if ttm_pe is not None:
                cur.execute("INSERT OR REPLACE INTO Index_PE_History VALUES (?,?, 'TTM', ?)",
                            (today, tk, ttm_pe))
            if fwd_pe is not None:
                cur.execute("INSERT OR REPLACE INTO Index_PE_History VALUES (?,?, 'Forward', ?)",
                            (today, tk, fwd_pe))
        conn.commit()

# ─── Summary, plotting & overview (all unchanged) ────────
# ... [rest of file identical to your posted version] ...

# ─── Public entry point ──────────────────────────────────
def index_growth(treasury_yield: float | None = None) -> str:
    _repair_last_week()                       # ← new auto-patch
    y = _resolve_yield(treasury_yield)
    print(f"[index_growth] Using 10-yr yield = {y}")
    _log_today(y)
    _refresh_assets()
    return _overview()

if __name__ == "__main__":
    html = index_growth()                     # uses last DB yield if none passed
    print("Assets built for:", ", ".join(IDXES))

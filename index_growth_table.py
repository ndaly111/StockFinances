#!/usr/bin/env python3
# index_growth_table.py  –  original yield path, updated growth formula
# --------------------------------------------------------------------
# • Uses (((PE/10)**0.1) + yield – 1) for implied growth
# • Yield is taken AS-IS (no decimal/percent conversion)
# • Falls back to last stored P/E when yfinance returns None
# • Generates charts + blue-framed tables under the same filenames
# --------------------------------------------------------------------

import os, sqlite3, numpy as np, pandas as pd
from datetime import datetime
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, FuncFormatter

DB_PATH, CHART_DIR = "Stock Data.db", "charts"
IDXES = ["SPY", "QQQ"]
os.makedirs(CHART_DIR, exist_ok=True)

# ─── CSS for summary tables ──────────────────────────────
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
        v=float(v)
        if v<=30: return "color:#008800;font-weight:bold"
        if v>=70: return "color:#CC0000;font-weight:bold"
    except: pass
    return ""
def _build_html(df):
    return (df.style.hide(axis="index")
                 .map(_pct_color, subset="%ctile")
                 .set_table_attributes('class="summary-table"')
            ).to_html()

# ─── Ensure DB tables exist ──────────────────────────────
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

# ─── Yield helpers (no conversion) ───────────────────────
def _latest_yield():
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT TenYr FROM Treasury_Yield_History "
            "ORDER BY Date DESC LIMIT 1").fetchone()
    return row[0] if row else None

def _resolve_yield(passed):
    if passed is not None:
        return float(passed)          # use exactly what caller passed
    y = _latest_yield()
    if y is None:
        raise RuntimeError("No treasury_yield supplied and DB is empty.")
    return float(y)                   # use stored value as-is

# ─── P/E helpers with fallback ───────────────────────────
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

# ─── Growth calculation (your formula) ───────────────────
def _growth(pe, y):
    if pe is None or pd.isna(pe): return None
    try:
        return (pe / 10) ** 0.1 + y - 1
    except (ValueError, ZeroDivisionError):
        return None

# ─── Log today’s data ────────────────────────────────────
def _log_today(y):
    today=datetime.today().strftime("%Y-%m-%d")
    with sqlite3.connect(DB_PATH) as conn:
        _ensure_tables(conn); cur=conn.cursor()
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

# ─── Table helpers ───────────────────────────────────────
def _percentile(s, v):
    s = pd.to_numeric(s, errors="coerce").dropna().sort_values()
    if s.empty or v is None or np.isnan(v): return None
    rank = np.searchsorted(s.values, float(v), side="right")
    return max(1, min(99, int(round(rank / len(s) * 100))))

def _row(label, s, pct_format=False):
    if s.empty:
        return {}
    r = dict(
        Metric=label,
        Latest=s.iloc[-1],
        Avg=s.mean(),
        Med=s.median(),
        Min=s.min(),
        Max=s.max(),
        **{"%ctile": _percentile(s, s.iloc[-1])}
    )
    for k in ("Latest", "Avg", "Med", "Min", "Max"):
        r[k] = f"{r[k]*100:.2f} %" if pct_format else f"{r[k]:.2f}"
    return r

def _write_summary(stats, path):
    if not stats:
        open(path, "w").write("<p>No data yet.</p>")
        return
    html = SUMMARY_CSS + _build_html(pd.DataFrame([stats]))
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)

# ─── Chart & pivot helpers ───────────────────────────────
def _plot(df, title, formatter, out_file):
    if df is None or df.empty:
        plt.figure(figsize=(0.01, 0.01)); plt.axis("off")
        plt.savefig(out_file, transparent=True, dpi=10); plt.close(); return
    fig, ax = plt.subplots(figsize=(10, 6))
    for col in df.columns:
        ax.plot(df.index, df[col], label=col, linewidth=2)
    ax.set_title(title); ax.grid("--", alpha=.4)
    ax.yaxis.set_major_formatter(formatter)
    ax.legend(); plt.tight_layout(); plt.savefig(out_file); plt.close()

def _pivot(tk, table, typ_col, val_col):
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            f"SELECT Date,{typ_col},{val_col} AS v FROM {table} "
            "WHERE Ticker=? ORDER BY Date ASC", conn, params=(tk,))
    if df.empty: return None
    df["Date"] = pd.to_datetime(df["Date"])
    return df.pivot(index="Date", columns=typ_col, values="v")

# ─── Build per-ticker files ─────────────────────────────
def _build_assets(tk):
    slug = tk.lower()

    gdf = _pivot(tk, "Index_Growth_History", "Growth_Type", "Implied_Growth")
    growth_stats = _row("Implied Growth (TTM)", gdf["TTM"], pct_format=True) \
                   if gdf is not None and "TTM" in gdf.columns else {}
    _write_summary(growth_stats, os.path.join(CHART_DIR, f"{slug}_growth_summary.html"))
    _plot(gdf, f"{tk} Implied Growth", PercentFormatter(1.0),
          os.path.join(CHART_DIR, f"{slug}_growth_chart.png"))

    pdf = _pivot(tk, "Index_PE_History", "PE_Type", "PE_Ratio")
    pe_stats = _row("P/E Ratio (TTM)", pdf["TTM"]) \
               if pdf is not None and "TTM" in pdf.columns else {}
    _write_summary(pe_stats, os.path.join(CHART_DIR, f"{slug}_pe_summary.html"))
    _plot(pdf, f"{tk} P/E Ratio", FuncFormatter(lambda x, _: f"{x:.0f}"),
          os.path.join(CHART_DIR, f"{slug}_pe_chart.png"))

def _refresh_assets():
    for tk in IDXES:
        _build_assets(tk)

# ─── Public entry point ──────────────────────────────────
def index_growth(treasury_yield: float | None = None) -> str:
    y = _resolve_yield(treasury_yield)
    print(f"[index_growth] Using 10-yr yield = {y}")
    _log_today(y); _refresh_assets()
    return "Assets refreshed."

if __name__ == "__main__":
    index_growth()     # uses last DB yield if none passed
    print("Assets built for:", ", ".join(IDXES))

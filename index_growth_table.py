#!/usr/bin/env python3
# index_growth_table.py  –  UNIFIED STYLE VERSION  (2025-07-14 rev h)
# ────────────────────────────────────────────────────────────
# Mini-main index_growth(treasury_yield)
#   • Logs SPY & QQQ implied-growth + P/E history
#   • Generates / refreshes matching charts + summary HTML
#   • Returns an overview table with current percentiles
# ────────────────────────────────────────────────────────────

import os, sqlite3, numpy as np, pandas as pd
from datetime import datetime
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, FuncFormatter

# ─── Config ───────────────────────────────────────────────
DB_PATH        = "Stock Data.db"
IDXES          = ["SPY", "QQQ"]
FALLBACK_YIELD = 0.045
CHART_DIR      = "charts"
os.makedirs(CHART_DIR, exist_ok=True)

# ─── Uniform CSS + helpers (same as index_growth_charts.py) ─
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

def _pct_color(v):                   # green ≤30, red ≥70
    try:
        v = float(v)
        if v <= 30:
            return "color:#008800;font-weight:bold"
        if v >= 70:
            return "color:#CC0000;font-weight:bold"
    except Exception:
        pass
    return ""

def _build_html(df: pd.DataFrame) -> str:
    """Return styled HTML identical to index_growth_charts.py."""
    sty = (
        df.style
          .hide(axis="index")
          .map(_pct_color, subset="%ctile", na_action="ignore")
          .set_table_attributes('class="summary-table"')
    )
    return SUMMARY_CSS + sty.to_html()

# ─── Yield normaliser ─────────────────────────────────────
def _norm_yld(v):
    try:
        if v is None:          return FALLBACK_YIELD
        v = float(v)
        if v < 0.5:            return v          # already decimal
        if v < 20:             return v / 100    # percent form
        return v / 1000                         # ^TNX quote
    except Exception:
        return FALLBACK_YIELD

# ─── Fetch P/E ratios via yfinance ────────────────────────
def _fetch_pe(tk):
    info = yf.Ticker(tk).info
    ttm  = info.get("trailingPE")
    fwd  = info.get("forwardPE")
    if fwd is None:
        px, eps = info.get("regularMarketPrice"), info.get("forwardEps")
        if px and eps:
            try:
                fwd = px / eps
            except ZeroDivisionError:
                fwd = None
    return ttm, fwd

# ─── Implied-growth calc ──────────────────────────────────
def _growth(ttm_pe, fwd_pe, y):
    return (
        y * ttm_pe - 1 if ttm_pe else None,
        y * fwd_pe - 1 if fwd_pe else None
    )

# ─── DB helpers ───────────────────────────────────────────
def _ensure_tables(conn):
    conn.executescript("""
      CREATE TABLE IF NOT EXISTS Index_Growth_History (
        Date TEXT, Ticker TEXT, Growth_Type TEXT, Implied_Growth REAL,
        PRIMARY KEY (Date,Ticker,Growth_Type)
      );
      CREATE TABLE IF NOT EXISTS Index_PE_History (
        Date TEXT, Ticker TEXT, PE_Type TEXT, PE_Ratio REAL,
        PRIMARY KEY (Date,Ticker,PE_Type)
      );
    """)

def _log_today(y):
    today = datetime.today().strftime("%Y-%m-%d")
    with sqlite3.connect(DB_PATH) as conn:
        _ensure_tables(conn)
        cur = conn.cursor()
        for tk in IDXES:
            ttm_pe, fwd_pe = _fetch_pe(tk)
            ttm_g, fwd_g   = _growth(ttm_pe, fwd_pe, y)

            if ttm_g is not None:
                cur.execute(
                    "INSERT OR REPLACE INTO Index_Growth_History VALUES (?,?, 'TTM', ?)",
                    (today, tk, ttm_g),
                )
            if fwd_g is not None:
                cur.execute(
                    "INSERT OR REPLACE INTO Index_Growth_History VALUES (?,?, 'Forward', ?)",
                    (today, tk, fwd_g),
                )

            if ttm_pe is not None:
                cur.execute(
                    "INSERT OR REPLACE INTO Index_PE_History VALUES (?,?, 'TTM', ?)",
                    (today, tk, ttm_pe),
                )
            if fwd_pe is not None:
                cur.execute(
                    "INSERT OR REPLACE INTO Index_PE_History VALUES (?,?, 'Forward', ?)",
                    (today, tk, fwd_pe),
                )
        conn.commit()

# ─── Helper: percentile (SciPy-free) ──────────────────────
def _percentile(series, value):
    s = pd.to_numeric(series, errors="coerce").dropna().sort_values()
    if s.empty or value is None or np.isnan(value):
        return None
    rank = np.searchsorted(s.values, float(value), side="right")
    pct  = (rank / len(s)) * 100
    return max(1, min(99, int(round(pct))))

# ─── Convenience: latest recorded values ──────────────────
def _latest_ttm_growth(tk):
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            """
            SELECT Implied_Growth FROM Index_Growth_History
            WHERE Ticker=? AND Growth_Type='TTM'
            ORDER BY Date DESC LIMIT 1
            """,
            (tk,),
        ).fetchone()
    return row[0] if row else None

def _latest_ttm_pe(tk):
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            """
            SELECT PE_Ratio FROM Index_PE_History
            WHERE Ticker=? AND PE_Type='TTM'
            ORDER BY Date DESC LIMIT 1
            """,
            (tk,),
        ).fetchone()
    return row[0] if row else None

def _history_series(conn, table, tk, col, where):
    q = f"SELECT {col} FROM {table} WHERE Ticker=? AND {where}"
    return pd.read_sql_query(q, conn, params=(tk,))[col].dropna()

# ─── Overview (homepage) table ────────────────────────────
def _overview():
    with sqlite3.connect(DB_PATH) as conn:
        rows = []
        for tk in IDXES:
            ttm_pe = _latest_ttm_pe(tk)
            growth = _latest_ttm_growth(tk)

            pe_hist = _history_series(
                conn, "Index_PE_History", tk, "PE_Ratio", "PE_Type='TTM'"
            )
            gr_hist = _history_series(
                conn, "Index_Growth_History", tk, "Implied_Growth", "Growth_Type='TTM'"
            )

            pe_pct = _percentile(pe_hist, ttm_pe)
            gr_pct = _percentile(gr_hist, growth)

            link   = f'<a href="{tk.lower()}_growth.html">{tk}</a>'
            fmtpct = lambda p: f"{p}<sup>th</sup>" if p is not None else "–"

            if ttm_pe is None or growth is None:
                rows.append(
                    f"<tr><td>{tk}</td><td colspan='4'>No data yet.</td></tr>"
                )
            else:
                rows.append(
                    "<tr>"
                    f"<td>{link}</td><td>{ttm_pe:.1f}</td><td>{growth:.1%}</td>"
                    f"<td>{fmtpct(pe_pct)}</td><td>{fmtpct(gr_pct)}</td></tr>"
                )

    return (
        "<table border='1' style='border-collapse:collapse;'>"
        "<thead><tr><th>Ticker</th><th>P/E</th><th>Implied Growth</th>"
        "<th>P/E percentile</th><th>Implied Growth Percentile</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table>"
    )

# ─── Chart helpers ────────────────────────────────────────
def _pivot(tk, tbl, typ_col):
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            f"""
            SELECT Date, {typ_col},
                   {'Implied_Growth' if 'Growth' in tbl else 'PE_Ratio'} AS v
            FROM {tbl} WHERE Ticker=? ORDER BY Date ASC
            """,
            conn,
            params=(tk,),
        )
    if df.empty:
        return None
    df["Date"] = pd.to_datetime(df["Date"])
    return df.pivot(index="Date", columns=typ_col, values="v")

# ─── Styled-summary builder ───────────────────────────────
def _summary(df):
    """
    Return dict {col: metrics …} including Latest & %ctile so the table
    can exactly match the SPY layout.
    """
    if df is None or df.empty:
        return {}

    stats = {}
    for col in df.columns:
        s = df[col].dropna()
        if s.empty:
            continue
        latest = s.iloc[-1]
        stats[col] = {
            "Latest": latest,
            "Avg":    s.mean(),
            "Med":    s.median(),
            "Min":    s.min(),
            "Max":    s.max(),
            "%ctile": _percentile(s, latest) or "—",
        }
    return stats

def _write_html(stats, path):
    """
    Convert *stats* to the same single-row table used by index_growth_charts.py
    (blue frame, grey grid, percentile colour).
    """
    if not stats:
        open(path, "w").write("<p>No data yet.</p>")
        return

    # take the TTM column if present, otherwise first available
    ttm_stats = stats.get("TTM") or next(iter(stats.values()))

    is_growth = "growth" in path.lower()
    label     = "Implied Growth (TTM)" if is_growth else "P/E Ratio (TTM)"

    def _fmt(v):
        if isinstance(v, str):
            return v
        return f"{v:.2%}" if is_growth else f"{v:.2f}"

    row = {
        "Metric": label,
        "Latest": _fmt(ttm_stats["Latest"]),
        "Avg":    _fmt(ttm_stats["Avg"]),
        "Med":    _fmt(ttm_stats["Med"]),
        "Min":    _fmt(ttm_stats["Min"]),
        "Max":    _fmt(ttm_stats["Max"]),
        "%ctile": ttm_stats["%ctile"],
    }

    html = _build_html(pd.DataFrame([row]))
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)

# ─── Plotting helper ─────────────────────────────────────
def _plot(df, out, title, fmt):
    if df is None or df.empty:
        plt.figure(figsize=(0.01, 0.01)); plt.axis("off")
        plt.savefig(out, transparent=True, dpi=10); plt.close(); return
    fig, ax = plt.subplots(figsize=(10, 6))
    for col in df.columns:
        ax.plot(df.index, df[col], label=col, linewidth=2)
    ax.set_title(title); ax.grid("--", alpha=.4)
    ax.yaxis.set_major_formatter(fmt); ax.legend()
    plt.tight_layout(); plt.savefig(out); plt.close()

# ─── Asset builder per-index ─────────────────────────────
def _build_assets(tk):
    # ― Implied Growth ―
    gdf = _pivot(tk, "Index_Growth_History", "Growth_Type")
    gs  = _summary(gdf)
    _write_html(gs, os.path.join(CHART_DIR, f"{tk.lower()}_growth_summary.html"))
    _plot(
        gdf,
        os.path.join(CHART_DIR, f"{tk.lower()}_growth_chart.png"),
        f"{tk} Implied Growth",
        PercentFormatter(1.0),
    )

    # ― P/E Ratio ―
    pdf = _pivot(tk, "Index_PE_History", "PE_Type")
    ps  = _summary(pdf)
    _write_html(ps, os.path.join(CHART_DIR, f"{tk.lower()}_pe_summary.html"))
    _plot(
        pdf,
        os.path.join(CHART_DIR, f"{tk.lower()}_pe_chart.png"),
        f"{tk} P/E Ratio",
        FuncFormatter(lambda x, _: f"{x:.0f}"),
    )

def _refresh_assets():
    for tk in IDXES:
        _build_assets(tk)

# ─── Mini-main (importable) ──────────────────────────────
def index_growth(treasury_yield: float | None = None) -> str:
    y = _norm_yld(treasury_yield)
    print(f"[index_growth] Using 10-yr yield = {y:.4f}")
    _log_today(y)
    _refresh_assets()
    return _overview()

# ─── Stand-alone test ───────────────────────────────────
if __name__ == "__main__":
    html = index_growth()
    open(os.path.join(CHART_DIR, "spy_qqq_overview.html"), "w").write(html)
    print("Wrote spy_qqq_overview.html")

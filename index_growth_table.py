#!/usr/bin/env python3
# index_growth_table.py  –  EPS-and-Yield version  (2025-08-01 rev j)
# ────────────────────────────────────────────────────────────────
# Mini-main  index_growth(treasury_yield)
#   • Logs SPY & QQQ implied-growth, P/E, EPS and 10-yr yield
#   • Generates / refreshes three charts per index
#       1) Implied-Growth    2) P/E ratio    3) EPS vs Ten-yr-yield
#   • Writes styled summary HTML identical to index_growth_charts.py
#   • Returns a simple overview table for the home page
# ────────────────────────────────────────────────────────────────

import os, sqlite3, numpy as np, pandas as pd
from datetime import datetime
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, FuncFormatter

# ─── Config ───────────────────────────────────────────────
DB_PATH        = "Stock Data.db"
IDXES          = ["SPY", "QQQ"]
FALLBACK_YIELD = 0.045          # 4.5 %
CHART_DIR      = "charts"
os.makedirs(CHART_DIR, exist_ok=True)

# ─── Uniform CSS + helpers (exact copy from index_growth_charts.py) ─
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
        if v <= 30:
            return "color:#008800;font-weight:bold"
        if v >= 70:
            return "color:#CC0000;font-weight:bold"
    except Exception:
        pass
    return ""

def _build_html(df: pd.DataFrame) -> str:
    sty = (
        df.style
          .hide(axis="index")
          .map(lambda v: _pct_color(v), subset="%ctile")
          .set_table_attributes('class="summary-table"')
    )
    return SUMMARY_CSS + sty.to_html()

# ─── Yield normaliser ─────────────────────────────────────
def _norm_yld(v):
    try:
        if v is None:
            return FALLBACK_YIELD
        v = float(v)
        if v < 0.5:   # already decimal (0.0423)
            return v
        if v < 20:    # quoted as “4.23”
            return v / 100
        return v / 1000  # ^TNX = 42.3
    except Exception:
        return FALLBACK_YIELD

# ─── yfinance helpers ────────────────────────────────────
def _fetch_pe_eps(tk):
    info = yf.Ticker(tk).info
    ttm_pe  = info.get("trailingPE")
    fwd_pe  = info.get("forwardPE")
    ttm_eps = info.get("trailingEps")

    if fwd_pe is None:
        px, eps_fwd = info.get("regularMarketPrice"), info.get("forwardEps")
        if px and eps_fwd:
            try:
                fwd_pe = px / eps_fwd
            except ZeroDivisionError:
                fwd_pe = None
    return ttm_pe, fwd_pe, ttm_eps

# ─── Implied-growth calc ─────────────────────────────────
def _growth(ttm_pe, fwd_pe, y):
    return (
        y * ttm_pe - 1 if ttm_pe else None,
        y * fwd_pe - 1 if fwd_pe else None,
    )

# ─── DB helpers ──────────────────────────────────────────
def _ensure_tables(conn):
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS Index_Growth_History (
          Date TEXT, Ticker TEXT, Growth_Type TEXT, Implied_Growth REAL,
          PRIMARY KEY (Date,Ticker,Growth_Type)
        );
        CREATE TABLE IF NOT EXISTS Index_PE_History (
          Date TEXT, Ticker TEXT, PE_Type TEXT, PE_Ratio REAL,
          PRIMARY KEY (Date,Ticker,PE_Type)
        );
        CREATE TABLE IF NOT EXISTS Index_EPS_History (
          Date TEXT, Ticker TEXT, EPS_Type TEXT, EPS REAL,
          PRIMARY KEY (Date,Ticker,EPS_Type)
        );
        CREATE TABLE IF NOT EXISTS Treasury_Yield_History (
          Date TEXT PRIMARY KEY, TenYr REAL
        );
        """
    )

def _log_today(y):
    today = datetime.today().strftime("%Y-%m-%d")
    with sqlite3.connect(DB_PATH) as conn:
        _ensure_tables(conn)
        cur = conn.cursor()

        # one row per-day for 10-yr yield
        cur.execute(
            "INSERT OR REPLACE INTO Treasury_Yield_History VALUES (?,?)",
            (today, y),
        )

        for tk in IDXES:
            ttm_pe, fwd_pe, ttm_eps = _fetch_pe_eps(tk)
            ttm_g, fwd_g = _growth(ttm_pe, fwd_pe, y)

            # — Implied growth —
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

            # — P/E —
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

            # — EPS —
            if ttm_eps is not None:
                cur.execute(
                    "INSERT OR REPLACE INTO Index_EPS_History VALUES (?,?, 'TTM', ?)",
                    (today, tk, ttm_eps),
                )
        conn.commit()

# ─── Percentile helper ───────────────────────────────────
def _percentile(series, value):
    s = pd.to_numeric(series, errors="coerce").dropna().sort_values()
    if s.empty or value is None or np.isnan(value):
        return None
    rank = np.searchsorted(s.values, float(value), side="right")
    pct  = (rank / len(s)) * 100
    return max(1, min(99, int(round(pct))))

# ─── Convenience: latest recorded values ─────────────────
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

# ─── Overview table (unchanged visual) ───────────────────
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

# ─── Series fetchers for charts ──────────────────────────
def _pivot(tk, tbl, typ_col, val_col):
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            f"""
            SELECT Date, {typ_col}, {val_col} AS v
            FROM {tbl} WHERE Ticker=? ORDER BY Date ASC
            """,
            conn,
            params=(tk,),
        )
    if df.empty:
        return None
    df["Date"] = pd.to_datetime(df["Date"])
    return df.pivot(index="Date", columns=typ_col, values="v")

def _yield_series():
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            "SELECT Date, TenYr FROM Treasury_Yield_History ORDER BY Date ASC",
            conn,
        )
    if df.empty:
        return None
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date")["TenYr"]

# ─── Styled summary builder (unchanged) ──────────────────
def _summary(df):
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
    if not stats:
        open(path, "w").write("<p>No data yet.</p>")
        return

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

# ─── Plotting helpers ────────────────────────────────────
def _plot(df, out, title, fmt):
    if df is None or df.empty:
        plt.figure(figsize=(0.01, 0.01))
        plt.axis("off")
        plt.savefig(out, transparent=True, dpi=10)
        plt.close()
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for col in df.columns:
        ax.plot(df.index, df[col], label=col, linewidth=2)
    ax.set_title(title)
    ax.grid("--", alpha=0.4)
    ax.yaxis.set_major_formatter(fmt)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

def _plot_eps_yield(tk, eps_s, yld_s):
    out = os.path.join(CHART_DIR, f"{tk.lower()}_eps_yield.png")
    if eps_s is None or eps_s.empty or yld_s is None or yld_s.empty:
        plt.figure(figsize=(0.01, 0.01))
        plt.axis("off")
        plt.savefig(out, transparent=True, dpi=10)
        plt.close()
        return

    # Align indices (inner join)
    df = pd.concat({"EPS": eps_s, "Yield": yld_s * 100}, axis=1).dropna()
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(df.index, df["EPS"], linewidth=2, label="EPS (TTM)")
    ax1.set_ylabel("EPS (dollars)")
    ax1.grid("--", alpha=0.4)

    ax2 = ax1.twinx()
    ax2.plot(df.index, df["Yield"], linewidth=2, linestyle="--", label="10-yr Yield")
    ax2.set_ylabel("Yield (%)")

    ax1.set_title(f"{tk} – EPS (left) vs 10-yr Treasury yield (right)")
    fig.tight_layout()
    # one combined legend
    lines, labels = ax1.get_legend_handles_labels() + ax2.get_legend_handles_labels()
    fig.legend(lines, labels, loc="upper left")
    plt.savefig(out)
    plt.close()

# ─── Asset builder per index ─────────────────────────────
def _build_assets(tk):
    # 1) Implied Growth
    gdf = _pivot(tk, "Index_Growth_History", "Growth_Type", "Implied_Growth")
    gs  = _summary(gdf)
    _write_html(gs, os.path.join(CHART_DIR, f"{tk.lower()}_growth_summary.html"))
    _plot(
        gdf,
        os.path.join(CHART_DIR, f"{tk.lower()}_growth_chart.png"),
        f"{tk} Implied Growth",
        PercentFormatter(1.0),
    )

    # 2) P/E Ratio
    pdf = _pivot(tk, "Index_PE_History", "PE_Type", "PE_Ratio")
    ps  = _summary(pdf)
    _write_html(ps, os.path.join(CHART_DIR, f"{tk.lower()}_pe_summary.html"))
    _plot(
        pdf,
        os.path.join(CHART_DIR, f"{tk.lower()}_pe_chart.png"),
        f"{tk} P/E Ratio",
        FuncFormatter(lambda x, _: f"{x:.0f}"),
    )

    # 3) EPS vs 10-yr Yield
    with sqlite3.connect(DB_PATH) as conn:
        eps_s = (
            pd.read_sql_query(
                """
                SELECT Date, EPS FROM Index_EPS_History
                WHERE Ticker=? AND EPS_Type='TTM' ORDER BY Date ASC
                """,
                conn,
                params=(tk,),
            )
            .assign(Date=lambda d: pd.to_datetime(d["Date"]))
            .set_index("Date")["EPS"]
            .dropna()
        )
    yld_s = _yield_series()
    _plot_eps_yield(tk, eps_s, yld_s)

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

# ─── Stand-alone test ────────────────────────────────────
if __name__ == "__main__":
    html = index_growth()  # uses FALLBACK_YIELD
    open(os.path.join(CHART_DIR, "spy_qqq_overview.html"), "w").write(html)
    print("Wrote spy_qqq_overview.html")

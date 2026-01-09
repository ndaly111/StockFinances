#!/usr/bin/env python3
# index_growth_table.py  –  FULL FILE  (v2025-08-06 patch)
# ───────────────────────────────────────────────────────────
# Mini-main index_growth(treasury_yield)
#  • Logs SPY & QQQ implied growth + P/E ratios
#  • Generates/refreshes charts + summary HTML
#  • Returns overview table with P/E & Implied-Growth percentiles
# ───────────────────────────────────────────────────────────

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
EPS_TYPE_IMPLIED = "IMPLIED_FROM_PE"
os.makedirs(CHART_DIR, exist_ok=True)

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

# ─── Fetch P/E ratios ─────────────────────────────────────
def _fetch_pe(tk):
    info = yf.Ticker(tk).info
    ttm  = info.get("trailingPE")
    fwd  = info.get("forwardPE")
    if fwd is None:
        px, eps = info.get("regularMarketPrice"), info.get("forwardEps")
        if px and eps:
            try: fwd = px / eps
            except ZeroDivisionError: fwd = None
    return ttm, fwd

# ─── Growth calc ──────────────────────────────────────────
def _growth(ttm_pe, fwd_pe, y):
    def solve(pe):
        if pe is None:
            return None
        try:
            pe = float(pe)
        except (TypeError, ValueError):
            return None
        if pe <= 0:
            return None

        try:
            # P/E = (((growth - y) + 1) ** 10) * 10
            # ⇒ growth = y - 1 + (P/E / 10) ** (1 / 10)
            return y - 1 + pow(pe / 10, 0.1)
        except (OverflowError, ValueError):
            return None

    return (solve(ttm_pe), solve(fwd_pe))

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
      CREATE TABLE IF NOT EXISTS Index_EPS_History (
        Date TEXT, Ticker TEXT, EPS_Type TEXT, EPS REAL,
        PRIMARY KEY (Date,Ticker,EPS_Type)
      );
      CREATE INDEX IF NOT EXISTS idx_Index_EPS_History_ticker_type_date
          ON Index_EPS_History (Ticker, EPS_Type, Date);
    """)

def _log_today(y):
    today = datetime.today().strftime("%Y-%m-%d")
    with sqlite3.connect(DB_PATH) as conn:
        _ensure_tables(conn); cur = conn.cursor()
        for tk in IDXES:
            ttm_pe, fwd_pe = _fetch_pe(tk)
            price = None
            try:
                price = yf.Ticker(tk).info.get("regularMarketPrice")
            except Exception:
                price = None
            if price is None:
                try:
                    hist = yf.Ticker(tk).history(period="5d", auto_adjust=False, actions=False)
                    if not hist.empty:
                        price = float(hist["Close"].iloc[-1])
                except Exception:
                    price = None
            ttm_g, fwd_g   = _growth(ttm_pe, fwd_pe, y)

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

            if price is not None and ttm_pe is not None:
                try:
                    price_f = float(price)
                    ttm_pe_f = float(ttm_pe)
                    if price_f > 0 and ttm_pe_f > 0:
                        eps = price_f / ttm_pe_f
                        cur.execute(
                            "INSERT OR REPLACE INTO Index_EPS_History VALUES (?,?, ?, ?)",
                            (today, tk, EPS_TYPE_IMPLIED, eps),
                        )
                except Exception:
                    pass
        conn.commit()

# ─── Helper: percentile without SciPy ─────────────────────
def _percentile(series, value):
    """Return percentile rank (1-99) for *value* within *series*."""
    if value is None or len(series) < 2:
        return None
    pct = (np.sum(series <= value) / len(series)) * 100
    return max(1, min(99, int(round(pct))))

# ─── Convenience: latest TTM implied growth ──────────────
def _latest_ttm_growth(tk):
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("""
            SELECT Implied_Growth FROM Index_Growth_History
            WHERE Ticker=? AND Growth_Type='TTM'
            ORDER BY Date DESC LIMIT 1
        """, (tk,)).fetchone()
    return row[0] if row else None

def _history_series(conn, table, tk, col, where):
    q = f"SELECT {col} FROM {table} WHERE Ticker=? AND {where}"
    return pd.read_sql_query(q, conn, params=(tk,))[col].dropna()

# ─── Build overview table ─────────────────────────────────
def _overview():
    with sqlite3.connect(DB_PATH) as conn:
        rows = []
        for tk in IDXES:
            ttm_pe, _ = _fetch_pe(tk)
            growth    = _latest_ttm_growth(tk)

            pe_hist = _history_series(conn, "Index_PE_History", tk,
                                      "PE_Ratio", "PE_Type='TTM'")
            gr_hist = _history_series(conn, "Index_Growth_History", tk,
                                      "Implied_Growth", "Growth_Type='TTM'")

            pe_pct = _percentile(pe_hist, ttm_pe)
            gr_pct = _percentile(gr_hist, growth)

            link   = f'<a href="{tk.lower()}_growth.html">{tk}</a>'
            fmtpct = lambda p: f"{p}<sup>th</sup>" if p is not None else "–"

            if ttm_pe is None or growth is None:
                rows.append(f"<tr><td>{tk}</td><td colspan='4'>No data yet.</td></tr>")
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

# ─── Chart builders (unchanged) ───────────────────────────
def _pivot(tk, tbl, typ_col):
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            f"SELECT Date,{typ_col},"
            f"{'Implied_Growth' if 'Growth' in tbl else 'PE_Ratio'} AS v "
            f"FROM {tbl} WHERE Ticker=? ORDER BY Date ASC",
            conn, params=(tk,)
        )
    if df.empty:
        return None
    df["Date"] = pd.to_datetime(df["Date"])
    return df.pivot(index="Date", columns=typ_col, values="v")

def _summary(df):
    return {c: {"Avg": df[c].mean(), "Med": df[c].median(),
                "Min": df[c].min(), "Max": df[c].max()}
            for c in df.columns} if df is not None else {}

def _write_html(stats, path, link_tk, tk):
    if not stats:
        open(path, "w").write("<p>No data yet.</p>")
        return
    rows = []
    link = f'<a href="{link_tk}">{tk}</a>'
    for typ, d in stats.items():
        for k, v in d.items():
            rows.append({
                "Ticker": link, "Type": typ, "Stat": k,
                "Value": f"{v:.2%}" if 'growth' in path else f"{v:.1f}"
            })
    pd.DataFrame(rows).to_html(path, index=False, escape=False)

def _plot(df, stats, out, title, fmt):
    if df is None or df.empty:
        plt.figure(figsize=(0.01, 0.01)); plt.axis("off")
        plt.savefig(out, transparent=True, dpi=10); plt.close(); return
    fig, ax = plt.subplots(figsize=(10, 6))
    for col in df.columns:
        ax.plot(df.index, df[col], label=col, linewidth=2)
    ax.set_title(title); ax.grid("--", alpha=.4)
    ax.yaxis.set_major_formatter(fmt); ax.legend()
    plt.tight_layout(); plt.savefig(out); plt.close()

def _build_assets(tk):
    gdf = _pivot(tk, "Index_Growth_History", "Growth_Type")
    gs  = _summary(gdf)
    _write_html(gs, os.path.join(CHART_DIR, f"{tk.lower()}_growth_summary.html"),
                f"{tk.lower()}_growth.html", tk)
    _plot(gdf, gs, os.path.join(CHART_DIR, f"{tk.lower()}_growth_chart.png"),
          f"{tk} Implied Growth", PercentFormatter(1.0))

    pdf = _pivot(tk, "Index_PE_History", "PE_Type")
    ps  = _summary(pdf)
    _write_html(ps, os.path.join(CHART_DIR, f"{tk.lower()}_pe_summary.html"),
                f"{tk.lower()}_pe.html", tk)
    _plot(pdf, ps, os.path.join(CHART_DIR, f"{tk.lower()}_pe_chart.png"),
          f"{tk} P/E Ratio", FuncFormatter(lambda x, _: f"{x:.0f}"))

def _refresh_assets():
    for tk in IDXES:
        _build_assets(tk)

# ─── Mini-main (called by main_remote.py) ────────────────
def index_growth(treasury_yield: float | None = None) -> str:
    y = _norm_yld(treasury_yield)
    print(f"[index_growth] Using 10-yr yield = {y:.4f}")
    _log_today(y); _refresh_assets()
    return _overview()

# ─── Stand-alone test ────────────────────────────────────
if __name__ == "__main__":
    html = _overview()
    open(os.path.join(CHART_DIR, "spy_qqq_overview.html"), "w").write(html)
    print("Wrote spy_qqq_overview.html")

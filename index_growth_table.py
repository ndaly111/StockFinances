# index_growth_table.py
# ───────────────────────────────────────────────────────────
# Mini-main index_growth(treasury_yield) – imported by main_remote.py
#
# • Logs today’s implied growth + P/E ratios for SPY & QQQ
# • Generates / refreshes charts + summary HTML for both series
# • Returns an overview table with percentiles for the dashboard
# ───────────────────────────────────────────────────────────

import os, sqlite3
from datetime import datetime
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, FuncFormatter
import scipy.stats as st            # ← NEW: used for percentile calc

# ─── Configuration ─────────────────────────────────────────
DB_PATH        = "Stock Data.db"
IDXES          = ["SPY", "QQQ"]
FALLBACK_YIELD = 0.045
CHART_DIR      = "charts"
os.makedirs(CHART_DIR, exist_ok=True)

# ─── Yield normaliser ─────────────────────────────────────
def _norm_yld(v):
    if v is None: return FALLBACK_YIELD
    try:
        v = float(v)
        if v < 0.5:  return v
        if v < 20:   return v / 100
        return v / 1000
    except Exception:
        return FALLBACK_YIELD

# ─── Fetch P/E ratios ─────────────────────────────────────
def _fetch_pe(tk):
    info = yf.Ticker(tk).info
    ttm  = info.get("trailingPE")
    fwd  = info.get("forwardPE")
    if fwd is None:
        price, eps = info.get("regularMarketPrice"), info.get("forwardEps")
        if price and eps:            # fallback
            try: fwd = price / eps
            except ZeroDivisionError: fwd = None
    return ttm, fwd

# ─── Growth maths ─────────────────────────────────────────
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
        _ensure_tables(conn); cur = conn.cursor()
        for tk in IDXES:
            ttm_pe, fwd_pe = _fetch_pe(tk)
            ttm_g , fwd_g  = _growth(ttm_pe, fwd_pe, y)

            # growth rows
            if ttm_g is not None:
                cur.execute("INSERT OR REPLACE INTO Index_Growth_History VALUES (?,?, 'TTM', ?)",
                            (today, tk, ttm_g))
            if fwd_g is not None:
                cur.execute("INSERT OR REPLACE INTO Index_Growth_History VALUES (?,?, 'Forward', ?)",
                            (today, tk, fwd_g))

            # P/E rows
            if ttm_pe is not None:
                cur.execute("INSERT OR REPLACE INTO Index_PE_History VALUES (?,?, 'TTM', ?)",
                            (today, tk, ttm_pe))
            if fwd_pe is not None:
                cur.execute("INSERT OR REPLACE INTO Index_PE_History VALUES (?,?, 'Forward', ?)",
                            (today, tk, fwd_pe))
        conn.commit()

# ─── Helpers for latest values & percentiles ──────────────
def _latest_ttm_growth(tk):
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("""
            SELECT Implied_Growth FROM Index_Growth_History
            WHERE Ticker=? AND Growth_Type='TTM'
            ORDER BY Date DESC LIMIT 1
        """,(tk,)).fetchone()
    return row[0] if row else None

def _series(conn, table, tk, col, where):
    q = f"SELECT {col} FROM {table} WHERE Ticker=? AND {where}"
    return pd.read_sql_query(q, conn, params=(tk,))[col].dropna()

def _percentile(series, value):
    if len(series) < 2 or value is None:
        return None
    return int(round(st.percentileofscore(series, value, kind="weak")))

# ─── Overview table with percentiles ──────────────────────
def _overview():
    rows=[]
    with sqlite3.connect(DB_PATH) as conn:
        for tk in IDXES:
            ttm_pe, _ = _fetch_pe(tk)
            growth    = _latest_ttm_growth(tk)

            pe_hist = _series(conn, "Index_PE_History", tk, "PE_Ratio", "PE_Type='TTM'")
            gr_hist = _series(conn, "Index_Growth_History", tk, "Implied_Growth", "Growth_Type='TTM'")

            pe_pct = _percentile(pe_hist, ttm_pe)
            gr_pct = _percentile(gr_hist, growth)

            if ttm_pe is None or growth is None:
                rows.append(
                    f"<tr><td>{tk}</td><td colspan='4'>No implied-growth data yet.</td></tr>"
                )
            else:
                link = f'<a href="{tk.lower()}_growth.html">{tk}</a>'
                fmt_pct = lambda p: f"{p}<sup>th</sup>" if p is not None else "–"
                rows.append(
                    "<tr>"
                    f"<td>{link}</td>"
                    f"<td>{ttm_pe:.1f}</td>"
                    f"<td>{growth:.1%}</td>"
                    f"<td>{fmt_pct(pe_pct)}</td>"
                    f"<td>{fmt_pct(gr_pct)}</td>"
                    "</tr>"
                )
    return (
        "<table border='1' style='border-collapse:collapse;'>"
        "<thead><tr>"
        "<th>Ticker</th><th>P/E Ratio</th><th>Implied Growth</th>"
        "<th>P/E Percentile</th><th>Growth Percentile</th></tr></thead>"
        "<tbody>"+ "".join(rows) +"</tbody></table>"
    )

# ─── Generic chart builders (unchanged logic) ─────────────
def _pivot(tk, tbl, col_label):
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            f"SELECT Date,{col_label},"
            f"{'Implied_Growth' if 'Growth' in tbl else 'PE_Ratio'} AS v "
            f"FROM {tbl} WHERE Ticker=? ORDER BY Date ASC",
            conn, params=(tk,)
        )
    if df.empty: return None
    df["Date"] = pd.to_datetime(df["Date"])
    return df.pivot(index="Date", columns=col_label, values="v")

def _summary(df):
    return {c:{"Avg":df[c].mean(),"Med":df[c].median(),"Min":df[c].min(),"Max":df[c].max()}
            for c in df.columns} if df is not None else {}

def _write_html(stats, path, link_target, tk):
    if not stats:
        open(path,"w",encoding="utf-8").write("<p>No data yet.</p>"); return
    rows=[]
    link=f'<a href="{link_target}">{tk}</a>'
    for typ,d in stats.items():
        for k,v in d.items():
            rows.append({"Ticker":link,"Type":typ,"Stat":k,"Value":f"{v:.2%}" if "growth" in path else f"{v:.1f}"})
    pd.DataFrame(rows).to_html(path,index=False,escape=False)

def _plot(df, stats, path, title, yfmt):
    if df is None or df.empty:
        plt.figure(figsize=(0.01,0.01)); plt.axis("off")
        plt.savefig(path, transparent=True, dpi=10); plt.close(); return
    fig,ax=plt.subplots(figsize=(10,6))
    for col in df.columns:
        ax.plot(df.index, df[col], label=col, linewidth=2)
    ax.set_title(title); ax.grid("--",alpha=.4)
    ax.yaxis.set_major_formatter(yfmt); ax.legend()
    plt.tight_layout(); plt.savefig(path); plt.close()

def _build_assets(tk):
    # Implied Growth
    gdf = _pivot(tk,"Index_Growth_History","Growth_Type")
    gstats=_summary(gdf)
    _write_html(gstats, os.path.join(CHART_DIR,f"{tk.lower()}_growth_summary.html"),
                f"{tk.lower()}_growth.html", tk)
    _plot(gdf,gstats, os.path.join(CHART_DIR,f"{tk.lower()}_growth_chart.png"),
          f"{tk} Implied Growth", PercentFormatter(1.0))
    # P/E Ratio
    pdf=_pivot(tk,"Index_PE_History","PE_Type")
    pstats=_summary(pdf)
    _write_html(pstats, os.path.join(CHART_DIR,f"{tk.lower()}_pe_summary.html"),
                f"{tk.lower()}_pe.html", tk)
    _plot(pdf,pstats, os.path.join(CHART_DIR,f"{tk.lower()}_pe_chart.png"),
          f"{tk} P/E Ratio", FuncFormatter(lambda x,_:f"{x:.0f}"))

def _refresh_assets():
    for tk in IDXES: _build_assets(tk)

# ─── MINI-MAIN (imported by main_remote.py) ───────────────
def index_growth(treasury_yield: float | None=None) -> str:
    y=_norm_yld(treasury_yield)
    print(f"[index_growth] Using 10-yr yield = {y:.4f}")
    _log_today(y)
    _refresh_assets()
    return _overview()

# ─── Stand-alone test run ─────────────────────────────────
if __name__ == "__main__":
    html=_overview()
    out=os.path.join(CHART_DIR,"spy_qqq_overview.html")
    open(out,"w",encoding="utf-8").write(html)
    print("Wrote", out)

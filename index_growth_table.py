# index_growth_table.py
# ───────────────────────────────────────────────────────────
# Logs SPY & QQQ implied growth   → Index_Growth_History
# Logs SPY & QQQ P/E ratios       → Index_PE_History
# Builds charts + summary tables for both series
# Returns a compact linked overview table for the dashboard.
# ───────────────────────────────────────────────────────────

import os, sqlite3
from datetime import datetime
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, FuncFormatter

# ─── Configuration ─────────────────────────────────────────
DB_PATH        = "Stock Data.db"
IDXES          = ["SPY", "QQQ"]
FALLBACK_YIELD = 0.045
CHART_DIR      = "charts"
os.makedirs(CHART_DIR, exist_ok=True)

# ─── Yield normaliser ─────────────────────────────────────
def _normalize_yield(v):
    if v is None:                    return FALLBACK_YIELD
    try:
        v = float(v)
        if v < 0.5:  return v        # already decimal
        if v < 20:   return v / 100  # percent form
        return v / 1000              # ^TNX quote
    except Exception:
        return FALLBACK_YIELD

# ─── Fetch trailing / forward PEs ──────────────────────────
def _fetch_pe(tk):
    info = yf.Ticker(tk).info
    ttm  = info.get("trailingPE")
    fwd  = info.get("forwardPE")
    if fwd is None:
        price, fwd_eps = info.get("regularMarketPrice"), info.get("forwardEps")
        if price and fwd_eps:
            try:  fwd = price / fwd_eps
            except ZeroDivisionError: fwd = None
    return ttm, fwd

# ─── Compute implied growth ───────────────────────────────
def _growth(ttm_pe, fwd_pe, y):
    ttm_g = y * ttm_pe - 1 if ttm_pe else None
    fwd_g = y * fwd_pe - 1 if fwd_pe else None
    return ttm_g, fwd_g

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

# ─── Latest TTM growth for dashboard ──────────────────────
def _latest_ttm_growth(tk):
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("""
            SELECT Implied_Growth FROM Index_Growth_History
            WHERE Ticker=? AND Growth_Type='TTM'
            ORDER BY Date DESC LIMIT 1
        """, (tk,)).fetchone()
    return row[0] if row else None

# ─── Dashboard overview table ─────────────────────────────
def _overview():
    rows = []
    for tk in IDXES:
        ttm_pe, _ = _fetch_pe(tk)
        g = _latest_ttm_growth(tk)
        if ttm_pe is None or g is None:
            rows.append(f"<tr><td>{tk}</td><td colspan='2'>No implied-growth data yet.</td></tr>")
        else:
            link = f'<a href="{tk.lower()}_growth.html">{tk}</a>'
            rows.append(f"<tr><td>{link}</td><td>{ttm_pe:.1f}</td><td>{g:.1%}</td></tr>")
    return ("<table border='1' style='border-collapse:collapse;'>"
            "<thead><tr><th>Ticker</th><th>P/E Ratio</th><th>Implied Growth</th></tr></thead>"
            "<tbody>"+ "".join(rows) +"</tbody></table>")

# ─── Generic chart builder ────────────────────────────────
def _pivot(tk, tbl, col_label):
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(f"""
            SELECT Date, {col_label}, {'Implied_Growth' if 'Growth' in tbl else 'PE_Ratio'} AS v
            FROM {tbl} WHERE Ticker=? ORDER BY Date ASC
        """, conn, params=(tk,))
    if df.empty: return None
    df["Date"] = pd.to_datetime(df["Date"])
    return df.pivot(index="Date", columns=col_label, values="v")

def _summary(df):
    out={}
    for col in df.columns:
        out[col] = {"Avg":df[col].mean(), "Med":df[col].median(),
                    "Min":df[col].min(),  "Max":df[col].max()}
    return out

def _write_html(stats, path, tk, link_target):
    if not stats:
        open(path,"w",encoding="utf-8").write("<p>No data yet.</p>"); return
    rows=[]
    link=f'<a href="{link_target}">{tk}</a>'
    for typ, d in stats.items():
        for k,v in d.items():
            rows.append({"Ticker":link,"Type":typ,"Stat":k,"Value":f"{v:.2%}" if 'Growth' in link_target else f"{v:.1f}"})
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

# ─── Build assets for each ticker ─────────────────────────
def _build_assets(tk):
    # ------- Implied Growth -------
    gdf  = _pivot(tk,"Index_Growth_History","Growth_Type")
    gsum = _summary(gdf) if gdf is not None else {}
    _write_html(gsum, os.path.join(CHART_DIR,f"{tk.lower()}_growth_summary.html"),
                tk, f"{tk.lower()}_growth.html")
    _plot(gdf,gsum, os.path.join(CHART_DIR,f"{tk.lower()}_growth_chart.png"),
          f"{tk} Implied Growth Rates", PercentFormatter(1.0))

    # ------- P/E Ratio -------
    pdf  = _pivot(tk,"Index_PE_History","PE_Type")
    psum = _summary(pdf) if pdf is not None else {}
    _write_html(psum, os.path.join(CHART_DIR,f"{tk.lower()}_pe_summary.html"),
                tk, f"{tk.lower()}_pe.html")
    _plot(pdf, psum, os.path.join(CHART_DIR,f"{tk.lower()}_pe_chart.png"),
          f"{tk} P/E Ratio History", FuncFormatter(lambda x,_:f"{x:.0f}"))

def _refresh_assets():
    for tk in IDXES:
        _build_assets(tk)

# ─── MINI-MAIN  (imported by main_remote.py) ───────────────
def index_growth(treasury_yield: float | None = None) -> str:
    y = _normalize_yield(treasury_yield)
    print(f"[index_growth] Using 10-yr yield = {y:.4f}")
    _log_today(y)
    _refresh_assets()
    return _overview()

# Stand-alone test
if __name__ == "__main__":
    html=_overview(); open(os.path.join(CHART_DIR,"spy_qqq_overview.html"),"w").write(html)
    print("Wrote spy_qqq_overview.html")

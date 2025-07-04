# index_growth_table.py
# ───────────────────────────────────────────────────────────
# Mini-main  index_growth(treasury_yield)  – imported by main_remote.py
#
# • Logs today’s implied growth for SPY & QQQ
# • Generates / refreshes charts + full summary tables
# • Returns a concise, linked overview table (Ticker | P/E | Implied Growth)
#   to embed on the dashboard home page.
# ───────────────────────────────────────────────────────────

import os, sqlite3
from datetime import datetime
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# ─── Configuration ─────────────────────────────────────────
DB_PATH        = "Stock Data.db"
TABLE_NAME     = "Index_Growth_History"
INDEXES        = ["SPY", "QQQ"]
FALLBACK_YIELD = 0.045            # 4.5 % if caller passes None
CHART_DIR      = "charts"
os.makedirs(CHART_DIR, exist_ok=True)

# ─── Yield normaliser ─────────────────────────────────────
def _normalize_yield(val):
    """
    Accepts:
      0.043     → 0.043
      4.3       → 0.043
      43.0 (^TNX quote) → 0.043
    """
    if val is None:
        return FALLBACK_YIELD
    try:
        v = float(val)
        if v < 0.5:   return v
        if v < 20:    return v / 100
        return v / 1000
    except Exception:
        return FALLBACK_YIELD

# ─── Helpers to fetch PE ratios ────────────────────────────
def _fetch_pe(ticker):
    """
    Returns (trailingPE, forwardPE).  If forwardPE missing,
    compute forwardPE = price / forwardEps if data present.
    """
    info = yf.Ticker(ticker).info
    ttm  = info.get("trailingPE")
    fwd  = info.get("forwardPE")
    if fwd is None:
        price   = info.get("regularMarketPrice")
        fwd_eps = info.get("forwardEps")
        if price and fwd_eps:
            try:    fwd = price / fwd_eps
            except ZeroDivisionError: fwd = None
    return ttm, fwd

# ─── Growth maths ──────────────────────────────────────────
def _compute_growth(ttm_pe, fwd_pe, y):
    """
    Return tuple (ttm_growth, fwd_growth).
    Compute each independently so missing forward PE does NOT
    prevent TTM growth from being recorded.
    """
    ttm_growth = y * ttm_pe - 1 if ttm_pe else None
    fwd_growth = y * fwd_pe - 1 if fwd_pe else None
    return ttm_growth, fwd_growth

# ─── DB write helpers ─────────────────────────────────────
def _ensure_table(conn):
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            Date TEXT,
            Ticker TEXT,
            Growth_Type TEXT,    -- 'TTM' | 'Forward'
            Implied_Growth REAL,
            PRIMARY KEY (Date, Ticker, Growth_Type)
        )
    """)

def _log_today(yld):
    today = datetime.today().strftime("%Y-%m-%d")
    with sqlite3.connect(DB_PATH) as conn:
        _ensure_table(conn)
        cur = conn.cursor()
        for tk in INDEXES:
            ttm_pe, fwd_pe   = _fetch_pe(tk)
            ttm_g, fwd_g     = _compute_growth(ttm_pe, fwd_pe, yld)

            if ttm_g is not None:
                cur.execute("INSERT OR REPLACE INTO Index_Growth_History VALUES (?,?, 'TTM', ?)",
                            (today, tk, ttm_g))
            if fwd_g is not None:
                cur.execute("INSERT OR REPLACE INTO Index_Growth_History VALUES (?,?, 'Forward', ?)",
                            (today, tk, fwd_g))
        conn.commit()

# ─── Latest TTM helper for dashboard ───────────────────────
def _latest_ttm_growth(ticker):
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("""
            SELECT Implied_Growth
            FROM Index_Growth_History
            WHERE Ticker=? AND Growth_Type='TTM'
            ORDER BY Date DESC LIMIT 1
        """, (ticker,)).fetchone()
    return row[0] if row else None

# ─── Overview table (three-column) ─────────────────────────
def _overview_table():
    html_rows = []
    for tk in INDEXES:
        ttm_pe, _ = _fetch_pe(tk)
        growth    = _latest_ttm_growth(tk)
        if ttm_pe is None or growth is None:
            html_rows.append(f"<tr><td>{tk}</td><td colspan='2'>No implied-growth data yet.</td></tr>")
        else:
            link = f'<a href="{tk.lower()}_growth.html">{tk}</a>'
            html_rows.append(
                f"<tr><td>{link}</td><td>{ttm_pe:.1f}</td><td>{growth:.1%}</td></tr>"
            )
    return (
        "<table border='1' style='border-collapse:collapse;'>"
        "<thead><tr><th>Ticker</th><th>P/E Ratio</th><th>Implied Growth</th></tr></thead>"
        "<tbody>" + "".join(html_rows) + "</tbody></table>"
    )

# ─── Historical DF for charts ──────────────────────────────
def _hist_df(tk):
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("""
            SELECT Date, Growth_Type, Implied_Growth
            FROM Index_Growth_History
            WHERE Ticker=? ORDER BY Date ASC
        """, conn, params=(tk,))
    if df.empty: return None
    df["Date"] = pd.to_datetime(df["Date"])
    return df.pivot(index="Date", columns="Growth_Type", values="Implied_Growth")

# ─── Chart + full summary table for each index ─────────────
def _plot_and_table(tk):
    df = _hist_df(tk)
    # Always write summary HTML (even with <3 rows)
    out_html = os.path.join(CHART_DIR, f"{tk.lower()}_growth_summary.html")
    if df is None or df.empty:
        open(out_html, "w", encoding="utf-8").write(
            "<p>No implied-growth data yet.</p>"
        )
        return

    summary = {
        col: {
            "Average": df[col].mean(),
            "Median":  df[col].median(),
            "Min":     df[col].min(),
            "Max":     df[col].max()
        } for col in ["TTM","Forward"] if col in df
    }
    rows = []
    link = f'<a href="{tk.lower()}_growth.html">{tk}</a>'
    for g in ["TTM","Forward"]:
        if g not in summary: continue
        for stat, val in summary[g].items():
            rows.append({
                "Ticker": link, "Growth Type": g,
                "Statistic": stat, "Value": f"{val:.2%}"
            })
    pd.DataFrame(rows).to_html(out_html, index=False, escape=False)

    # draw chart only when ≥3 rows
    if len(df) < 3: return

    fig, ax = plt.subplots(figsize=(10,6))
    if "TTM" in df:
        ax.plot(df.index, df["TTM"], label="TTM", color="blue")
        for lbl, ls in [("Average",":"),("Median","--"),("Min","-."),("Max","-.")]:
            ax.axhline(summary["TTM"][lbl], color="blue", linestyle=ls, linewidth=1)
    if "Forward" in df:
        ax.plot(df.index, df["Forward"], label="Forward", color="green")
        for lbl, ls in [("Average",":"),("Median","--"),("Min","-."),("Max","-.")]:
            ax.axhline(summary["Forward"][lbl], color="green", linestyle=ls, linewidth=1)

    ax.set_title(f"{tk} Implied Growth Rates Over Time")
    ax.set_ylabel("Implied Growth Rate")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(True, linestyle="--", alpha=.4)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, f"{tk.lower()}_growth_chart.png"))
    plt.close()

def _refresh_assets():
    for tk in INDEXES:
        _plot_and_table(tk)

# ─── MINI-MAIN (imported by main_remote.py) ────────────────
def index_growth(treasury_yield: float | None = None) -> str:
    """
    Called from main_remote.py

    Parameters
    ----------
    treasury_yield : float or None
        Raw 10-yr yield (0.043, 4.3, 43.0, etc.)
    Returns
    -------
    str – HTML snippet (three-column overview table).
    """
    y = _normalize_yield(treasury_yield)
    print(f"[index_growth] Using 10-yr yield = {y:.4f}")
    _log_today(y)
    _refresh_assets()
    return _overview_table()

# ─── Stand-alone test run ─────────────────────────────────
if __name__ == "__main__":
    html = index_growth()  # uses fallback yield
    path = os.path.join(CHART_DIR, "spy_qqq_overview.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print("Wrote", path)

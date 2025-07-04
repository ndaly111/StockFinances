# index_growth_table.py
# ───────────────────────────────────────────────────────────
# Mini-main  index_growth(treasury_yield)  — called by main_remote.py
#
# • Logs today’s implied growth for SPY & QQQ into SQLite
# • Regenerates charts + summary tables
# • Returns one combined HTML snippet for the dashboard
#
# Stand-alone test:  python index_growth_table.py
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
FALLBACK_YIELD = 0.045     # 4.5 % used if caller passes None
CHART_DIR      = "charts"
os.makedirs(CHART_DIR, exist_ok=True)

# ─── Helper: normalize yield to decimal ────────────────────
def _normalize_yield(value):
    """
    Accepts 0.043 → 0.043   | 4.3 → 0.043   | 43.0 (^TNX) → 0.043
    """
    if value is None:
        return FALLBACK_YIELD
    try:
        v = float(value)
        if v < 0.5:   # already decimal
            return v
        if v < 20:    # percent form
            return v / 100
        return v / 1000  # ^TNX quote
    except Exception:
        return FALLBACK_YIELD

# ─── DB helpers ─────────────────────────────────────────────
def _compute_growth(ttm_pe, fwd_pe, y):
    if not ttm_pe or not fwd_pe:
        return None, None
    return y * ttm_pe - 1, y * fwd_pe - 1

def _fetch_pe(ticker):
    info = yf.Ticker(ticker).info
    return info.get("trailingPE"), info.get("forwardPE")

def _ensure_table(conn):
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            Date TEXT,
            Ticker TEXT,
            Growth_Type TEXT,      -- 'TTM' | 'Forward'
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
                cur.execute(f"""
                    INSERT OR REPLACE INTO {TABLE_NAME}
                    VALUES (?,?, 'TTM', ?)
                """, (today, tk, ttm_g))
            if fwd_g is not None:
                cur.execute(f"""
                    INSERT OR REPLACE INTO {TABLE_NAME}
                    VALUES (?,?, 'Forward', ?)
                """, (today, tk, fwd_g))
        conn.commit()

# ─── Chart & summary generation ────────────────────────────
def _hist_df(tk):
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            "SELECT Date, Growth_Type, Implied_Growth "
            "FROM Index_Growth_History "
            "WHERE Ticker = ? ORDER BY Date ASC",
            conn, params=(tk,))
    if df.empty:
        return None
    df["Date"] = pd.to_datetime(df["Date"])
    return df.pivot(index="Date", columns="Growth_Type", values="Implied_Growth")

def _plot_and_table(tk):
    df = _hist_df(tk)
    if df is None or len(df) < 3:
        return
    summary = {
        col: {
            "Average": df[col].mean(),
            "Median":  df[col].median(),
            "Min":     df[col].min(),
            "Max":     df[col].max()
        } for col in ["TTM", "Forward"] if col in df
    }

    # chart
    fig, ax = plt.subplots(figsize=(10, 6))
    if "TTM" in df:
        ax.plot(df.index, df["TTM"], label="TTM", color="blue")
        for lbl, ls in [("Average", ":"), ("Median", "--"), ("Min", "-."), ("Max", "-.")]:
            ax.axhline(summary["TTM"][lbl], color="blue", linestyle=ls, linewidth=1)
    if "Forward" in df:
        ax.plot(df.index, df["Forward"], label="Forward", color="green")
        for lbl, ls in [("Average", ":"), ("Median", "--"), ("Min", "-."), ("Max", "-.")]:
            ax.axhline(summary["Forward"][lbl], color="green", linestyle=ls, linewidth=1)

    ax.set_title(f"{tk} Implied Growth Rates Over Time")
    ax.set_ylabel("Implied Growth Rate")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, f"{tk.lower()}_growth_chart.png"))
    plt.close()

    # summary HTML
    rows, link = [], f'<a href="{tk.lower()}_growth.html">{tk}</a>'
    for g in ["TTM", "Forward"]:
        if g not in summary: continue
        for stat, val in summary[g].items():
            rows.append({
                "Ticker": link, "Growth Type": g,
                "Statistic": stat, "Value": f"{val:.2%}"
            })
    pd.DataFrame(rows).to_html(
        os.path.join(CHART_DIR, f"{tk.lower()}_growth_summary.html"),
        index=False, escape=False
    )

def _refresh_assets():
    for tk in INDEXES:
        _plot_and_table(tk)

# ─── MINI-MAIN (imported by main_remote.py) ─────────────────
def index_growth(treasury_yield: float | None = None) -> str:
    """
    Parameters
    ----------
    treasury_yield : float or None
        Raw 10-year yield fetched in main_remote.py.
        Accepts decimal (0.043), percent (4.3) or ^TNX quote (43.0).
    Returns
    -------
    str – combined SPY & QQQ summary HTML snippet.
    """
    y = _normalize_yield(treasury_yield)
    print(f"[index_growth] Using 10-yr yield = {y:.4f}")

    _log_today(y)
    _refresh_assets()

    blocks = []
    for tk in INDEXES:
        blocks.append(f"<h3>{tk}</h3>")
        with open(os.path.join(CHART_DIR, f"{tk.lower()}_growth_summary.html"),
                  encoding="utf-8") as f:
            blocks.append(f.read())
    return "\n".join(blocks)

# ─── Stand-alone test run ──────────────────────────────────
if __name__ == "__main__":
    html = index_growth()  # uses fallback yield
    out  = os.path.join(CHART_DIR, "spy_qqq_combined_summary.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print("Wrote", out)

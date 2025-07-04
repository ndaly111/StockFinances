# index_growth_table.py
# ---------------------------------------------------------------------
# Mini-main function `index_growth()` for main_remote.py
#
# • Logs today’s implied growth for SPY & QQQ into SQLite
# • Regenerates charts + summary tables
# • Returns one combined HTML snippet for the dashboard
#
# Stand-alone use:  python index_growth_table.py
# ---------------------------------------------------------------------

import os, sqlite3
from datetime import datetime
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# ───────────────────────────────────────────────────────────
#  Configuration
# ───────────────────────────────────────────────────────────
DB_PATH        = "Stock Data.db"
TABLE_NAME     = "Index_Growth_History"
INDEXES        = ["SPY", "QQQ"]
TREASURY_YIELD = 0.045         # 10-yr Treasury yield for Gordon model
CHART_DIR      = "charts"
os.makedirs(CHART_DIR, exist_ok=True)

# ───────────────────────────────────────────────────────────
#  DB helpers
# ───────────────────────────────────────────────────────────
def compute_growth(ttm_pe: float, fwd_pe: float):
    if not ttm_pe or not fwd_pe:
        return None, None
    return TREASURY_YIELD * ttm_pe - 1, TREASURY_YIELD * fwd_pe - 1

def fetch_pe_ratios(ticker: str):
    info = yf.Ticker(ticker).info
    return info.get("trailingPE"), info.get("forwardPE")

def ensure_table_exists(conn):
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            Date           TEXT,
            Ticker         TEXT,
            Growth_Type    TEXT,    -- 'TTM' | 'Forward'
            Implied_Growth REAL,
            PRIMARY KEY (Date, Ticker, Growth_Type)
        )
    """)
    conn.commit()

def log_today():
    """Append today’s implied growth for each index."""
    today = datetime.today().strftime("%Y-%m-%d")
    conn  = sqlite3.connect(DB_PATH)
    ensure_table_exists(conn)
    cur   = conn.cursor()

    for tk in INDEXES:
        ttm_pe, fwd_pe         = fetch_pe_ratios(tk)
        ttm_growth, fwd_growth = compute_growth(ttm_pe, fwd_pe)

        if ttm_growth is not None:
            cur.execute(f"""
                INSERT OR REPLACE INTO {TABLE_NAME}
                VALUES (?, ?, 'TTM', ?)
            """, (today, tk, ttm_growth))

        if fwd_growth is not None:
            cur.execute(f"""
                INSERT OR REPLACE INTO {TABLE_NAME}
                VALUES (?, ?, 'Forward', ?)
            """, (today, tk, fwd_growth))

    conn.commit(); conn.close()

# ───────────────────────────────────────────────────────────
#  Chart & summary generation
# ───────────────────────────────────────────────────────────
def fetch_history(ticker: str):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(f"""
        SELECT Date, Growth_Type, Implied_Growth
        FROM {TABLE_NAME} WHERE Ticker = ?
        ORDER BY Date ASC
    """, conn, params=(ticker,))
    conn.close()
    if df.empty: 
        return None
    df["Date"] = pd.to_datetime(df["Date"])
    return df.pivot(index="Date", columns="Growth_Type", values="Implied_Growth")

def build_chart_and_table(ticker: str):
    df = fetch_history(ticker)
    if df is None or len(df) < 3:
        return  # not enough data yet

    summary = {
        col: {
            "Average": df[col].mean(),
            "Median":  df[col].median(),
            "Min":     df[col].min(),
            "Max":     df[col].max()
        } for col in ["TTM", "Forward"] if col in df
    }

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(10, 6))
    if "TTM" in df:
        ax.plot(df.index, df["TTM"], label="TTM Growth", color="blue")
        for lbl, ls in [("Average", ":"), ("Median", "--"), ("Min", "-."), ("Max", "-.")]:
            ax.axhline(summary["TTM"][lbl], color="blue", linestyle=ls, linewidth=1)
    if "Forward" in df:
        ax.plot(df.index, df["Forward"], label="Forward Growth", color="green")
        for lbl, ls in [("Average", ":"), ("Median", "--"), ("Min", "-."), ("Max", "-.")]:
            ax.axhline(summary["Forward"][lbl], color="green", linestyle=ls, linewidth=1)

    ax.set_title(f"{ticker} Implied Growth Rates Over Time")
    ax.set_ylabel("Implied Growth Rate")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, f"{ticker.lower()}_growth_chart.png"))
    plt.close()

    # ---- summary table ----
    rows, link = [], f'<a href="{ticker.lower()}_growth.html">{ticker}</a>'
    for gtype in ["TTM", "Forward"]:
        if gtype not in summary: 
            continue
        for stat, val in summary[gtype].items():
            rows.append({
                "Ticker": link,
                "Growth Type": gtype,
                "Statistic": stat,
                "Value": f"{val:.2%}"
            })
    pd.DataFrame(rows).to_html(
        os.path.join(CHART_DIR, f"{ticker.lower()}_growth_summary.html"),
        index=False, escape=False
    )

def refresh_outputs():
    for tk in INDEXES:
        build_chart_and_table(tk)

# ───────────────────────────────────────────────────────────
#  MINI-MAIN FUNCTION (imported by main_remote.py)
# ───────────────────────────────────────────────────────────
def index_growth() -> str:
    """
    • Logs today’s data
    • Rebuilds charts + summary tables
    • Returns combined SPY/QQQ HTML snippet for dashboard
    """
    print("[index_growth] Updating SPY & QQQ implied-growth assets …")
    log_today()
    refresh_outputs()

    blocks = []
    for tk in INDEXES:
        blocks.append(f"<h3>{tk}</h3>")
        with open(os.path.join(CHART_DIR, f"{tk.lower()}_growth_summary.html"),
                  encoding="utf-8") as f:
            blocks.append(f.read())
    return "\n".join(blocks)

# ───────────────────────────────────────────────────────────
#  Stand-alone execution for testing
# ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    snippet = index_growth()   # calls the same mini-main
    out = os.path.join(CHART_DIR, "spy_qqq_combined_summary.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(snippet)
    print(f"Combined summary written → {out}")

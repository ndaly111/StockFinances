import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DB_PATH        = "Stock Data.db"
TABLE_NAME     = "Implied_Growth_History"
CHART_DIR      = "charts"
HTML_TEMPLATE  = os.path.join(CHART_DIR, "{ticker}_implied_growth_summary.html")
CHART_TEMPLATE = os.path.join(CHART_DIR, "{ticker}_implied_growth_plot.png")

TIME_FRAMES = {
    "1 Year":   365,
    "3 Years":  365 * 3,
    "5 Years":  365 * 5,
    "10 Years": 365 * 10,
}

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def ensure_output_directory():
    os.makedirs(CHART_DIR, exist_ok=True)

def load_growth_data():
    """Return a DataFrame of the history table (empty if file/table missing)."""
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    with sqlite3.connect(DB_PATH) as conn:
        try:
            return pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)
        except sqlite3.OperationalError:
            return pd.DataFrame()

def calculate_stats(series: pd.Series):
    """Return avg, med, std, cur, pct (or '-')."""
    if series.empty:
        return "-", "-", "-", "-", "-"
    avg = series.mean()
    med = series.median()
    std = series.std()
    cur = series.iloc[-1]
    pct = series.rank(pct=True).iloc[-1] * 100
    fmt = lambda x: f"{x:.2%}"
    return fmt(avg), fmt(med), fmt(std), fmt(cur), f"{pct:.1f}%"

# ---------------------------------------------------------------------------
# RENDERING
# ---------------------------------------------------------------------------
def generate_summary_table(df: pd.DataFrame, ticker: str):
    """Create per-ticker HTML summary table."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date_recorded"])
    now = datetime.now()

    rows = []
    for label, days in TIME_FRAMES.items():
        cutoff = now - pd.Timedelta(days=days)
        window = df[df["date"] >= cutoff]
        for typ in ["TTM", "Forward"]:
            subset = window[window["growth_type"] == typ]["growth_value"].dropna()
            avg, med, std, cur, pct = calculate_stats(subset)
            rows.append(
                dict(Timeframe=label, Type=typ,
                     **{"Average": avg, "Median": med,
                        "Std Dev": std, "Current": cur, "Percentile": pct})
            )

    out_path = HTML_TEMPLATE.format(ticker=ticker)
    pd.DataFrame(rows).to_html(out_path, index=False, na_rep="-", justify="center")
    return out_path

def plot_growth_chart(df: pd.DataFrame, ticker: str):
    """Save a PNG chart of TTM & Forward implied growth."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date_recorded"])

    fig, ax = plt.subplots(figsize=(10, 6))
    for typ, color in [("TTM", "blue"), ("Forward", "green")]:
        s = df[df["growth_type"] == typ].sort_values("date")
        if s.empty:
            continue
        ax.plot(s["date"], s["growth_value"], label=f"{typ} Growth",
                color=color, linewidth=1.5)
        m, med, std = s["growth_value"].mean(), s["growth_value"].median(), s["growth_value"].std()
        ax.axhline(m,    ls="--", label=f"{typ} Avg",    color=color, alpha=0.7)
        ax.axhline(med,  ls=":",  label=f"{typ} Median", color=color, alpha=0.7)
        ax.axhline(m+std, ls="-.", label=f"{typ} +1σ",   color=color, alpha=0.5)
        ax.axhline(m-std, ls="-.", label=f"{typ} -1σ",   color=color, alpha=0.5)

    ax.set_title("Implied Growth Rates Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth Rate")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(loc="upper left", fontsize="small")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    out_path = CHART_TEMPLATE.format(ticker=ticker)
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path

# ---------------------------------------------------------------------------
# MINI-MAIN (do NOT rename)
# ---------------------------------------------------------------------------
def generate_all_summaries():
    """Loop over tickers, write HTML + PNG for each."""
    ensure_output_directory()
    df_all = load_growth_data()
    if "ticker" not in df_all.columns:
        print("[generate_all_summaries] No history data found.")
        return
    for ticker in df_all["ticker"].unique():
        df_t = df_all[df_all["ticker"] == ticker]
        print(f"[generate_all_summaries] Processing {ticker}")
        generate_summary_table(df_t, ticker)
        plot_growth_chart(df_t, ticker)

# Keep this EXACT name; main_remote.py relies on it
if __name__ == "__main__":
    generate_all_summaries()

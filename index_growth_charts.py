# index_growth_charts.py
# -----------------------------------------------------------
# Builds SPY & QQQ implied-growth charts + summary tables
# (keeps Avg / Med / Min / Max, then appends Percentile rows)
# -----------------------------------------------------------

import os, sqlite3, pandas as pd, matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy.stats import percentileofscore            # already in requirements

DB_PATH   = "Stock Data.db"
OUT_DIR   = "charts"
INDEXES   = ["SPY", "QQQ"]

os.makedirs(OUT_DIR, exist_ok=True)

# ────────────────────────────────────────────────────────────
# UNIVERSAL HELPERS
# ────────────────────────────────────────────────────────────
def _percentile(series, value):
    """Return percentile (0-100) of `value` within `series`."""
    series = [v for v in series if v is not None]
    if not series:
        return None
    return round(percentileofscore(series, value), 2)

def _sql(conn, q, params=()):
    return pd.read_sql_query(q, conn, params=params)

# ────────────────────────────────────────────────────────────
# PRICE LOOK-UP  (for Forward-EPS calc later)
# ────────────────────────────────────────────────────────────
def _latest_price(conn, tk):
    df = _sql(conn, "SELECT last_price FROM MarketData WHERE ticker = ?", (tk,))
    return None if df.empty else df["last_price"].iloc[0]

# ────────────────────────────────────────────────────────────
# PULL LATEST GROWTH / PE VALUES -- ENTIRE UNIVERSE
# ────────────────────────────────────────────────────────────
def _latest_universe(table, type_col, type_val, value_col):
    """
    One latest row per ticker for a given type (TTM or Forward).
    Returns dataframe with columns: Ticker, Value
    """
    df = _sql(
        sqlite3.connect(DB_PATH),
        f"""
           SELECT Ticker, {value_col} as val
           FROM   {table}
           WHERE  {type_col} = ?
             AND  {value_col} IS NOT NULL
           QUALIFY row_number() OVER (PARTITION BY Ticker ORDER BY Date DESC)=1
        """,
        (type_val,),
    )
    if df.empty:
        return pd.DataFrame(columns=["Ticker", "val"])
    df["Ticker"] = df["Ticker"].str.upper()
    return df

IG_UNIV = _latest_universe("Index_Growth_History", "Growth_Type", "TTM", "Implied_Growth")
PE_UNIV = _latest_universe("Index_PE_History",     "PE_Type",    "TTM", "PE_Ratio")
FWD_PE_UNIV = _latest_universe("Index_PE_History", "PE_Type",    "Forward", "PE_Ratio")

# ────────────────────────────────────────────────────────────
# PER-TICKER HISTORY FETCH  (Implied-Growth series for charts)
# ────────────────────────────────────────────────────────────
def _growth_history(tk):
    with sqlite3.connect(DB_PATH) as conn:
        df = _sql(
            conn,
            """
            SELECT Date, Implied_Growth
            FROM   Index_Growth_History
            WHERE  Ticker = ? AND Growth_Type = 'TTM'
            ORDER  BY Date
            """,
            (tk,),
        )
    if df.empty:
        return None
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date")["Implied_Growth"]

# ────────────────────────────────────────────────────────────
# SUMMARY-OF-SERIES STATS (Avg / Med / Min / Max)
# ────────────────────────────────────────────────────────────
def _series_stats(series):
    return {
        "Average": series.mean(),
        "Median":  series.median(),
        "Min":     series.min(),
        "Max":     series.max(),
    }

# ────────────────────────────────────────────────────────────
# BUILD HTML SUMMARY TABLE
# ────────────────────────────────────────────────────────────
def _build_html(tk, ig_stats, pe_stats,
                ig_latest, ig_pct,
                pe_latest, pe_pct,
                fwd_eps=None, fwd_pct=None):
    rows = []

    # 1-- full stats block
    for stat, val in ig_stats.items():
        rows.append({"Metric": "Implied Growth (TTM)", "Statistic": stat, "Value": f"{val:.2%}"})
    for stat, val in pe_stats.items():
        rows.append({"Metric": "PE Ratio (TTM)", "Statistic": stat, "Value": f"{val:.2f}"})

    # 2-- percentile rows
    rows.append({"Metric": "Implied Growth (TTM)", "Statistic": "Percentile", "Value": f"{ig_pct:.2f}"})
    rows.append({"Metric": "PE Ratio (TTM)",       "Statistic": "Percentile", "Value": f"{pe_pct:.2f}"})

    # 3-- optional Forward-EPS rows
    if fwd_eps is not None:
        rows.append({"Metric": "Forward EPS", "Statistic": "Value",      "Value": f"{fwd_eps:.2f}"})
        rows.append({"Metric": "Forward EPS", "Statistic": "Percentile", "Value": f"{fwd_pct:.2f}"})

    out = pd.DataFrame(rows)[["Metric", "Statistic", "Value"]]
    out.to_html(os.path.join(OUT_DIR, f"{tk.lower()}_growth_summary.html"),
                index=False, escape=False)

# ────────────────────────────────────────────────────────────
# BUILD SIMPLE GROWTH CHART  (TTM only)
# ────────────────────────────────────────────────────────────
def _build_chart(series, tk):
    png = os.path.join(OUT_DIR, f"{tk.lower()}_growth_chart.png")
    if series is None or series.empty:
        plt.figure(figsize=(0.01,0.01)); plt.axis("off"); plt.savefig(png, transparent=True)
        plt.close(); return
    plt.figure(figsize=(10,6))
    plt.plot(series.index, series, color="blue")
    plt.title(f"{tk} Implied Growth (TTM)")
    plt.ylabel("Implied Growth Rate")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.grid(True, ls="--", alpha=.4)
    plt.tight_layout(); plt.savefig(png); plt.close()

# ────────────────────────────────────────────────────────────
# MAIN DRIVER
# ────────────────────────────────────────────────────────────
def render_index_growth_charts():
    print("[index_growth_charts] building SPY & QQQ pages …")

    for tk in INDEXES:
        # ---------- series for stats ----------
        series = _growth_history(tk)
        if series is None:
            print(f"  ! {tk}: no growth history."); continue
        ig_stats = _series_stats(series)

        # ---------- latest values / percentiles ----------
        ig_row = IG_UNIV[IG_UNIV["Ticker"] == tk]
        pe_row = PE_UNIV[PE_UNIV["Ticker"] == tk]
        if ig_row.empty or pe_row.empty:
            print(f"  ! {tk}: missing latest universe rows."); continue

        ig_latest = ig_row["val"].iloc[0]
        pe_latest = pe_row["val"].iloc[0]
        ig_pct    = _percentile(IG_UNIV["val"], ig_latest)
        pe_pct    = _percentile(PE_UNIV["val"], pe_latest)

        # ---------- PE series stats (TTM) ----------
        with sqlite3.connect(DB_PATH) as conn:
            pe_series = _sql(conn,
                """
                SELECT Date, PE_Ratio
                FROM   Index_PE_History
                WHERE  Ticker = ? AND PE_Type = 'TTM'
                ORDER BY Date
                """, (tk,)
            )
        pe_stats = _series_stats(pe_series["PE_Ratio"]) if not pe_series.empty else {}

        # ---------- Forward EPS (optional) ----------
        fwd_eps = fwd_pct = None
        if not FWD_PE_UNIV.empty and tk in FWD_PE_UNIV["Ticker"].values:
            with sqlite3.connect(DB_PATH) as conn:
                price = _latest_price(conn, tk)
            fwd_pe  = FWD_PE_UNIV[FWD_PE_UNIV["Ticker"] == tk]["val"].iloc[0]
            if price and fwd_pe:
                fwd_eps = price / fwd_pe
                fwd_pct = _percentile(price / FWD_PE_UNIV["val"], fwd_eps)

        # ---------- output ----------
        _build_html(tk, ig_stats, pe_stats,
                    ig_latest, ig_pct, pe_latest, pe_pct,
                    fwd_eps, fwd_pct)
        _build_chart(series, tk)

    print("[index_growth_charts] done.")

# run standalone
if __name__ == "__main__":
    render_index_growth_charts()

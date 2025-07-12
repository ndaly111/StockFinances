# index_growth_charts.py
# -----------------------------------------------------------
# Builds SPY & QQQ implied-growth charts + summary tables
#  • keeps Avg / Med / Min / Max
#  • appends trailing-percentile rows
# -----------------------------------------------------------

import os, sqlite3, pandas as pd, matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy.stats import percentileofscore

DB_PATH  = "Stock Data.db"
OUT_DIR  = "charts"
INDEXES  = ["SPY", "QQQ"]

os.makedirs(OUT_DIR, exist_ok=True)

# ─── helpers ────────────────────────────────────────────────
def _sql(conn, q, params=()):
    return pd.read_sql_query(q, conn, params=params)

def _percentile(series, value):
    series = [v for v in series if v is not None]
    return round(percentileofscore(series, value), 2) if series else None

# ---------- pull latest value per-ticker without QUALIFY ----
def _latest_universe(table, type_col, type_val, value_col):
    with sqlite3.connect(DB_PATH) as conn:
        df = _sql(
            conn,
            f"""
              SELECT Ticker, Date, {value_col} AS val
              FROM   {table}
              WHERE  {type_col} = ?
                     AND {value_col} IS NOT NULL
            """,
            (type_val,),
        )
    if df.empty:
        return pd.DataFrame(columns=["Ticker", "val"])
    df["Ticker"] = df["Ticker"].str.upper()
    df = (
        df.sort_values(["Ticker", "Date"])
          .drop_duplicates(subset=["Ticker"], keep="last")
          .reset_index(drop=True)[["Ticker", "val"]]
    )
    return df

# universes
IG_UNIV   = _latest_universe("Index_Growth_History", "Growth_Type", "TTM",     "Implied_Growth")
PE_UNIV   = _latest_universe("Index_PE_History",     "PE_Type",    "TTM",     "PE_Ratio")
FWD_PE_U  = _latest_universe("Index_PE_History",     "PE_Type",    "Forward", "PE_Ratio")

# ---------- price lookup ----------
def _latest_price(conn, tk):
    df = _sql(conn, "SELECT last_price FROM MarketData WHERE ticker = ?", (tk,))
    return None if df.empty else df["last_price"].iloc[0]

# ---------- per-ticker TTM growth series (for chart) ----------
def _growth_series(tk):
    with sqlite3.connect(DB_PATH) as conn:
        df = _sql(
            conn,
            """
            SELECT Date, Implied_Growth
            FROM   Index_Growth_History
            WHERE  Ticker = ? AND Growth_Type='TTM'
            ORDER  BY Date
            """,
            (tk,),
        )
    if df.empty:
        return None
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date")["Implied_Growth"]

def _series_stats(series):
    return {
        "Average": series.mean(),
        "Median":  series.median(),
        "Min":     series.min(),
        "Max":     series.max(),
    }

# ---------- HTML summary ----------
def _build_html(tk, ig_stats, pe_stats,
                ig_latest, ig_pct,
                pe_latest, pe_pct,
                fwd_eps=None, fwd_pct=None):

    rows = []
    # full stats
    for s, v in ig_stats.items():
        rows.append({"Metric":"Implied Growth (TTM)", "Statistic":s, "Value":f"{v:.2%}"})
    for s, v in pe_stats.items():
        rows.append({"Metric":"PE Ratio (TTM)",       "Statistic":s, "Value":f"{v:.2f}"})
    # trailing percentiles
    rows.append({"Metric":"Implied Growth (TTM)", "Statistic":"Percentile", "Value":f"{ig_pct:.2f}"})
    rows.append({"Metric":"PE Ratio (TTM)",       "Statistic":"Percentile", "Value":f"{pe_pct:.2f}"})
    # optional Forward EPS
    if fwd_eps is not None:
        rows.append({"Metric":"Forward EPS", "Statistic":"Value",      "Value":f"{fwd_eps:.2f}"})
        rows.append({"Metric":"Forward EPS", "Statistic":"Percentile", "Value":f"{fwd_pct:.2f}"})

    pd.DataFrame(rows)[["Metric","Statistic","Value"]].to_html(
        os.path.join(OUT_DIR, f"{tk.lower()}_growth_summary.html"),
        index=False, escape=False
    )

# ---------- growth chart ----------
def _build_chart(series, tk):
    png = os.path.join(OUT_DIR, f"{tk.lower()}_growth_chart.png")
    if series is None or series.empty:
        plt.figure(figsize=(0.01,0.01)); plt.axis("off"); plt.savefig(png, transparent=True); plt.close(); return
    plt.figure(figsize=(10,6))
    plt.plot(series.index, series, color="blue")
    plt.title(f"{tk} Implied Growth (TTM)")
    plt.ylabel("Implied Growth Rate")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.grid(True, ls="--", alpha=.4)
    plt.tight_layout(); plt.savefig(png); plt.close()

# ---------- main driver ----------
def render_index_growth_charts():
    print("[index_growth_charts] building SPY & QQQ pages …")

    for tk in INDEXES:
        # series and stats
        g_series = _growth_series(tk)
        if g_series is None: 
            print(f"  ! {tk}: no TTM growth history"); continue
        ig_stats = _series_stats(g_series)

        with sqlite3.connect(DB_PATH) as conn:
            pe_series = _sql(conn,
                """
                SELECT PE_Ratio FROM Index_PE_History
                WHERE  Ticker = ? AND PE_Type='TTM' AND PE_Ratio IS NOT NULL
                """,(tk,))
        pe_stats = _series_stats(pe_series["PE_Ratio"]) if not pe_series.empty else {}

        # latest values + percentiles
        ig_latest = IG_UNIV.loc[IG_UNIV["Ticker"]==tk,"val"].iloc[0]
        pe_latest = PE_UNIV.loc[PE_UNIV["Ticker"]==tk,"val"].iloc[0]
        ig_pct = _percentile(IG_UNIV["val"], ig_latest)
        pe_pct = _percentile(PE_UNIV["val"], pe_latest)

        # optional Forward EPS
        fwd_eps = fwd_pct = None
        if not FWD_PE_U.empty and tk in FWD_PE_U["Ticker"].values:
            with sqlite3.connect(DB_PATH) as conn:
                price = _latest_price(conn, tk)
            fwd_pe  = FWD_PE_U.loc[FWD_PE_U["Ticker"]==tk,"val"].iloc[0]
            if price and fwd_pe:
                fwd_eps = price / fwd_pe
                fwd_pct = _percentile(price / FWD_PE_U["val"], fwd_eps)

        # output
        _build_html(tk, ig_stats, pe_stats,
                    ig_latest, ig_pct,
                    pe_latest, pe_pct,
                    fwd_eps, fwd_pct)
        _build_chart(g_series, tk)

    print("[index_growth_charts] done.")

# standalone
if __name__ == "__main__":
    render_index_growth_charts()

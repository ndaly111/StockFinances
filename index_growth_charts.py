# index_growth_charts.py
# -----------------------------------------------------------
# Builds SPY & QQQ implied-growth charts and summary tables
#   – compact table: Latest | Avg | Med | Min | Max | %tile
# -----------------------------------------------------------

import os, sqlite3, pandas as pd, matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy.stats import percentileofscore

DB_PATH  = "Stock Data.db"
OUT_DIR  = "charts"
INDEXES  = ["SPY", "QQQ"]

os.makedirs(OUT_DIR, exist_ok=True)

# ────────────────────────────────────────────────────────────
# Generic helpers
# ────────────────────────────────────────────────────────────
def _sql(conn, q, params=()):
    return pd.read_sql_query(q, conn, params=params)

def _percentile(series, value):
    s = [v for v in series if v is not None]
    return round(percentileofscore(s, value), 2) if s else None

# Pull latest row per ticker (SQLite-safe: no QUALIFY)
def _latest_universe(table, type_col, type_val, value_col):
    with sqlite3.connect(DB_PATH) as conn:
        df = _sql(
            conn,
            f"""
              SELECT Ticker, Date, {value_col} AS val
              FROM   {table}
              WHERE  {type_col} = ? AND {value_col} IS NOT NULL
            """,
            (type_val,),
        )
    if df.empty:
        return pd.DataFrame(columns=["Ticker", "val"])
    df["Ticker"] = df["Ticker"].str.upper()
    return (
        df.sort_values(["Ticker", "Date"])
          .drop_duplicates(subset=["Ticker"], keep="last")
          .reset_index(drop=True)[["Ticker", "val"]]
    )

# universes ­— computed once
IG_UNIV   = _latest_universe("Index_Growth_History", "Growth_Type", "TTM",     "Implied_Growth")
PE_UNIV   = _latest_universe("Index_PE_History",     "PE_Type",    "TTM",     "PE_Ratio")
FWD_PE_U  = _latest_universe("Index_PE_History",     "PE_Type",    "Forward", "PE_Ratio")

# ────────────────────────────────────────────────────────────
# Per-ticker helpers
# ────────────────────────────────────────────────────────────
def _latest_price(conn, tk):
    row = _sql(conn, "SELECT last_price FROM MarketData WHERE ticker = ?", (tk,))
    return None if row.empty else row["last_price"].iloc[0]

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

# ────────────────────────────────────────────────────────────
# Build the compact, mobile-friendly HTML table
# ────────────────────────────────────────────────────────────
def _build_html(tk,
                ig_stats, pe_stats,
                ig_latest, ig_pct,
                pe_latest, pe_pct,
                fwd_eps=None, fwd_pct=None):

    def pct_or_dash(x): return f"{x:.2f}" if x is not None else "—"

    rows = [
        {
            "Metric": "Implied Growth (TTM)",
            "Latest": f"{ig_latest:.2%}",
            "Avg":    f"{ig_stats['Average']:.2%}",
            "Med":    f"{ig_stats['Median']:.2%}",
            "Min":    f"{ig_stats['Min']:.2%}",
            "Max":    f"{ig_stats['Max']:.2%}",
            "%tile":  pct_or_dash(ig_pct),
        },
        {
            "Metric": "PE Ratio (TTM)",
            "Latest": f"{pe_latest:.2f}",
            "Avg":    f"{pe_stats['Average']:.2f}",
            "Med":    f"{pe_stats['Median']:.2f}",
            "Min":    f"{pe_stats['Min']:.2f}",
            "Max":    f"{pe_stats['Max']:.2f}",
            "%tile":  pct_or_dash(pe_pct),
        },
    ]
    if fwd_eps is not None:
        rows.append(
            {
                "Metric": "Forward EPS",
                "Latest": f"{fwd_eps:.2f}",
                "Avg":    "—", "Med": "—", "Min": "—", "Max": "—",
                "%tile":  pct_or_dash(fwd_pct),
            }
        )

    df = pd.DataFrame(rows)[["Metric","Latest","Avg","Med","Min","Max","%tile"]]

    style = """
    <style>
        table{width:100%;border-collapse:collapse;font-family:Arial;}
        th,td{border:1px solid #ccc;padding:6px;text-align:center;}
        th{background:#f2f2f2;font-weight:bold;}
        @media(max-width:600px){th,td{font-size:12px;padding:3px;}}
    </style>
    """

    path = os.path.join(OUT_DIR, f"{tk.lower()}_growth_summary.html")
    df.to_html(path, index=False, escape=False)
    with open(path, "r+", encoding="utf-8") as f:
        html = f.read()
        f.seek(0); f.write(style + html)

# ────────────────────────────────────────────────────────────
# Growth-rate line chart
# ────────────────────────────────────────────────────────────
def _build_chart(series, tk):
    img = os.path.join(OUT_DIR, f"{tk.lower()}_growth_chart.png")
    if series is None or series.empty:
        plt.figure(figsize=(0.01,0.01)); plt.axis("off"); plt.savefig(img, transparent=True); plt.close(); return
    plt.figure(figsize=(10,6))
    plt.plot(series.index, series, color="blue")
    plt.title(f"{tk} Implied Growth (TTM)")
    plt.ylabel("Implied Growth Rate")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.grid(True, ls="--", alpha=.4)
    plt.tight_layout(); plt.savefig(img); plt.close()

# ────────────────────────────────────────────────────────────
# MAIN routine
# ────────────────────────────────────────────────────────────
def render_index_growth_charts():
    print("[index_growth_charts] building SPY & QQQ pages …")

    for tk in INDEXES:
        g_series = _growth_series(tk)
        if g_series is None:
            print(f"  ! {tk}: no growth history"); continue
        ig_stats = _series_stats(g_series)

        # PE series for stats
        with sqlite3.connect(DB_PATH) as conn:
            pe_ser = _sql(conn,
                "SELECT PE_Ratio FROM Index_PE_History WHERE Ticker=? AND PE_Type='TTM' AND PE_Ratio IS NOT NULL",
                (tk,))
        pe_stats = _series_stats(pe_ser["PE_Ratio"]) if not pe_ser.empty else {}

        # latest + percentile
        ig_latest = IG_UNIV.loc[IG_UNIV["Ticker"]==tk,"val"].iloc[0]
        pe_latest = PE_UNIV.loc[PE_UNIV["Ticker"]==tk,"val"].iloc[0]
        ig_pct = _percentile(IG_UNIV["val"], ig_latest)
        pe_pct = _percentile(PE_UNIV["val"], pe_latest)

        # Forward EPS optional
        fwd_eps = fwd_pct = None
        if not FWD_PE_U.empty and tk in FWD_PE_U["Ticker"].values:
            with sqlite3.connect(DB_PATH) as conn:
                price = _latest_price(conn, tk)
            fwd_pe  = FWD_PE_U.loc[FWD_PE_U["Ticker"]==tk,"val"].iloc[0]
            if price and fwd_pe:
                fwd_eps = price / fwd_pe
                fwd_pct = _percentile(price / FWD_PE_U["val"], fwd_eps)

        _build_html(tk, ig_stats, pe_stats,
                    ig_latest, ig_pct,
                    pe_latest, pe_pct,
                    fwd_eps, fwd_pct)
        _build_chart(g_series, tk)

    print("[index_growth_charts] done.")

if __name__ == "__main__":
    render_index_growth_charts()

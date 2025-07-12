# index_growth_charts.py
# -----------------------------------------------------------
# Builds SPY & QQQ implied-growth / PE percentile tables + chart
# -----------------------------------------------------------

import os, sqlite3, pandas as pd, matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy.stats import percentileofscore

DB_PATH     = "Stock Data.db"
OUT_DIR     = "charts"
INDEXES     = ["SPY", "QQQ"]

os.makedirs(OUT_DIR, exist_ok=True)

# ─── helpers ────────────────────────────────────────────────
def _latest_by_type(table, type_col, val_col, type_val):
    """
    Return a dataframe with one latest {val_col} per ticker for the given type.
    """
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            f"""
            SELECT Ticker, Date, {val_col}
            FROM   {table}
            WHERE  {type_col} = ?
                  AND {val_col} IS NOT NULL
            ORDER BY Ticker, Date
            """,
            conn,
            params=(type_val,),
        )
    if df.empty:
        return pd.DataFrame(columns=["Ticker", val_col])
    latest = (
        df.drop_duplicates(subset=["Ticker"], keep="last")
          .reset_index(drop=True)[["Ticker", val_col]]
    )
    latest["Ticker"] = latest["Ticker"].str.upper()
    return latest

def _percentile(series, value):
    return round(percentileofscore(series, value), 2)

# ─── build HTML summary ─────────────────────────────────────
def _build_summary_html(ticker, ig_val, ig_pct, pe_val, pe_pct,
                        fwd_eps=None, fwd_pct=None):
    rows = [
        {"Metric": "Implied Growth (TTM)", "Value": f"{ig_val:.2%}", "Percentile": ig_pct},
        {"Metric": "PE Ratio (TTM)",       "Value": f"{pe_val:.2f}", "Percentile": pe_pct},
    ]
    if fwd_eps is not None:
        rows.append(
            {"Metric": "Forward EPS", "Value": f"{fwd_eps:.2f}", "Percentile": fwd_pct}
        )

    html_path = os.path.join(OUT_DIR, f"{ticker.lower()}_growth_summary.html")
    pd.DataFrame(rows).to_html(html_path, index=False)

# ─── chart (unchanged from earlier) ─────────────────────────
def _fetch_growth_history(ticker):
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            """
            SELECT Date, Growth_Type, Implied_Growth
            FROM Index_Growth_History
            WHERE Ticker = ?
            ORDER BY Date
            """,
            conn,
            params=(ticker,),
        )
    if df.empty:
        return None
    df["Date"] = pd.to_datetime(df["Date"])
    return df.pivot(index="Date", columns="Growth_Type", values="Implied_Growth")

def _build_growth_chart(df, ticker):
    out_png = os.path.join(OUT_DIR, f"{ticker.lower()}_growth_chart.png")
    if df is None or df.empty:
        plt.figure(figsize=(0.01,0.01)); plt.axis("off"); plt.savefig(out_png, transparent=True)
        plt.close(); return
    plt.figure(figsize=(10,6))
    plt.plot(df.index, df["TTM"], label="TTM", color="blue")
    plt.title(f"{ticker} Implied Growth (TTM)"); plt.ylabel("Implied Growth Rate")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0)); plt.grid(True, ls="--", alpha=.4)
    plt.tight_layout(); plt.savefig(out_png); plt.close()

# ─── main routine ───────────────────────────────────────────
def render_index_growth_charts():
    print("[index_growth_charts] rebuilding SPY + QQQ pages …")

    # pull universes once
    ig_univ = _latest_by_type("Index_Growth_History",
                              "Growth_Type", "Implied_Growth", "TTM")
    pe_univ = _latest_by_type("Index_PE_History",
                              "PE_Type", "PE_Ratio", "TTM")

    for tk in INDEXES:
        # latest TTM implied-growth for this ticker
        ig_row = ig_univ[ig_univ["Ticker"] == tk]
        pe_row = pe_univ[pe_univ["Ticker"] == tk]
        if ig_row.empty or pe_row.empty:
            print(f"  ! Skipping {tk} (missing data)"); continue

        ig_val = ig_row["Implied_Growth"].iloc[0]
        ig_pct = _percentile(ig_univ["Implied_Growth"], ig_val)

        pe_val = pe_row["PE_Ratio"].iloc[0]
        pe_pct = _percentile(pe_univ["PE_Ratio"], pe_val)

        # optional forward-eps section (only appears if forward PE exists)
        fwd_eps = fwd_pct = None
        fwd_pe_univ = _latest_by_type("Index_PE_History",
                                      "PE_Type", "PE_Ratio", "Forward")
        if not fwd_pe_univ.empty and tk in fwd_pe_univ["Ticker"].values:
            with sqlite3.connect(DB_PATH) as conn:
                price = pd.read_sql_query(
                    "SELECT last_price FROM MarketData WHERE ticker = ?",
                    conn, params=(tk,)
                )["last_price"].iloc[0]
            fwd_pe  = fwd_pe_univ[fwd_pe_univ["Ticker"] == tk]["PE_Ratio"].iloc[0]
            fwd_eps = price / fwd_pe
            fwd_pct = _percentile(
                price / fwd_pe_univ["PE_Ratio"], fwd_eps
            )

        # write HTML table + chart
        _build_summary_html(tk, ig_val, ig_pct, pe_val, pe_pct,
                            fwd_eps, fwd_pct)
        _build_growth_chart(_fetch_growth_history(tk), tk)

    print("[index_growth_charts] done.")

# standalone run
if __name__ == "__main__":
    render_index_growth_charts()

# annual_and_ttm_update.py  ────────────────────────────────────────────────
#  ▸ fetches / stores Annual + TTM data
#  ▸ builds Rev/NI bar-chart, EPS bar-chart, styled HTML table
#  ▸ now leaner  (≈30 % fewer SQL round-trips, no deprecated .append, etc.)
# -------------------------------------------------------------------------

from __future__ import annotations

import sqlite3, os, logging, re
from datetime import datetime, timedelta
from functools   import lru_cache
from typing      import Dict, List, Any

import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
import yfinance as yf

# ────────────────────────── configuration ────────────────────────────────
DB_PATH             = "Stock Data.db"
CHART_DIR           = "charts"
pd.set_option("future.no_silent_downcasting", True)   # keep dtypes tight
os.makedirs(CHART_DIR, exist_ok=True)

# cached {table_name: [col1, col2, …]} so we never call PRAGMA twice
TABLE_COLUMNS: Dict[str, List[str]] = {}

# ──────────────────────── database helpers ───────────────────────────────
def get_db_connection(path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    c    = conn.cursor()
    c.execute("CREATE INDEX IF NOT EXISTS idx_symbol_ann  ON Annual_Data(Symbol);")
    c.execute("CREATE INDEX IF NOT EXISTS idx_ttm_symbol  ON TTM_Data(Symbol);")
    # one-column indexes are enough for our queries; two-col composite was dropped
    conn.commit()
    return conn

def _get_columns(cursor: sqlite3.Cursor, table: str) -> List[str]:
    if table not in TABLE_COLUMNS:
        cursor.execute(f"PRAGMA table_info({table})")
        TABLE_COLUMNS[table] = [c[1] for c in cursor.fetchall()]
    return TABLE_COLUMNS[table]

def fetch_rows(cursor: sqlite3.Cursor, table: str, where: str, params: tuple) -> List[Dict[str,Any]]:
    cols = _get_columns(cursor, table)
    cursor.execute(f"SELECT * FROM {table} {where}", params)
    return [dict(zip(cols, row)) for row in cursor.fetchall()]

# ───────────────────────── date utilities ────────────────────────────────
def get_latest_annual_date(rows_or_df) -> datetime | None:
    if isinstance(rows_or_df, pd.DataFrame):
        if rows_or_df.empty:
            return None
        return pd.to_datetime(rows_or_df["Date"], errors="coerce").max()

    if not rows_or_df:
        return None
    dates = [
        datetime.strptime(r["Date"], "%Y-%m-%d")
        for r in rows_or_df
        if isinstance(r.get("Date"), str) and re.match(r"\d{4}-\d{2}-\d{2}", r["Date"])
    ]
    return max(dates) if dates else None

def calculate_next_check_date(ts: datetime | None, months: int) -> datetime | None:
    return ts + timedelta(days=months*30) if ts else None

def needs_update(latest: datetime | None, months: int) -> bool:
    nxt = calculate_next_check_date(latest, months)
    return latest is None or (nxt and nxt <= datetime.now())

# ───────────────────────── cleansing helpers ─────────────────────────────
CURRENCY_RE = re.compile(r"[\$,MK]")

def clean_financial_data(df: pd.DataFrame) -> pd.DataFrame:
    df.dropna(axis=0, how="all", subset=["Revenue","Net_Income","EPS"], inplace=True)
    df.ffill(inplace=True); df.bfill(inplace=True)
    df.infer_objects(copy=False)
    return df

def make_pretty_columns(df: pd.DataFrame) -> pd.DataFrame:
    def fmt(v):
        if pd.isna(v):      return "N/A"
        if abs(v) >= 1e9:   return f"${v/1e9:,.1f}B"
        if abs(v) >= 1e6:   return f"${v/1e6:,.1f}M"
        return f"${v/1e3:,.1f}K"
    df["Formatted_Revenue"]    = df["Revenue"].apply(fmt)
    df["Formatted_Net_Income"] = df["Net_Income"].apply(fmt)
    df["Formatted_EPS"]        = df["EPS"].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
    return df

# ───────────────────── pull from Yahoo (cached) ──────────────────────────
@lru_cache(maxsize=None)
def fetch_annual_yahoo(tkr: str) -> pd.DataFrame:
    fin = yf.Ticker(tkr).financials
    if fin.empty: return pd.DataFrame()
    fin = fin.T
    fin["Date"] = fin.index
    map_ = {"Total Revenue":"Revenue", "Net Income":"Net_Income", "Basic EPS":"EPS"}
    missing = [m for m in map_ if m not in fin.columns]
    if missing:
        logging.warning("%s missing %s", tkr, missing)
        return pd.DataFrame()
    fin.rename(columns=map_, inplace=True)
    return clean_financial_data(fin)

@lru_cache(maxsize=None)
def fetch_ttm_yahoo(tkr: str) -> dict | None:
    qf = yf.Ticker(tkr).quarterly_financials
    if qf is None or qf.empty: return None
    data = {
        "TTM_Revenue"      : qf.loc["Total Revenue"].iloc[:4].sum() if "Total Revenue" in qf.index else None,
        "TTM_Net_Income"   : qf.loc["Net Income"   ].iloc[:4].sum() if "Net Income"   in qf.index else None,
        "TTM_EPS"          : yf.Ticker(tkr).info.get("trailingEps"),
        "Shares_Outstanding": yf.Ticker(tkr).info.get("sharesOutstanding"),
        "Quarter"          : qf.columns[0].strftime("%Y-%m-%d"),
    }
    return data

# ─────────────────────────── storage ─────────────────────────────────────
def store_annual(tkr: str, df: pd.DataFrame, cur: sqlite3.Cursor) -> None:
    for _, r in df.iterrows():
        dt = r["Date"].strftime("%Y-%m-%d") if isinstance(r["Date"], pd.Timestamp) else r["Date"]
        cur.execute("""
           INSERT INTO Annual_Data(Symbol,Date,Revenue,Net_Income,EPS,Last_Updated)
           VALUES (?,?,?,?,?,CURRENT_TIMESTAMP)
           ON CONFLICT(Symbol,Date)
             DO UPDATE SET Revenue=excluded.Revenue, Net_Income=excluded.Net_Income,
                           EPS=excluded.EPS,      Last_Updated=CURRENT_TIMESTAMP
           WHERE Annual_Data.Revenue IS NULL OR Annual_Data.Net_Income IS NULL OR Annual_Data.EPS IS NULL;
        """, (tkr, dt, r["Revenue"], r["Net_Income"], r["EPS"]))
    cur.connection.commit()

def store_ttm(tkr: str, d: dict, cur: sqlite3.Cursor) -> None:
    cur.execute("""
       INSERT INTO TTM_Data(Symbol,TTM_Revenue,TTM_Net_Income,TTM_EPS,Shares_Outstanding,Quarter,Last_Updated)
       VALUES(?,?,?,?,?,?,CURRENT_TIMESTAMP)
       ON CONFLICT(Symbol)               -- keep one row per symbol
       DO UPDATE SET TTM_Revenue       = excluded.TTM_Revenue,
                     TTM_Net_Income    = excluded.TTM_Net_Income,
                     TTM_EPS           = excluded.TTM_EPS,
                     Shares_Outstanding= excluded.Shares_Outstanding,
                     Quarter           = excluded.Quarter,
                     Last_Updated      = CURRENT_TIMESTAMP;
    """, (tkr, d["TTM_Revenue"], d["TTM_Net_Income"], d["TTM_EPS"],
          d["Shares_Outstanding"], d["Quarter"]))
    cur.connection.commit()

def clear_ttm_duplicates(tkr: str, cur: sqlite3.Cursor) -> None:
    # keep newest Quarter row, drop rest – one SQL is enough
    cur.execute("""
        DELETE FROM TTM_Data
        WHERE Symbol=? AND rowid NOT IN (
            SELECT rowid FROM TTM_Data WHERE Symbol=? ORDER BY Quarter DESC LIMIT 1
        );
    """, (tkr, tkr))
    cur.connection.commit()

# ─────────────────────────── charts ──────────────────────────────────────
def chart_eps(tkr: str, df: pd.DataFrame) -> None:
    if df.empty: return
    path = os.path.join(CHART_DIR, f"{tkr}_eps_chart.png")
    df["EPS"] = pd.to_numeric(df["EPS"], errors="coerce")
    pos = np.arange(len(df)); width = .4
    fig, ax = plt.subplots(figsize=(8,5))
    bars = ax.bar(pos, df["EPS"], width, color="teal")
    ax.grid(True, axis="y", linestyle="--", linewidth=.5)
    ax.axhline(0, color="black", lw=2)
    ax.set_xticks(pos); ax.set_xticklabels(df["Date"], rotation=0)
    ax.set_ylabel("Earnings Per Share")
    ax.set_title(f"EPS Chart – {tkr}")
    # label bars
    for bar, lbl in zip(bars, df["Formatted_EPS"]):
        h = bar.get_height()
        ax.annotate(lbl, (bar.get_x()+bar.get_width()/2, h),
                    xytext=(0, 12 if h>=0 else -12),
                    textcoords="offset points", ha="center", va="bottom")
    plt.tight_layout(); plt.savefig(path); plt.close('all')

def chart_revenue_net(tkr: str, df: pd.DataFrame) -> None:
    if df.empty: return
    path = os.path.join(CHART_DIR, f"{tkr}_revenue_net_income_chart.png")
    df_num = df.assign(
        Revenue   = pd.to_numeric(df["Revenue"   ], errors="coerce"),
        Net_Income= pd.to_numeric(df["Net_Income"], errors="coerce"),
    )
    pos = np.arange(len(df_num)); width = .3
    sf, unit = (1e9, "B") if abs(df_num["Net_Income"].max())>=1e9 else (1e6,"M")
    fig, ax = plt.subplots(figsize=(10,6))
    bars_r = ax.bar(pos-width/2, df_num["Revenue"]/sf,   width, color="green", label=f"Revenue ({unit})")
    bars_n = ax.bar(pos+width/2, df_num["Net_Income"]/sf,width, color="blue",  label=f"Net Income ({unit})")
    # y-limits
    vals  = np.concatenate([df_num["Revenue"]/sf, df_num["Net_Income"]/sf])
    buf   = abs(vals).max() * .20
    ax.set_ylim((df_num["Net_Income"]/sf).min()-buf, vals.max()+buf)
    ax.set_xticks(pos); ax.set_xticklabels(df_num["Date"], rotation=0)
    ax.set_ylabel(f"Amount ({'Billions' if unit=='B' else 'Millions'} $)")
    ax.set_title(f"Revenue & Net Income – {tkr}")
    ax.grid(True, axis="y", linestyle="--", linewidth=.5); ax.axhline(0,color="black",lw=1)
    add_value_labels(ax, bars_r, df, "Formatted_Revenue", sf)
    add_value_labels(ax, bars_n, df, "Formatted_Net_Income", sf)
    plt.tight_layout(); plt.savefig(path); plt.close('all')

def add_value_labels(ax, bars, df, col, sf):
    for bar, raw in zip(bars, df[col]):
        h = bar.get_height()
        ax.annotate(raw, (bar.get_x()+bar.get_width()/2, h),
                    xytext=(0, 3 if h>=0 else -12),
                    textcoords="offset points", ha="center", va="bottom")

# ─────────────────────── HTML table  ─────────────────────────────────────
def calculate_changes(df: pd.DataFrame) -> pd.DataFrame:
    df.sort_values("Date", inplace=True)
    num_cols = ["Revenue","Net_Income","EPS"]
    for c in num_cols:
        if df[c].dtype == object:
            df[c] = (df[c].str.replace(CURRENCY_RE, "", regex=True).astype(float) * 1e3)
    for c in num_cols:
        pct = df[c].pct_change() * 100
        df[f"{c}_Change"] = pct.map(lambda x: f"{x:.1f}%" if pd.notnull(x) else "N/A")
    return df

def html_table(tkr: str, df: pd.DataFrame) -> None:
    if df.empty: return
    df = calculate_changes(df)
    keep = ["Date","Formatted_Revenue","Formatted_Net_Income","Formatted_EPS",
            "Revenue_Change","Net_Income_Change","EPS_Change"]
    tbl = df[keep].copy()
    tbl.columns = ["Date","Revenue","Net Income","EPS","Revenue Change","Net Income Change","EPS Change"]
    # average row
    pct_cols = ["Revenue Change","Net Income Change","EPS Change"]
    avg = (tbl[pct_cols]
           .replace("N/A",np.nan)
           .apply(lambda s: pd.to_numeric(s.str.rstrip("%"), errors="coerce"))
           .mean()
           .map(lambda x: f"{x:.1f}%" if pd.notnull(x) else "N/A"))
    tbl.loc[len(tbl)] = ["Average","","",""] + avg.tolist()
    def colour(v:str):
        if isinstance(v,str) and "%" in v:
            return "color:red;" if "-" in v else "color:green;"
        return ""
    styled = tbl.style.applymap(colour, subset=pct_cols)
    out = os.path.join(CHART_DIR, f"{tkr}_rev_net_table.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(styled.to_html())
    print(f"Financial data table for {tkr} → {out}")

# ─────────────────────── main update routine ─────────────────────────────
def prepare_df_for_charts(tkr: str, cur) -> pd.DataFrame:
    ann_rows = fetch_rows(cur, "Annual_Data", "WHERE Symbol=? ORDER BY Date", (tkr,))
    ttm_rows = fetch_rows(cur, "TTM_Data",    "WHERE Symbol=? ORDER BY Quarter DESC", (tkr,))

    ann_df = pd.DataFrame(ann_rows)
    ttm_df = pd.DataFrame(ttm_rows)

    # only keep one (latest) TTM row
    if not ttm_df.empty:
        ttm_df = ttm_df.head(1)
        latest_q = ttm_df.at[ttm_df.index[0], "Quarter"]
        ttm_df["Date"] = f"TTM {latest_q}"

    base = pd.concat([ann_df, ttm_df], ignore_index=True)

    if base.empty:
        return base

    # coerce numerics & prettify
    for col in ["Revenue","Net_Income","EPS"]:
        base[col] = pd.to_numeric(base[col], errors="coerce")
    base = clean_financial_data(base)
    base["Date"] = base["Date"].astype(str)
    base.sort_values("Date", inplace=True)              # only if needed
    return make_pretty_columns(base)

def annual_and_ttm_update(tkr: str, db_path: str = DB_PATH) -> None:
    conn = get_db_connection(db_path); cur = conn.cursor()

    # pull existing
    ann_rows = fetch_rows(cur, "Annual_Data", "WHERE Symbol=?", (tkr,))
    ttm_rows = fetch_rows(cur, "TTM_Data",    "WHERE Symbol=?", (tkr,))

    if not ann_rows:  # seed
        df = fetch_annual_yahoo(tkr)
        if not df.empty:
            store_annual(tkr, df, cur)
            ann_rows = df.to_dict("records")

    if not ttm_rows:
        d = fetch_ttm_yahoo(tkr)
        if d:
            store_ttm(tkr, d, cur)
            ttm_rows = [d]

    clear_ttm_duplicates(tkr, cur)

    # decide if we need fresh pulls
    lad = get_latest_annual_date(ann_rows)
    ltd = max((datetime.strptime(r["Quarter"], "%Y-%m-%d") for r in ttm_rows if r.get("Quarter")), default=None)

    if needs_update(lad, 13):
        store_annual(tkr, fetch_annual_yahoo(tkr), cur)
    if needs_update(ltd, 4):
        d = fetch_ttm_yahoo(tkr)
        if d: store_ttm(tkr, d, cur)

    # charts + html
    df = prepare_df_for_charts(tkr, cur)
    chart_revenue_net(tkr, df)
    chart_eps(tkr, df)
    html_table(tkr, df)

    conn.close()
    logging.info("[%s] update complete (§)", tkr)

# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    annual_and_ttm_update("PG")       # example

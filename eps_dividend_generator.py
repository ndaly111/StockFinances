"""
eps_dividend_generator.py
-------------------------
Pulls dividend history from Yahoo, stores per-year totals plus TTM values in
SQLite, and produces an EPS + Dividend bar-chart for each ticker.

If a company has **no dividends at all in the past 10 years**, it just writes a
tiny PNG reading “no dividend” so dashboards don’t break.
"""

import os
import sqlite3
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

DB_PATH   = "Stock Data.db"
CHART_DIR = "charts"

# ──────────────────────────────────────────────────────────────────────────────
#  Schema helpers
# ──────────────────────────────────────────────────────────────────────────────
def _ensure_schema(conn: sqlite3.Connection) -> None:
    """
    • Creates the two tables if they don’t exist.
    • Adds a UNIQUE index on Symbol for legacy DBs that were created without one.
    """
    cur = conn.cursor()

    # Always create the tables with the modern schema if they’re missing
    cur.executescript("""
    CREATE TABLE IF NOT EXISTS Dividends (
        ticker    TEXT,
        year      INTEGER,
        dividend  REAL,
        PRIMARY KEY (ticker, year)
    );
    CREATE TABLE IF NOT EXISTS TTM_Data (
        Symbol        TEXT PRIMARY KEY,        -- one row per ticker
        TTM_Dividend  REAL,
        TTM_EPS       REAL,
        Last_Updated  TEXT
    );
    """)

    # Legacy safety: make sure Symbol has a UNIQUE constraint
    cur.execute("""
        PRAGMA table_info(TTM_Data)
    """)
    cols = {row[1]: row for row in cur.fetchall()}          # name → column info
    if 'Symbol' in cols and cols['Symbol'][5] == 0:          # col[5] == 1 if PK
        # Add UNIQUE index only if one doesn’t already exist
        cur.execute("""
            PRAGMA index_list(TTM_Data)
        """)
        indexes = [row[1] for row in cur.fetchall()]         # index names
        if 'TTM_Symbol_uq' not in indexes:
            cur.execute("CREATE UNIQUE INDEX TTM_Symbol_uq ON TTM_Data(Symbol);")
            print("ℹ️  Added UNIQUE index on TTM_Data.Symbol for legacy DB.")

    conn.commit()

# ──────────────────────────────────────────────────────────────────────────────
#  Upsert helpers
# ──────────────────────────────────────────────────────────────────────────────
def _upsert_dividend_year(cur, tic: str, yr: int, amt: float) -> None:
    """
    Per-year totals. Uses the PK (ticker,year) so a re-run merely overwrites
    the same row rather than creating duplicates.
    """
    cur.execute("""
        INSERT INTO Dividends(ticker, year, dividend)
        VALUES (?,?,?)
        ON CONFLICT(ticker,year) DO UPDATE
          SET dividend = excluded.dividend;
    """, (tic, yr, amt))

def _update_ttm_div(cur, tic: str, ttm_amt: float) -> None:
    """
    Legacy-proof upsert: if the row exists → UPDATE, else → INSERT.
    Avoids the ON CONFLICT clause that blew up on old DBs.
    """
    ts = dt.datetime.utcnow().strftime("%F %T")

    cur.execute("SELECT 1 FROM TTM_Data WHERE Symbol=?;", (tic,))
    if cur.fetchone():            # row already there → update
        cur.execute("""
            UPDATE TTM_Data
               SET TTM_Dividend = ?,
                   Last_Updated = ?
             WHERE Symbol = ?;
        """, (ttm_amt, ts, tic))
    else:                         # no row yet → insert
        cur.execute("""
            INSERT INTO TTM_Data(Symbol, TTM_Dividend, Last_Updated)
            VALUES (?,?,?);
        """, (tic, ttm_amt, ts))

# ──────────────────────────────────────────────────────────────────────────────
#  Chart helpers
# ──────────────────────────────────────────────────────────────────────────────
def _make_no_dividend_placeholder(tic: str) -> str:
    """
    Writes a 400×200 PNG that says “no dividend” so dashboards don’t break.
    """
    path = os.path.join(CHART_DIR, f"{tic}_eps_dividend_forecast.png")
    os.makedirs(CHART_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4, 2), dpi=100)
    ax.text(0.5, 0.5, "no dividend",
            ha="center", va="center", fontsize=12)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    print(f"📄 {tic}: No dividends – placeholder saved.")
    return path

def _build_chart(tic: str, conn: sqlite3.Connection) -> str:
    """
    Assembles trailing EPS, forward EPS, and dividends → bar chart.
    Assumes *at least one* dividend exists in the last 10 years.
    """
    cur = conn.cursor()
    path = os.path.join(CHART_DIR, f"{tic}_eps_dividend_forecast.png")

    # Trailing EPS (max 10 yrs)
    cur.execute("""
        SELECT Date, EPS
          FROM Annual_Data
         WHERE Symbol = ?
         ORDER BY Date ASC
         LIMIT 10;
    """, (tic,))
    trailing = [(int(d[:4]), float(eps) if eps is not None else 0.0)
                for d, eps in cur.fetchall()]

    # Forward EPS (max 3 yrs)
    cur.execute("""
        SELECT Date, ForwardEPS
          FROM ForwardFinancialData
         WHERE Ticker = ?
         ORDER BY Date ASC
         LIMIT 3;
    """, (tic,))
    forward = [(int(d[:4]), float(v)) for d, v in cur.fetchall()]

    # Dividend map for those trailing years
    yrs = [y for y, _ in trailing]
    q   = ",".join("?" * len(yrs)) if yrs else "NULL"
    cur.execute(f"""
        SELECT year, dividend
          FROM Dividends
         WHERE ticker = ?
           AND year IN ({q});
    """, (tic, *yrs))
    div_map = {int(y): float(d) for y, d in cur.fetchall()}

    # TTM row
    cur.execute("""
        SELECT TTM_EPS, TTM_Dividend
          FROM TTM_Data
         WHERE Symbol = ?;
    """, (tic,))
    ttm_eps, ttm_div = cur.fetchone() or (0.0, 0.0)

    # Assemble data series -----------------------------------------------------
    labels, eps_hist, eps_fwd, divs = [], [], [], []

    for yr, eps in trailing:
        labels.append(str(yr))
        eps_hist.append(eps)
        eps_fwd.append(0)
        divs.append(div_map.get(yr, 0.0))

    labels.append("TTM")
    eps_hist.append(ttm_eps)
    eps_fwd.append(0)
    divs.append(ttm_div)

    for yr, fwd in forward:
        labels.append(str(yr))
        eps_hist.append(0)
        eps_fwd.append(fwd)
        divs.append(0.0)

    # Plot ---------------------------------------------------------------------
    x = range(len(labels))
    w = 0.25
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

    b1 = ax.bar([i - w for i in x], eps_hist, w, label="Trailing EPS")
    b2 = ax.bar(x,                 eps_fwd, w, label="Forecast EPS", color="#70a6ff")
    b3 = ax.bar([i + w for i in x], divs,    w, label="Dividend",     color="orange")

    # Numbers on bars
    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(f"{h:.2f}",
                            xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel("USD per share")
    ax.set_title(f"{tic} – EPS & Dividend")
    ax.legend()
    plt.tight_layout()

    os.makedirs(CHART_DIR, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"💾 {tic}: chart saved → {path}")
    return path

# ──────────────────────────────────────────────────────────────────────────────
#  Main driver
# ──────────────────────────────────────────────────────────────────────────────
def generate_eps_dividend(tickers, db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    _ensure_schema(conn)
    cur = conn.cursor()

    os.makedirs(CHART_DIR, exist_ok=True)
    outputs = {}

    for tic in tickers:
        print(f"🔧 Processing {tic}")

        # 1) Fetch raw dividend history ----------------------------------------
        try:
            raw_divs = yf.Ticker(tic).dividends
        except Exception as exc:
            print(f"⚠️  Couldn’t fetch dividends for {tic}: {exc}")
            raw_divs = pd.Series(dtype="float64")

        if raw_divs.empty or raw_divs.sum() == 0:
            # No dividends at all → placeholder PNG
            outputs[tic] = _make_no_dividend_placeholder(tic)
            continue

        # 2) Store yearly totals & TTM -----------------------------------------
        raw_divs.index = pd.to_datetime(raw_divs.index, utc=True).tz_localize(None)

        for yr, amt in raw_divs.groupby(raw_divs.index.year).sum().items():
            _upsert_dividend_year(cur, tic, int(yr), float(amt))

        one_year_ago = dt.datetime.utcnow() - dt.timedelta(days=365)
        ttm_amount   = float(raw_divs[raw_divs.index >= one_year_ago].sum())
        _update_ttm_div(cur, tic, ttm_amount)
        conn.commit()

        # 3) Build full EPS + Dividend chart -----------------------------------
        try:
            outputs[tic] = _build_chart(tic, conn)
        except Exception as exc:
            print(f"❌ Error building chart for {tic}: {exc}")
            outputs[tic] = _make_no_dividend_placeholder(tic)

    conn.close()
    return outputs

# Convenience wrapper for your existing CLI util
def eps_dividend_generator():
    from ticker_manager import read_tickers   # local helper you already have
    return generate_eps_dividend(read_tickers("tickers.csv"))

if __name__ == "__main__":
    print(eps_dividend_generator())

import os
import sqlite3
import datetime as dt
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

DB_PATH   = "Stock Data.db"
CHART_DIR = "charts"

# ───────────────────────────  DB helpers  ──────────────────────────
def _ensure_schema(conn):
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS Dividends(
            ticker TEXT,
            year   INTEGER,
            dividend REAL,
            PRIMARY KEY(ticker, year)
        );
    """)
    cur.execute("PRAGMA table_info(TTM_Data);")
    cols = [c[1].lower() for c in cur.fetchall()]
    if "ttm_dividend" not in cols:
        cur.execute("ALTER TABLE TTM_Data ADD COLUMN TTM_Dividend REAL;")
    conn.commit()

def _upsert_dividend_year(cur, tic, yr, val):
    cur.execute("""
        INSERT INTO Dividends(ticker, year, dividend)
        VALUES(?,?,?)
        ON CONFLICT(ticker,year) DO UPDATE
        SET dividend = excluded.dividend;
    """, (tic, yr, val))

def _update_ttm_div(cur, tic, val):
    ts = dt.datetime.utcnow().strftime("%F %T")
    cur.execute("""
        UPDATE TTM_Data
           SET TTM_Dividend = ?, Last_Updated = ?
         WHERE Symbol = ?;
    """, (val, ts, tic))
    if cur.rowcount == 0:          # row missing → insert stub
        cur.execute("""
            INSERT INTO TTM_Data(Symbol, TTM_Dividend, Last_Updated)
            VALUES(?,?,?);
        """, (tic, val, ts))

# ─────────────────────  Chart builder  ─────────────────────────────
def _build_chart(tic: str, conn: sqlite3.Connection) -> str:
    cur = conn.cursor()

    # 10 trailing EPS (oldest→newest)
    cur.execute("""
        SELECT Date, EPS
          FROM Annual_Data
         WHERE Symbol = ?
         ORDER BY Date ASC
         LIMIT 10;
    """, (tic,))
    trailing = [(int(d[:4]), float(eps) if eps is not None else 0.0)
                for d, eps in cur.fetchall()]

    # 3 forward EPS
    cur.execute("""
        SELECT Date, ForwardEPS
          FROM ForwardFinancialData
         WHERE Ticker = ?
         ORDER BY Date ASC
         LIMIT 3;
    """, (tic,))
    forward = [(int(d[:4]), float(v)) for d, v in cur.fetchall()]

    # dividend map for trailing years
    years = [y for y, _ in trailing]
    div_map = {}
    if years:
        q = ",".join("?" * len(years))
        cur.execute(f"""
            SELECT year, dividend
              FROM Dividends
             WHERE ticker = ? AND year IN ({q});
        """, (tic, *years))
        div_map = {int(y): float(d) for y, d in cur.fetchall()}

    # latest TTM values
    cur.execute("""
        SELECT TTM_EPS, TTM_Dividend
          FROM TTM_Data
         WHERE Symbol = ?
         ORDER BY Last_Updated DESC
         LIMIT 1;
    """, (tic,))
    ttm_eps, ttm_div = cur.fetchone() or (0.0, 0.0)

    # assemble series
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

    # plot
    x = range(len(labels)); w = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar([i - w for i in x], eps_hist, w, label="Trailing EPS")
    ax.bar(x,               eps_fwd,  w, label="Forecast EPS", color="#70a6ff")
    ax.bar([i + w for i in x], divs,  w, label="Dividend",     color="orange")

    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel("USD per share")
    ax.set_title(f"{tic} – EPS (Actual & Forecast) vs Dividend")
    ax.legend()

    os.makedirs(CHART_DIR, exist_ok=True)
    path = os.path.join(CHART_DIR, f"{tic}_eps_dividend_forecast.png")
    plt.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)
    return path

# ───────────────────  Public entry point  ─────────────────────────
def generate_eps_dividend(tickers,
                          db_path: str = DB_PATH,
                          chart_dir: str = CHART_DIR):
    conn = sqlite3.connect(db_path)
    _ensure_schema(conn)
    cur  = conn.cursor()
    out  = {}

    for tic in tickers:
        # 1) refresh dividends -----------------------------
        divs = yf.Ticker(tic).dividends
        if divs.empty or divs.sum() == 0:            # no dividend at all
            os.makedirs(chart_dir, exist_ok=True)
            path = os.path.join(chart_dir, f"{tic}_eps_dividend_forecast.png")
            fig, ax = plt.subplots(figsize=(4, 2))
            ax.text(0.5, 0.5, "no dividend",
                    ha="center", va="center", fontsize=12)
            ax.axis("off")
            fig.savefig(path, dpi=100, bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            out[tic] = path
            continue

        divs.index = pd.to_datetime(divs.index, utc=True).tz_localize(None)
        for yr, amt in divs.groupby(divs.index.year).sum().items():
            _upsert_dividend_year(cur, tic, int(yr), float(amt))

        last_year_total = divs[divs.index >=
                               (dt.datetime.utcnow() - dt.timedelta(days=365))].sum()
        _update_ttm_div(cur, tic, float(last_year_total))
        conn.commit()

        # 2) build chart from DB data ----------------------
        out[tic] = _build_chart(tic, conn)
        conn.commit()

    conn.close()
    return out

# helper for other modules
def eps_dividend_generator():
    from ticker_manager import read_tickers
    return generate_eps_dividend(read_tickers("tickers.csv"))

if __name__ == "__main__":
    print(eps_dividend_generator())

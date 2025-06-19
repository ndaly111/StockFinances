import os
import sqlite3
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

DB_PATH = "Stock Data.db"
CHART_DIR = "charts"

def _ensure_schema(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.executescript("""
    CREATE TABLE IF NOT EXISTS Dividends(
        ticker TEXT,
        year INTEGER,
        dividend REAL,
        PRIMARY KEY(ticker,year)
    );
    CREATE TABLE IF NOT EXISTS TTM_Data(
        Symbol        TEXT PRIMARY KEY,
        TTM_Dividend  REAL,
        TTM_EPS       REAL,
        Last_Updated  TEXT
    );
    """)
    conn.commit()

def _upsert_dividend_year(cur, tic: str, yr: int, amt: float):
    cur.execute("""
        INSERT INTO Dividends(ticker,year,dividend)
        VALUES (?,?,?)
        ON CONFLICT(ticker,year) DO UPDATE
          SET dividend=excluded.dividend;
    """, (tic, yr, amt))

def _update_ttm_div(cur, tic: str, last365: float):
    ts = dt.datetime.utcnow().strftime("%F %T")
    cur.execute("""
        INSERT INTO TTM_Data(Symbol,TTM_Dividend,Last_Updated)
        VALUES (?,?,?)
        ON CONFLICT(Symbol) DO UPDATE
          SET TTM_Dividend=excluded.TTM_Dividend,
              Last_Updated=excluded.Last_Updated;
    """, (tic, last365, ts))

def _build_chart(tic: str, conn: sqlite3.Connection) -> str:
    cur = conn.cursor()
    path = os.path.join(CHART_DIR, f"{tic}_eps_dividend_forecast.png")

    # ── fetch trailing EPS ───────────────────────────────────
    cur.execute(
        "SELECT Date, EPS FROM Annual_Data WHERE Symbol=? ORDER BY Date ASC LIMIT 10;",
        (tic,)
    )
    trailing = [
        (int(d[:4]), float(eps) if eps is not None else 0.0)
        for d, eps in cur.fetchall()
    ]

    # ── fetch forward EPS ────────────────────────────────────
    cur.execute(
        "SELECT Date, ForwardEPS FROM ForwardFinancialData WHERE Ticker=? ORDER BY Date ASC LIMIT 3;",
        (tic,)
    )
    forward = [
        (int(d[:4]), float(v))
        for d, v in cur.fetchall()
    ]

    # ── load dividends from DB ──────────────────────────────
    years = [y for y, _ in trailing]
    if years:
        q = ",".join("?" * len(years))
        cur.execute(
            f"SELECT year,dividend FROM Dividends WHERE ticker=? AND year IN ({q});",
            (tic, *years)
        )
        div_map = {int(y): float(d) for y, d in cur.fetchall()}
    else:
        div_map = {}

    # ── load TTM data ───────────────────────────────────────
    cur.execute("SELECT TTM_EPS,TTM_Dividend FROM TTM_Data WHERE Symbol=?;", (tic,))
    ttm_eps, ttm_div = cur.fetchone() or (0.0, 0.0)

    # ── fallback if truly no dividends ever ─────────────────
    if (not div_map or all(v == 0.0 for v in div_map.values())) and ttm_div == 0.0:
        os.makedirs(CHART_DIR, exist_ok=True)
        plt.figure(figsize=(4, 2), dpi=100)
        plt.text(0.5, 0.5, "no dividend", ha="center", va="center", fontsize=12)
        plt.axis("off")
        plt.savefig(path, bbox_inches="tight", pad_inches=0)
        plt.close()
        print(f"📄 {tic}: No dividends – placeholder saved.")
        return path

    # ── assemble the three series ───────────────────────────
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

    # ── plot ─────────────────────────────────────────────────
    x = range(len(labels))
    w = 0.25
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    bars1 = ax.bar([i - w for i in x], eps_hist, w, label="Trailing EPS")
    bars2 = ax.bar(x,             eps_fwd,  w, label="Forecast EPS", color="#70a6ff")
    bars3 = ax.bar([i + w for i in x], divs,  w, label="Dividend",      color="orange")

    # add data labels
    for bars in (bars1, bars2, bars3):
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(
                    f"{h:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=8
                )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel("USD per share")
    ax.set_title(f"{tic} – EPS & Dividend")
    ax.legend()

    os.makedirs(CHART_DIR, exist_ok=True)
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"💾 {tic}: chart saved → {path}")
    return path

def generate_eps_dividend(tickers, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    _ensure_schema(conn)
    cur = conn.cursor()
    os.makedirs(CHART_DIR, exist_ok=True)
    out = {}

    for tic in tickers:
        print(f"🔧 Processing {tic}")
        # ── fetch & store dividends & TTM ───────────────
        try:
            divs = yf.Ticker(tic).dividends
            if not divs.empty and float(divs.sum()) > 0:
                divs.index = pd.to_datetime(divs.index, utc=True).tz_localize(None)
                for yr, amt in divs.groupby(divs.index.year).sum().items():
                    _upsert_dividend_year(cur, tic, int(yr), float(amt))
                one_yr_ago = dt.datetime.utcnow() - dt.timedelta(days=365)
                ttm = float(divs[divs.index >= one_yr_ago].sum())
                _update_ttm_div(cur, tic, ttm)
                conn.commit()
        except Exception as e:
            print(f"⚠️ Warning: failed to fetch/store dividends for {tic}: {e}")

        # ── build or fallback chart ─────────────────────
        try:
            out[tic] = _build_chart(tic, conn)
        except Exception as e:
            print(f"❌ Error building chart for {tic}: {e}")

    conn.close()
    return out

def eps_dividend_generator():
    from ticker_manager import read_tickers
    return generate_eps_dividend(read_tickers("tickers.csv"))

if __name__ == "__main__":
    print(eps_dividend_generator())

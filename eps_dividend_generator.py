"""
eps_dividend_generator.py  –  v2  (event-level dividends)

• If a ticker has paid **no dividend at all for 10 years** we save a
  “no dividend” placeholder PNG so downstream HTML still has an image.

• Otherwise we:
  1) Insert/ignore every ex-date payment in DividendEvents
  2) Re-compute the last-365-days sum and upsert it into DividendTTM
  3) Build the full EPS / Dividend chart

Run directly or call generate_eps_dividend([...]).
"""

import os, sqlite3, datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

DB_PATH   = "Stock Data.db"
CHART_DIR = "charts"

# ─────────────────────────────────────────────────────────────────────────────
#  1. Schema helpers
# ─────────────────────────────────────────────────────────────────────────────
def _ensure_schema(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.executescript(
        """
        /* one immutable row per ex-date */
        CREATE TABLE IF NOT EXISTS DividendEvents(
            ticker   TEXT,
            ex_date  TEXT,        -- ISO date
            amount   REAL,
            PRIMARY KEY(ticker, ex_date)
        );

        /* one snapshot row per ticker (refreshable) */
        CREATE TABLE IF NOT EXISTS DividendTTM(
            ticker        TEXT PRIMARY KEY,
            ttm_dividend  REAL,
            last_updated  TEXT
        );
        """
    )
    conn.commit()


# ─────────────────────────────────────────────────────────────────────────────
#  2. DB write helpers
# ─────────────────────────────────────────────────────────────────────────────
def _insert_dividend_event(cur, tic: str, ex_date: dt.date, amt: float):
    cur.execute(
        """
        INSERT OR IGNORE INTO DividendEvents(ticker, ex_date, amount)
        VALUES (?,?,?)
        """,
        (tic, ex_date.isoformat(), amt),
    )


def _refresh_ttm_snapshot(cur, tic: str):
    today      = dt.datetime.utcnow()
    one_year   = today - dt.timedelta(days=365)

    cur.execute(
        """
        SELECT COALESCE(SUM(amount),0)
        FROM DividendEvents
        WHERE ticker=? AND ex_date>=?
        """,
        (tic, one_year.date().isoformat()),
    )
    ttm_sum = float(cur.fetchone()[0])

    cur.execute(
        """
        INSERT INTO DividendTTM(ticker, ttm_dividend, last_updated)
        VALUES (?,?,?)
        ON CONFLICT(ticker) DO UPDATE
          SET ttm_dividend=excluded.ttm_dividend,
              last_updated=excluded.last_updated;
        """,
        (tic, ttm_sum, today.strftime("%F %T")),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Chart helpers
# ─────────────────────────────────────────────────────────────────────────────
def _make_no_dividend_placeholder(tic: str) -> str:
    path = os.path.join(CHART_DIR, f"{tic}_eps_dividend_forecast.png")
    os.makedirs(CHART_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4, 2), dpi=100)
    ax.text(0.5, 0.5, "no dividend", ha="center", va="center", fontsize=12)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    print(f"📄 {tic}: no-dividend placeholder saved.")
    return path


def _build_chart(tic: str, conn: sqlite3.Connection) -> str:
    """
    Builds the EPS + Dividend chart.
    Assumes at least one dividend in the last 10 yrs is present.
    """
    cur  = conn.cursor()
    path = os.path.join(CHART_DIR, f"{tic}_eps_dividend_forecast.png")

    # -- trailing EPS (≤10 yrs) ------------------------------------------------
    cur.execute(
        """
        SELECT Date, EPS
        FROM   Annual_Data
        WHERE  Symbol=?
        ORDER  BY Date ASC
        LIMIT  10;
        """,
        (tic,),
    )
    trailing = [(int(d[:4]), float(eps) if eps else 0.0) for d, eps in cur.fetchall()]

    # -- forward EPS (≤3 yrs) --------------------------------------------------
    cur.execute(
        """
        SELECT Date, ForwardEPS
        FROM   ForwardFinancialData
        WHERE  Ticker=?
        ORDER  BY Date ASC
        LIMIT  3;
        """,
        (tic,),
    )
    forward = [(int(d[:4]), float(v)) for d, v in cur.fetchall()]

    # -- yearly dividend totals (same years as trailing EPS) -------------------
    years = [yr for yr, _ in trailing]
    qmarks = ",".join("?" * len(years))
    cur.execute(
        f"""
        SELECT strftime('%Y', ex_date) AS yr, SUM(amount)
        FROM   DividendEvents
        WHERE  ticker=? AND yr IN ({qmarks})
        GROUP  BY yr;
        """,
        (tic, *years),
    )
    div_map = {int(yr): float(tot) for yr, tot in cur.fetchall()}

    # -- TTM snapshot ----------------------------------------------------------
    cur.execute(
        "SELECT ttm_dividend, last_updated FROM DividendTTM WHERE ticker=?;",
        (tic,),
    )
    ttm_div, _updated = cur.fetchone() or (0.0, None)

    cur.execute(
        "SELECT TTM_EPS FROM TTM_Data WHERE Symbol=?;",
        (tic,),
    )
    ttm_eps = cur.fetchone()
    ttm_eps = float(ttm_eps[0]) if ttm_eps else 0.0

    # -- assemble series -------------------------------------------------------
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

    # -- plot ------------------------------------------------------------------
    x = range(len(labels))
    w = 0.25
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    b1 = ax.bar([i - w for i in x], eps_hist, w, label="Trailing EPS")
    b2 = ax.bar(x, eps_fwd, w, label="Forecast EPS", color="#70a6ff")
    b3 = ax.bar([i + w for i in x], divs, w, label="Dividend", color="orange")

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(
                    f"{h:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

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


# ─────────────────────────────────────────────────────────────────────────────
# 4. Main driver
# ─────────────────────────────────────────────────────────────────────────────
def generate_eps_dividend(tickers, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    _ensure_schema(conn)
    cur  = conn.cursor()
    os.makedirs(CHART_DIR, exist_ok=True)

    out = {}
    for tic in tickers:
        print(f"🔧 Processing {tic}")

        # 1 · Grab every raw dividend event from Yahoo
        try:
            s = yf.Ticker(tic).dividends   # pandas Series (index = ex-date)
        except Exception as e:
            print(f"⚠️  {tic}: Yahoo fetch failed – {e}")
            s = pd.Series(dtype="float64")

        # 2 · Bail early if *zero* dividends in the last 10 years
        ten_years_ago = dt.datetime.utcnow() - dt.timedelta(days=365 * 10)
        recent = s[s.index >= ten_years_ago]
        if recent.empty or recent.sum() == 0:
            out[tic] = _make_no_dividend_placeholder(tic)
            continue

        # 3 · Insert events (idempotent thanks to PK)  & refresh TTM snapshot
        s.index = pd.to_datetime(s.index, utc=True).tz_localize(None)
        for ex_date, amt in s.items():
            _insert_dividend_event(cur, tic, ex_date.date(), float(amt))
        _refresh_ttm_snapshot(cur, tic)
        conn.commit()

        # 4 · Build the chart – fall back to placeholder if it blows up
        try:
            out[tic] = _build_chart(tic, conn)
        except Exception as e:
            print(f"❌  {tic}: chart build failed – {e}")
            out[tic] = _make_no_dividend_placeholder(tic)

    conn.close()
    return out


def eps_dividend_generator():
    from ticker_manager import read_tickers
    return generate_eps_dividend(read_tickers("tickers.csv"))


if __name__ == "__main__":
    print(eps_dividend_generator())

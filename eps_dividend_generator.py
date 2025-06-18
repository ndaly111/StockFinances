import os
import sqlite3
import datetime as dt
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

DB_PATH   = "Stock Data.db"
CHART_DIR = "charts"


def _ensure_schema(conn):
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS Dividends(
            ticker TEXT, year INTEGER, dividend REAL,
            PRIMARY KEY(ticker,year)
        );
    """)
    cur.execute("PRAGMA table_info(TTM_Data);")
    cols = [r[1].lower() for r in cur.fetchall()]
    if "ttm_dividend" not in cols:
        cur.execute("ALTER TABLE TTM_Data ADD COLUMN TTM_Dividend REAL;")
    conn.commit()


def _upsert_dividend_year(cur, tic, yr, val):
    cur.execute("""
        INSERT INTO Dividends(ticker,year,dividend)
        VALUES(?,?,?)
        ON CONFLICT(ticker,year) DO UPDATE SET dividend=excluded.dividend;
    """, (tic, yr, val))


def _update_ttm_div(cur, tic, val):
    timestamp = dt.datetime.utcnow().strftime("%F %T")
    cur.execute("""
        UPDATE TTM_Data
           SET TTM_Dividend=?, Last_Updated=?
         WHERE Symbol=?;
    """, (val, timestamp, tic))
    if cur.rowcount == 0:
        cur.execute("""
            INSERT INTO TTM_Data(Symbol,TTM_Dividend,Last_Updated)
            VALUES(?,?,?);
        """, (tic, val, timestamp))


def _build_chart(tic: str, conn: sqlite3.Connection, tkr: yf.Ticker) -> str:
    cur = conn.cursor()

    # ── fetch last 10 annual EPS ────────────────────────────
    cur.execute("""
        SELECT Date, EPS
          FROM Annual_Data
         WHERE Symbol=?
         ORDER BY Date DESC
         LIMIT 10;
    """, (tic,))
    raw = cur.fetchall()

    # collect trailing EPS, track missing years
    trailing = []
    missing_years = set()
    for date_str, eps in raw:
        yr = int(date_str[:4])
        if eps is None:
            missing_years.add(yr)
            trailing.append((yr, 0.0))
        else:
            trailing.append((yr, float(eps)))

    # ── back-fill missing EPS in one go ─────────────────────
    if missing_years:
        inc = tkr.get_income_stmt(freq="yearly")                  # correct freq
        # normalize columns to "YYYY"
        inc.columns = [str(c.year) for c in pd.to_datetime(inc.columns)]
        # normalize row labels
        inc.index = inc.index.str.strip()
        # detect an EPS row
        eps_rows = [r for r in inc.index if "eps" in r.lower()]
        if eps_rows:
            eps_label = eps_rows[0]
            for idx, (yr, _) in enumerate(trailing):
                s = str(yr)
                if s in inc.columns:
                    val = inc.at[eps_label, s]
                    if pd.notna(val):
                        cur.execute("""
                            UPDATE Annual_Data
                               SET EPS=?
                             WHERE Symbol=? AND Date LIKE ?;
                        """, (float(val), tic, f"{yr}-%"))
                        trailing[idx] = (yr, float(val))

    # ── fetch next 3 years of forecast EPS ──────────────────
    cur.execute("""
        SELECT Date, ForwardEPS
          FROM ForwardFinancialData
         WHERE Ticker=?
         ORDER BY Date ASC
         LIMIT 3;
    """, (tic,))
    forward = [(int(d[:4]), float(v)) for d, v in cur.fetchall()]

    # ── load historical dividends ───────────────────────────
    years = [y for y, _ in trailing]
    if years:
        q = ",".join("?" * len(years))
        cur.execute(f"""
            SELECT year, dividend
              FROM Dividends
             WHERE ticker=? AND year IN ({q});
        """, (tic, *years))
        div_map = {int(y): float(d) for y, d in cur.fetchall()}
    else:
        div_map = {}

    # ── latest TTM EPS & dividend ──────────────────────────
    cur.execute("""
        SELECT TTM_EPS, TTM_Dividend
          FROM TTM_Data
         WHERE Symbol=?
         ORDER BY Last_Updated DESC
         LIMIT 1;
    """, (tic,))
    ttm_eps, ttm_div = cur.fetchone() or (0, 0)

    # ── prepare plot series: chronological trailing → TTM → forecast
    labels, eps_hist, eps_fwd, divs = [], [], [], []

    for yr, val in reversed(trailing):   # oldest→newest
        labels.append(str(yr))
        eps_hist.append(val)
        eps_fwd.append(0)
        divs.append(div_map.get(yr, 0.0))

    labels.append("TTM")
    eps_hist.append(float(ttm_eps or 0))
    eps_fwd.append(0)
    divs.append(float(ttm_div or 0))

    for yr, val in forward:
        labels.append(str(yr))
        eps_hist.append(0)
        eps_fwd.append(val)
        divs.append(0.0)

    # ── plot ────────────────────────────────────────────────
    x = range(len(labels))
    w = 0.25
    fig, ax = plt.subplots(figsize=(10,6))
    ax.bar([i-w for i in x],     eps_hist, w, label="Trailing EPS")
    ax.bar(x,                    eps_fwd,  w, label="Forecast EPS", color="#70a6ff")
    ax.bar([i+w for i in x],      divs,   w, label="Dividend",     color="orange")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel("USD per share")
    ax.set_title(f"{tic} – EPS (Actual & Forecast) vs Dividend")
    ax.legend()

    # ── annotate current yield if possible ─────────────────
    try:
        hist = tkr.history(period="1d")
        price = hist["Close"].iloc[-1]
        # the year of the last trailing bar is labels[-len(forward)-2]
        last_year = int(labels[-len(forward)-2])
        last_div  = div_map.get(last_year, 0.0)
        ax.text(0.01, .95,
                f"Current Yield ≈ {last_div/price*100:0.2f}%",
                transform=ax.transAxes, va="top", fontsize=9)
    except Exception:
        pass

    os.makedirs(CHART_DIR, exist_ok=True)
    path = os.path.join(CHART_DIR, f"{tic}_eps_dividend_forecast.png")
    plt.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def generate_eps_dividend(tickers, db_path=DB_PATH, chart_dir=CHART_DIR):
    conn = sqlite3.connect(db_path)
    _ensure_schema(conn)
    cur = conn.cursor()
    out = {}

    for tic in tickers:
        tkr  = yf.Ticker(tic)
        divs = tkr.dividends

        # ── no-dividend fallback ───────────────────────────
        if divs.empty or float(divs.sum()) == 0:
            os.makedirs(chart_dir, exist_ok=True)
            path = os.path.join(chart_dir, f"{tic}_eps_dividend_forecast.png")
            fig, ax = plt.subplots(figsize=(4,2))
            ax.text(0.5, 0.5, "no dividend", ha="center", va="center", fontsize=12)
            ax.axis('off')
            fig.savefig(path, dpi=100, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            out[tic] = path
            conn.commit()
            continue

        # ── upsert yearly dividends & TTM ──────────────────
        divs.index = pd.to_datetime(divs.index, utc=True).tz_localize(None)
        for yr, amt in divs.groupby(divs.index.year).sum().items():
            _upsert_dividend_year(cur, tic, int(yr), float(amt))
        one_year_ago = dt.datetime.utcnow() - dt.timedelta(days=365)
        _update_ttm_div(cur, tic,
                        float(divs[divs.index >= one_year_ago].sum()))
        conn.commit()

        # ── build & save chart ────────────────────────────
        out[tic] = _build_chart(tic, conn, tkr)
        conn.commit()

    conn.close()
    return out


def eps_dividend_generator():
    from ticker_manager import read_tickers
    return generate_eps_dividend(read_tickers("tickers.csv"))


if __name__ == "__main__":
    print(eps_dividend_generator())

"""
EPS-Dividend module
─────────────────────────────────────────────
Call   generate_eps_dividend(ticker_list)
for each ticker it will:
    • ensure Dividends table & TTM_Dividend column exist
    • pull & store dividend history + TTM dividend (last 365 days, tz-handled)
    • back-fill missing EPS from Yahoo Finance (Diluted EPS)
    • create charts/{ticker}_eps_dividend_forecast.png
Returns dict {ticker: chart_path}
"""
import os, sqlite3, datetime as dt
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

DB_PATH   = "Stock Data.db"
CHART_DIR = "charts"


# ──────────────────────────  DB helpers  ──────────────────────────
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


def _update_ttm_div(conn, tic, val):
    cur = conn.cursor()
    cur.execute("""
        UPDATE TTM_Data SET TTM_Dividend=? WHERE Symbol=?;
    """, (val, tic))
    if cur.rowcount == 0:   # row missing – create minimal stub
        cur.execute("""
            INSERT INTO TTM_Data(Symbol,TTM_Dividend,Last_Updated)
            VALUES(?,?,?);
        """, (tic, val, dt.datetime.utcnow().strftime("%F %T")))
    conn.commit()


# ──────────────────────  EPS back-fill helper  ────────────────────
def _fetch_eps_from_yf(tic: str, year: int) -> float | None:
    """
    Try to pull Diluted EPS for *year* directly from yfinance.
    Returns a float if found, otherwise None.
    """
    try:
        inc = yf.Ticker(tic).get_income_stmt(freq="a")  # annual
        col = str(year)
        if col in inc.columns:
            eps_val = inc.loc["Diluted EPS"].get(col)
            return float(eps_val) if pd.notna(eps_val) else None
    except Exception:
        pass
    return None


# ──────────────────────────  Chart builder  ───────────────────────
def _build_chart(tic: str, conn: sqlite3.Connection) -> str:
    cur = conn.cursor()

    # ── pull last 10 annual EPS ──────────────────────────────
    cur.execute("""
        SELECT Date, EPS
        FROM   Annual_Data
        WHERE  Symbol=?
        ORDER  BY Date DESC
        LIMIT 10;
    """, (tic,))
    raw_rows = cur.fetchall()

    trailing: list[tuple[int,float]] = []
    for d, eps in raw_rows:
        yr = int(d[:4])
        if eps is None:
            # try one Yahoo-Finance rescue; store back if we succeed
            new_eps = _fetch_eps_from_yf(tic, yr)
            if new_eps is not None:
                cur.execute("""
                    UPDATE Annual_Data SET EPS=? 
                    WHERE Symbol=? AND Date LIKE ?;
                """, (new_eps, tic, f"{yr}-%"))
                conn.commit()
                eps = new_eps
        trailing.append((yr, float(eps) if eps is not None else 0.0))
    trailing = trailing[::-1]     # chronological order, oldest → newest

    # ── 3 forward EPS from DB ────────────────────────────────
    cur.execute("""
        SELECT Date, ForwardEPS
        FROM   ForwardFinancialData
        WHERE  Ticker=?
        ORDER  BY Date ASC
        LIMIT 3;
    """, (tic,))
    forward = [(int(d[:4]), float(v)) for d, v in cur.fetchall()]

    # ── dividends map for historical bars ────────────────────
    years_needed = [y for y, _ in trailing]
    if years_needed:
        q_marks = ",".join("?"*len(years_needed))
        cur.execute(f"""
            SELECT year, dividend FROM Dividends
            WHERE  ticker=? AND year IN ({q_marks});
        """, (tic, *years_needed))
        div_map = {int(y): float(d) for y, d in cur.fetchall()}
    else:
        div_map = {}

    # ── latest TTM EPS/Dividend ──────────────────────────────
    cur.execute("""
        SELECT TTM_EPS, TTM_Dividend
        FROM   TTM_Data
        WHERE  Symbol=?
        ORDER  BY Last_Updated DESC
        LIMIT 1;
    """, (tic,))
    ttm_row = cur.fetchone() or (0, 0)
    ttm_eps = float(ttm_row[0] or 0)
    ttm_div = float(ttm_row[1] or 0)

    # ── assemble series for bar plot ─────────────────────────
    labels, eps_hist, eps_fwd, divs = [], [], [], []
    for y, v in trailing:
        labels.append(str(y));      eps_hist.append(v); eps_fwd.append(0)
        divs.append(div_map.get(y, 0))

    for y, v in forward:
        labels.append(str(y));      eps_hist.append(0); eps_fwd.append(v)
        divs.append(0)

    labels.append("TTM"); eps_hist.append(ttm_eps); eps_fwd.append(0); divs.append(ttm_div)

    # ── plot ────────────────────────────────────────────────
    x = range(len(labels)); width = .25
    fig, ax = plt.subplots(figsize=(10,6))
    ax.bar([i-width for i in x], eps_hist, width, label="Trailing EPS")
    ax.bar(x, eps_fwd, width, label="Forecast EPS", color="#70a6ff")
    ax.bar([i+width for i in x], divs, width, label="Dividend", color="orange")

    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel("USD per share")
    ax.set_title(f"{tic} – EPS (Actual & Forecast) vs Dividend"); ax.legend()

    # current yield annotation (ignore failures quietly)
    try:
        price = yf.Ticker(tic).history(period="1d")["Close"].iloc[-1]
        last_div = div_map.get(trailing[-1][0], 0) if trailing else 0
        ax.text(0.01, .95, f"Current Yield ≈ {last_div/price*100:0.2f}%",
                transform=ax.transAxes, va="top", fontsize=9)
    except Exception:
        pass

    os.makedirs(CHART_DIR, exist_ok=True)
    path = os.path.join(CHART_DIR, f"{tic}_eps_dividend_forecast.png")
    plt.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return path


# ────────────────────────  public entry  ──────────────────────────
def generate_eps_dividend(tickers, db_path=DB_PATH, chart_dir=CHART_DIR):
    """
    charts = generate_eps_dividend(ticker_list)
    returns {ticker: chart_path}
    """
    conn = sqlite3.connect(db_path); _ensure_schema(conn)
    cur = conn.cursor()

    out = {}
    for tic in tickers:
        # 1) refresh dividends
        divs = yf.Ticker(tic).dividends
        if not divs.empty:
            divs.index = pd.to_datetime(divs.index, utc=True).tz_localize(None)
            for yr, amt in divs.groupby(divs.index.year).sum().items():
                _upsert_dividend_year(cur, tic, int(yr), float(amt))
            one_year_ago = dt.datetime.utcnow() - dt.timedelta(days=365)
            _update_ttm_div(conn, tic, float(divs[divs.index >= one_year_ago].sum()))
        conn.commit()

        # 2) build / save chart
        out[tic] = _build_chart(tic, conn)

    conn.close()
    return out


# helper for main_remote.py
def eps_dividend_generator():
    from ticker_manager import read_tickers
    return generate_eps_dividend(read_tickers("tickers.csv"))


if __name__ == "__main__":
    print(eps_dividend_generator())

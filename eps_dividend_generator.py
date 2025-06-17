"""
EPS-Dividend module
---------------------------------------------
Call   generate_eps_dividend(ticker_list)
for each ticker it will:
    • ensure Dividends table & TTM_Dividend column exist
    • pull & store dividend history + TTM dividend
    • create charts/{ticker}_eps_dividend_forecast.png
Returns dict {ticker: chart_path}
"""

import os, sqlite3, datetime as dt
import yfinance as yf
import matplotlib.pyplot as plt

DB_PATH   = "Stock Data.db"
CHART_DIR = "charts"


# ─────────────────────────────────────────────
# DATABASE HELPERS
# ─────────────────────────────────────────────
def _ensure_schema(conn):
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS Dividends (
            ticker   TEXT,
            year     INTEGER,
            dividend REAL,
            PRIMARY KEY (ticker, year)
        );
    """)
    # add TTM_Dividend column to TTM_Data if missing
    c.execute("PRAGMA table_info(TTM_Data);")
    cols = [row[1].lower() for row in c.fetchall()]
    if 'ttm_dividend' not in cols:
        c.execute("ALTER TABLE TTM_Data ADD COLUMN TTM_Dividend REAL;")
    conn.commit()


def _upsert_dividend_year(c, tic, yr, amt):
    c.execute("""
        INSERT INTO Dividends (ticker, year, dividend)
        VALUES (?, ?, ?)
        ON CONFLICT(ticker, year) DO UPDATE SET dividend=excluded.dividend;
    """, (tic, yr, amt))


def _update_ttm_div(conn, tic, ttm_div):
    cur = conn.cursor()
    cur.execute("""
        UPDATE TTM_Data SET TTM_Dividend = ?
        WHERE Symbol = ?;
    """, (ttm_div, tic))
    if cur.rowcount == 0:            # if row absent create minimal record
        cur.execute("""INSERT INTO TTM_Data (Symbol, TTM_Dividend, Last_Updated)
                       VALUES (?, ?, ?)""",
                    (tic, ttm_div,
                     dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()


# ─────────────────────────────────────────────
# CHART
# ─────────────────────────────────────────────
def _build_chart(tic, conn):
    cur = conn.cursor()

    # 10 trailing EPS
    cur.execute("""
        SELECT Date, EPS
        FROM   Annual_Data
        WHERE  Symbol = ?
        ORDER BY Date DESC
        LIMIT 10;
    """, (tic,))
    trailing = [(int(d[:4]), float(eps)) for d, eps in cur.fetchall()][::-1]

    # 3 forward EPS
    cur.execute("""
        SELECT Date, ForwardEPS
        FROM   ForwardFinancialData
        WHERE  Ticker = ?
        ORDER BY Date ASC
        LIMIT 3;
    """, (tic,))
    forward  = [(int(d[:4]), float(eps)) for d, eps in cur.fetchall()]

    # dividends for needed years
    need_years = [y for y, _ in trailing]
    if need_years:
        placeholders = ",".join("?" * len(need_years))
        cur.execute(f"""
            SELECT year, dividend
            FROM   Dividends
            WHERE  ticker = ?
              AND  year IN ({placeholders});
        """, (tic, *need_years))
        div_map = {int(y): float(d) for y, d in cur.fetchall()}
    else:
        div_map = {}

    # TTM EPS & Dividend
    cur.execute("""
        SELECT TTM_EPS, TTM_Dividend
        FROM   TTM_Data
        WHERE  Symbol = ?
        ORDER BY Last_Updated DESC
        LIMIT 1;
    """, (tic,))
    row = cur.fetchone()
    ttm_eps = float(row[0]) if row and row[0] else 0
    ttm_div = float(row[1]) if row and row[1] else 0

    # labels & series
    labels, eps_hist, eps_fwd, divs = [], [], [], []
    for y, v in trailing:
        labels.append(str(y))
        eps_hist.append(v)
        eps_fwd.append(0)
        divs.append(div_map.get(y, 0))

    for y, v in forward:
        labels.append(str(y))
        eps_hist.append(0)
        eps_fwd.append(v)
        divs.append(0)

    labels.append("TTM")
    eps_hist.append(ttm_eps)
    eps_fwd.append(0)
    divs.append(ttm_div)

    # plot
    x = range(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar([i - width for i in x], eps_hist, width=width, label="Trailing EPS")
    ax.bar(x,                       eps_fwd, width=width, label="Forecast EPS",
           color="#70a6ff")
    ax.bar([i + width for i in x], divs,     width=width, label="Dividend",
           color="orange")

    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel("USD per Share")
    ax.set_title(f"{tic} – EPS (Actual & Forecast) vs Dividend")
    ax.legend()

    # current yield
    try:
        price = yf.Ticker(tic).history(period="1d")["Close"][-1]
        last_div_yr = trailing[-1][0] if trailing else None
        last_div    = div_map.get(last_div_yr, 0)
        yld = (last_div / price) * 100 if price else 0
        ax.text(0.01, 0.95, f"Current Dividend Yield: {yld:.2f}%",
                transform=ax.transAxes, fontsize=9, va="top")
    except Exception:
        pass

    plt.tight_layout()
    if not os.path.isdir(CHART_DIR):
        os.makedirs(CHART_DIR)
    path = os.path.join(CHART_DIR,
                        f"{tic}_eps_dividend_forecast.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ─────────────────────────────────────────────
# PUBLIC MINI-MAIN
# ─────────────────────────────────────────────
def generate_eps_dividend(tickers,
                          db_path=DB_PATH,
                          chart_dir=CHART_DIR):
    """
    Main entry point. Use exactly one line:
        charts = generate_eps_dividend(ticker_list)
    Returns dict {ticker: chart_path}
    """
    conn = sqlite3.connect(db_path)
    _ensure_schema(conn)
    cur = conn.cursor()

    chart_paths = {}
    for tic in tickers:
        # fetch dividends from yfinance
        div_series = yf.Ticker(tic).dividends
        if not div_series.empty:
            yearly_totals = div_series.groupby(div_series.index.year).sum()
            for yr, amt in yearly_totals.items():
                _upsert_dividend_year(cur, tic, int(yr), float(amt))
            # TTM
            last_year_cut = dt.datetime.utcnow() - dt.timedelta(days=365)
            ttm_total = float(div_series[div_series.index >= last_year_cut].sum())
            _update_ttm_div(conn, tic, ttm_total)
        conn.commit()

        # build & store chart
        chart_paths[tic] = _build_chart(tic, conn)

    conn.close()
    return chart_paths


# ─────────────────────────────────────────────
# MINI MAIN for use in main.py
# ─────────────────────────────────────────────

def eps_dividend_generator():
    from ticker_manager import read_tickers  # adjust if this is imported differently
    tickers = read_tickers("tickers.csv")    # replace with your real ticker path
    return generate_eps_dividend(tickers)


# Manual run (optional)
if __name__ == "__main__":
    chart_map = eps_dividend_generator()
    print(chart_map)

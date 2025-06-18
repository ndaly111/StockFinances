"""
eps_dividend_fast.py
────────────────────────────────────────────────────────────────────────
Fast generation of EPS-vs-Dividend charts (or “no dividend” placeholders).

Key speed tricks
• one shared HTTP session (keep-alive)
• 8-second timeout patch on every yfinance GET
• batch UPSERT + WAL + synchronous=OFF in SQLite
• skip chart build if PNG already newer than last DB update
"""

# ── ultra-light timeout patch for all yfinance GETs ──────────────────────────
import yfinance, requests, logging
_session = requests.Session()                # keep-alive
_orig_get = yfinance.utils._requests.get
def _quick(url,*a,**k):
    k.setdefault("timeout", 8)
    k.setdefault("session", _session)
    try:
        return _orig_get(url,*a,**k)
    except Exception as e:                   # 401, timeout, etc.
        logging.warning("yfinance quick-fail %s → %s", url.split('/')[-1], e)
        r = requests.models.Response(); r.status_code, r._content = 200, b"{}"
        return r
yfinance.utils._requests.get = _quick
# ─────────────────────────────────────────────────────────────────────────────

import os, sqlite3, datetime as dt, time, matplotlib
matplotlib.use("Agg")                        # no GUI; avoid font-cache spin-up
import matplotlib.pyplot as plt
import pandas as pd

DB_PATH   = "Stock Data.db"
CHART_DIR = "charts"

# ─────────────────────────  DB bootstrap  ──────────────────────────
def _open_db(path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(path, isolation_level=None,
                           detect_types=sqlite3.PARSE_DECLTYPES)
    conn.execute("PRAGMA journal_mode=WAL;")        # high-concurrency, fast read
    conn.execute("PRAGMA synchronous=OFF;")         # fastest safe writes
    conn.execute("PRAGMA temp_store=MEMORY;")
    _ensure_schema(conn)
    return conn

def _ensure_schema(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.executescript("""
    CREATE TABLE IF NOT EXISTS Dividends(
        ticker TEXT,
        year   INTEGER,
        dividend REAL,
        PRIMARY KEY(ticker,year)
    );
    CREATE TABLE IF NOT EXISTS TTM_Data(
        Symbol TEXT PRIMARY KEY,
        TTM_Dividend REAL,
        TTM_EPS REAL,
        Last_Updated TEXT
    );
    CREATE INDEX IF NOT EXISTS Div_idx ON Dividends(ticker,year);
    CREATE INDEX IF NOT EXISTS TTM_idx ON TTM_Data(Symbol);
    """)
    conn.commit()

# ─────────────────────────  helpers  ───────────────────────────────
def _bulk_upsert_dividends(cur, tic: str, series: pd.Series):
    rows = [(tic, int(y), float(v)) for y, v in
            series.groupby(series.index.year).sum().items()]
    cur.executemany("""
        INSERT INTO Dividends(ticker,year,dividend)
        VALUES(?,?,?)
        ON CONFLICT(ticker,year) DO UPDATE
        SET dividend = excluded.dividend;
    """, rows)

def _update_ttm(cur, tic: str, last365_sum: float):
    ts = dt.datetime.utcnow().strftime("%F %T")
    cur.execute("""
        INSERT INTO TTM_Data(Symbol, TTM_Dividend, Last_Updated)
        VALUES(?,?,?)
        ON CONFLICT(Symbol) DO UPDATE
        SET TTM_Dividend=excluded.TTM_Dividend,
            Last_Updated=excluded.Last_Updated;
    """, (tic, last365_sum, ts))

# ──────────────────────  chart builder  ────────────────────────────
def _chart_needed(path: str, last_update: str | None) -> bool:
    if not os.path.exists(path) or last_update is None:
        return True
    png_mtime = os.path.getmtime(path)
    db_time   = time.mktime(dt.datetime.strptime(last_update, "%Y-%m-%d %H:%M:%S").timetuple())
    return db_time > png_mtime           # DB newer → regenerate chart

def _build_chart(tic: str, conn: sqlite3.Connection) -> str:
    cur = conn.cursor()

    # Quickly bail if PNG already up-to-date
    cur.execute("SELECT Last_Updated FROM TTM_Data WHERE Symbol=?;", (tic,))
    lu = cur.fetchone(); lu = lu[0] if lu else None
    path = os.path.join(CHART_DIR, f"{tic}_eps_dividend_forecast.png")
    if not _chart_needed(path, lu):
        return path

    # trailing EPS (oldest→newest, up to 10)
    cur.execute("""
        SELECT Date, EPS FROM Annual_Data
        WHERE Symbol=? ORDER BY Date ASC LIMIT 10;
    """, (tic,))
    trailing = [(int(d[:4]), float(eps) if eps is not None else 0.0)
                for d, eps in cur.fetchall()]

    # forward EPS (max 3)
    cur.execute("""
        SELECT Date, ForwardEPS FROM ForwardFinancialData
        WHERE Ticker=? ORDER BY Date ASC LIMIT 3;
    """, (tic,))
    forward = [(int(d[:4]), float(v)) for d, v in cur.fetchall()]

    # dividend look-up
    years = [y for y, _ in trailing]
    q = ",".join("?"*len(years)) if years else "NULL"
    cur.execute(f"""
        SELECT year, dividend FROM Dividends
        WHERE ticker=? AND year IN ({q});
    """, (tic, *years))
    div_map = {int(y): float(d) for y, d in cur.fetchall()}

    # TTM row
    cur.execute("SELECT TTM_EPS, TTM_Dividend FROM TTM_Data WHERE Symbol=?;", (tic,))
    ttm_eps, ttm_div = cur.fetchone() or (0.0, 0.0)

    # assemble bars
    labels, eps_hist, eps_fwd, divs = [], [], [], []
    for yr, eps in trailing:
        labels.append(str(yr)); eps_hist.append(eps); eps_fwd.append(0); divs.append(div_map.get(yr,0))
    labels.append("TTM"); eps_hist.append(ttm_eps); eps_fwd.append(0); divs.append(ttm_div)
    for yr, fwd in forward:
        labels.append(str(yr)); eps_hist.append(0); eps_fwd.append(fwd); divs.append(0)

    # plot
    x = range(len(labels)); w = .25
    fig, ax = plt.subplots(figsize=(10,6), dpi=100)
    ax.bar([i-w for i in x], eps_hist, w, label="Trailing EPS")
    ax.bar(x, eps_fwd, w, label="Forecast EPS", color="#70a6ff")
    ax.bar([i+w for i in x], divs, w, label="Dividend", color="orange")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel("USD per share")
    ax.set_title(f"{tic} – EPS (Actual & Forecast) vs Dividend")
    ax.legend()

    os.makedirs(CHART_DIR, exist_ok=True)
    plt.tight_layout(); fig.savefig(path); plt.close(fig)
    return path

# ─────────────────────  main driver  ───────────────────────────────
def generate_eps_dividend(tickers, db_path=DB_PATH, chart_dir=CHART_DIR):
    conn = _open_db(db_path)
    cur  = conn.cursor()
    out  = {}

    import yfinance as yf     # after patch
    share = yf.utils.get_shared_session()

    for tic in tickers:
        t0 = time.perf_counter()
        divs = yf.Ticker(tic, session=share).dividends   # 8-s max
        if divs.empty or divs.sum() == 0:
            # quick placeholder
            path = os.path.join(chart_dir, f"{tic}_eps_dividend_forecast.png")
            if not os.path.exists(path):                 # avoid overwrite I/O
                plt.figure(figsize=(4,2), dpi=100)
                plt.text(0.5,0.5,"no dividend",ha="center",va="center",fontsize=12)
                plt.axis("off"); os.makedirs(chart_dir, exist_ok=True)
                plt.savefig(path, bbox_inches="tight", pad_inches=0); plt.close()
            out[tic] = path; continue

        divs.index = pd.to_datetime(divs.index, utc=True).tz_localize(None)
        _bulk_upsert_dividends(cur, tic, divs)
        last365 = divs[divs.index >= dt.datetime.utcnow()-dt.timedelta(days=365)].sum()
        _update_ttm_div(cur, tic, float(last365))
        conn.commit()

        out[tic] = _build_chart(tic, conn)
        logging.info("✓ %s in %.2fs", tic, time.perf_counter()-t0)

    conn.close(); return out

# helper for external use
def eps_dividend_generator():
    from ticker_manager import read_tickers
    return generate_eps_dividend(read_tickers("tickers.csv"))

if __name__ == "__main__":
    print(eps_dividend_generator())

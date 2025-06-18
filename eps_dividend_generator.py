"""
eps_dividend_fast.py  –  speed-tuned EPS-vs-Dividend generator
————————————————————————————————————————————————————————————————————
Adds:
• yfinance patch that works on *all* 0.2.x paths
• print/log lines so you can see progress in CI logs
"""

import sys, logging, time, importlib, requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    stream=sys.stdout
)

print("🔧  boot-start")

# ─── universal 8-second timeout patch for yfinance GETs ─────────────
def _install_yf_quickpatch():
    """
    Patches whichever internal path exists:
      ≤ 0.2.35 → yfinance.utils._requests.get
      ≥ 0.2.36 → yfinance._utils.requests.get
    Returns True if patch installed, False otherwise.
    """
    paths = ("yfinance.utils._requests", "yfinance._utils.requests")
    for p in paths:
        try:
            mod = importlib.import_module(p)
            orig = mod.get
            break
        except (ModuleNotFoundError, AttributeError):
            continue
    else:
        logging.warning("⚠️  yfinance internal path not found – timeout patch skipped")
        return False

    if getattr(orig, "_fast_patched", False):
        logging.info("patch already installed")
        return True

    _session = requests.Session()        # keep-alive

    def _fast(url, *a, **k):
        k.setdefault("timeout", 8)
        k.setdefault("session", _session)
        try:
            return orig(url, *a, **k)
        except Exception as e:
            logging.warning("yfinance quick-fail %s → %s", url.split('/')[-1], e)
            resp = requests.models.Response()
            resp.status_code, resp._content = 200, b"{}"
            return resp

    _fast._fast_patched = True
    mod.get = _fast
    logging.info("✅  yfinance patched at %s", p)
    return True

_install_yf_quickpatch()
# ───────────────────────────────────────────────────────────────────

print("🔧  imports …")
import os, sqlite3, datetime as dt, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf                       # safe: patch already in place

DB_PATH, CHART_DIR = "Stock Data.db", "charts"

# ─────────────────────────  DB bootstrap  ──────────────────────────
def _open_db(path: str = DB_PATH) -> sqlite3.Connection:
    print(f"📂  opening DB {path}")
    conn = sqlite3.connect(path, isolation_level=None,
                           detect_types=sqlite3.PARSE_DECLTYPES)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=OFF;")
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
    print("✅  schema ensured")

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
def _build_chart(tic: str, conn: sqlite3.Connection) -> str:
    cur = conn.cursor()
    cur.execute("SELECT Last_Updated FROM TTM_Data WHERE Symbol=?;", (tic,))
    lu = cur.fetchone(); lu = lu[0] if lu else None
    path = os.path.join(CHART_DIR, f"{tic}_eps_dividend_forecast.png")
    if os.path.exists(path) and lu:
        png_ts = os.path.getmtime(path)
        db_ts  = dt.datetime.strptime(lu, "%Y-%m-%d %H:%M:%S").timestamp()
        if db_ts <= png_ts:
            print(f"🔄  {tic} chart up-to-date → skip")
            return path

    print(f"📊  building chart for {tic}")
    # trailing EPS
    cur.execute("""
        SELECT Date, EPS FROM Annual_Data
        WHERE Symbol=? ORDER BY Date ASC LIMIT 10;
    """, (tic,))
    trailing = [(int(d[:4]), float(eps) if eps is not None else 0.0)
                for d, eps in cur.fetchall()]
    # forward EPS
    cur.execute("""
        SELECT Date, ForwardEPS FROM ForwardFinancialData
        WHERE Ticker=? ORDER BY Date ASC LIMIT 3;
    """, (tic,))
    forward = [(int(d[:4]), float(v)) for d, v in cur.fetchall()]
    # dividends
    years = [y for y, _ in trailing]
    q = ",".join("?"*len(years)) if years else "NULL"
    cur.execute(f"""
        SELECT year, dividend FROM Dividends
        WHERE ticker=? AND year IN ({q});
    """, (tic, *years))
    div_map = {int(y): float(d) for y, d in cur.fetchall()}
    # TTM
    cur.execute("SELECT TTM_EPS, TTM_Dividend FROM TTM_Data WHERE Symbol=?;", (tic,))
    ttm_eps, ttm_div = cur.fetchone() or (0.0, 0.0)

    labels, eps_hist, eps_fwd, divs = [], [], [], []
    for yr, eps in trailing:
        labels.append(str(yr)); eps_hist.append(eps); eps_fwd.append(0); divs.append(div_map.get(yr,0))
    labels.append("TTM"); eps_hist.append(ttm_eps); eps_fwd.append(0); divs.append(ttm_div)
    for yr, fwd in forward:
        labels.append(str(yr)); eps_hist.append(0); eps_fwd.append(fwd); divs.append(0)

    x = range(len(labels)); w = .25
    fig, ax = plt.subplots(figsize=(10,6), dpi=100)
    ax.bar([i-w for i in x], eps_hist, w, label="Trailing EPS")
    ax.bar(x, eps_fwd, w, label="Forecast EPS", color="#70a6ff")
    ax.bar([i+w for i in x], divs,   w, label="Dividend",     color="orange")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel("USD per share")
    ax.set_title(f"{tic} – EPS (Actual & Forecast) vs Dividend")
    ax.legend(); os.makedirs(CHART_DIR, exist_ok=True)
    plt.tight_layout(); fig.savefig(path); plt.close(fig)
    print(f"💾  saved {path}")
    return path

# ─────────────────────  main driver  ───────────────────────────────
def generate_eps_dividend(tickers, db_path=DB_PATH, chart_dir=CHART_DIR):
    conn = _open_db(db_path); cur = conn.cursor(); out = {}
    share = yf.utils.get_shared_session()

    for tic in tickers:
        print(f"▶️  {tic} …")
        t0 = time.perf_counter()
        divs = yf.Ticker(tic, session=share).dividends
        if divs.empty or divs.sum() == 0:
            print(f"🚫  {tic} no dividends → placeholder")
            path = os.path.join(chart_dir, f"{tic}_eps_dividend_forecast.png")
            if not os.path.exists(path):
                plt.figure(figsize=(4,2), dpi=100)
                plt.text(0.5,0.5,"no dividend",ha="center",va="center",fontsize=12)
                plt.axis("off"); os.makedirs(chart_dir, exist_ok=True)
                plt.savefig(path, bbox_inches="tight", pad_inches=0); plt.close()
            out[tic] = path; continue

        divs.index = pd.to_datetime(divs.index, utc=True).tz_localize(None)
        _bulk_upsert_dividends(cur, tic, divs)
        last365 = divs[divs.index >= dt.datetime.utcnow()-dt.timedelta(days=365)].sum()
        _update_ttm(cur, tic, float(last365)); conn.commit()

        out[tic] = _build_chart(tic, conn)
        logging.info("✓ %s done in %.2fs", tic, time.perf_counter()-t0)

    conn.close(); return out

def eps_dividend_generator():
    from ticker_manager import read_tickers
    return generate_eps_dividend(read_tickers("tickers.csv"))

if __name__ == "__main__":
    print(eps_dividend_generator())

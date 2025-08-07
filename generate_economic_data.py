#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────
#  generate_economic_data.py          (rev 07-Aug-2025)
# ----------------------------------------------------------------
"""
Pull key U.S. macro indicators from FRED, store to SQLite, render
economic_data.html, and save history charts in charts/*.png.

Mini-main: generate_economic_data()
"""

import os, re, sqlite3, datetime as dt
from pathlib import Path
import requests, pandas as pd, matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from fredapi import Fred

# ───────────── configuration ──────────────
DB_FILE   = Path("Stock Data.db")
CHART_DIR = Path("charts")
HTML_OUT  = CHART_DIR / "economic_data.html"

FRED_KEY  = os.getenv("FRED_API_KEY", "").strip()
fred      = Fred(api_key=FRED_KEY) if FRED_KEY else None
TODAY_ISO = dt.datetime.now(tz=dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
# ──────────────────────────────────────────

# ---------- next-release helpers ----------
_BLS_ROOT = "https://www.bls.gov/schedule/news_release"

def _next_bls(slug: str) -> str:
    url  = f"{_BLS_ROOT}/{slug}.htm"
    soup = BeautifulSoup(requests.get(url, timeout=20).text, "html.parser")
    m    = soup.find(string=re.compile(r"Next Release", re.I))
    if m:
        return m.find_next("div").get_text(strip=True)
    m = re.search(r"Next release:? ?(\w+ \d{1,2}, \d{4})", soup.text)
    return m.group(1) if m else "—"

def _next_bea_gdp() -> str:
    url  = "https://www.bea.gov/data/gdp/gross-domestic-product"
    soup = BeautifulSoup(requests.get(url, timeout=20).text, "html.parser")
    m    = re.search(r"Next release:\s+([A-Z][a-z]+ \d{1,2}, \d{4})", soup.text)
    return m.group(1) if m else "—"

# ---------- indicator spec ---------------
INDICATORS = {
    "UNRATE":  {"name": "Unemployment Rate",      "units": "%",          "group": "labor",
                "schedule_func": lambda: _next_bls("empsit")},
    "CPIAUCSL":{"name": "CPI (All-Items YoY)",    "units": "%",          "group": "labor",
                "schedule_func": lambda: _next_bls("cpi")},
    "FEDFUNDS":{"name": "Fed Funds Target",       "units": "%",          "group": "rates",
                "schedule_func": None},
    "DGS10":   {"name": "10-Year Treasury",       "units": "%",          "group": "rates",
                "schedule_func": None},
    "GDPC1":   {"name": "Real GDP (2017$ SAAR)",  "units": "T",          "group": "rates",
                "schedule_func": _next_bea_gdp},
}

LABOR_KEYS  = [k for k,v in INDICATORS.items() if v["group"]=="labor"]
RATES_KEYS  = [k for k,v in INDICATORS.items() if v["group"]=="rates"]

# ---------- SQLite helpers ----------------
def _ensure_tables(conn: sqlite3.Connection):
    conn.execute("""CREATE TABLE IF NOT EXISTS economic_data(
                      indicator TEXT,
                      date      TEXT,
                      value     REAL,
                      PRIMARY KEY(indicator, date))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS economic_meta(
                      indicator    TEXT PRIMARY KEY,
                      name         TEXT,
                      units        TEXT,
                      source       TEXT,
                      last_release TEXT,
                      next_release TEXT)""")

def _upsert_data(conn: sqlite3.Connection, df: pd.DataFrame):
    rows = df[['indicator','date','value']].itertuples(index=False, name=None)
    conn.executemany("INSERT OR REPLACE INTO economic_data VALUES (?,?,?)", rows)

# ---------- FRED fetcher ------------------
def _fetch_series(sid: str) -> pd.Series:
    start = (dt.date.today() - dt.timedelta(days=15*365)).strftime("%Y-%m-%d")
    return fred.get_series(sid, observation_start=start)

# ---------- deltas helpers ----------------
def _pct(a,b):            # percent change a vs b
    return (a/b - 1.0) * 100.0 if b else None

def _fmt(x, unit="%"):
    return f"{x:,.1f} {unit}" if x is not None else "—"

# ---------- HTML writer -------------------
def _render_html(rows: list[dict]):
    # split by group
    labor  = [r for r in rows if r["group"]=="labor"]
    rates  = [r for r in rows if r["group"]=="rates"]

    def _block(title, data, delta1, delta2):
        hdr = (f'<h3>{title}</h3>'
               '<table class="econ-table"><thead>'
               f'<tr><th>Indicator</th><th>Latest</th>'
               f'<th>{delta1}</th><th>{delta2}</th><th>Next</th></tr>'
               '</thead><tbody>')
        body = []
        for r in data:
            body.append(
                f"<tr><td>{r['name']}</td>"
                f"<td>{r['latest']}</td>"
                f"<td>{r['d1']}</td>"
                f"<td>{r['d2']}</td>"
                f"<td>{r['next_release']}</td></tr>")
        return hdr + "".join(body) + "</tbody></table>"

    html = [
        '<div class="econ-wrap">',
        f'<h2>Key U.S. Economic Indicators</h2>',
        f'<p class="stamp">Updated: {TODAY_ISO} &nbsp;|&nbsp; Sources: '
        'BLS · FRED · BEA · U.S. Treasury</p>',
        _block("Labor & Prices", labor, "1-mo Δ", "YoY Δ"),
        _block("Rates & Growth", rates, "1-wk Δ", "3-mo / QoQ Δ"),
        '</div>'
    ]
    HTML_OUT.write_text("\n".join(html), encoding="utf-8")

# ---------- main driver -------------------
def generate_economic_data():
    if not fred:
        print("⚠️  FRED_API_KEY missing – skipping economic-data update.")
        HTML_OUT.write_text("Economic data not available", encoding="utf-8")
        return HTML_OUT

    CHART_DIR.mkdir(exist_ok=True)
    rows_out = []

    with sqlite3.connect(DB_FILE) as conn:
        _ensure_tables(conn)

        for sid, meta in INDICATORS.items():
            ser = _fetch_series(sid)
            if ser.empty:
                continue

            df = (ser.to_frame("value")
                      .reset_index()
                      .rename(columns={"index":"date"}))
            df["indicator"] = sid
            df["date"]      = pd.to_datetime(df["date"]).dt.date.astype(str)
            _upsert_data(conn, df)

            last_date  = df.iloc[-1]["date"]
            last_value = df.iloc[-1]["value"]
            next_rel   = meta["schedule_func"]() if meta["schedule_func"] else "Daily"

            # ---------- compute deltas ----------
            d1 = d2 = None
            if sid == "UNRATE":                      # monthly pct-point change
                d1 = _fmt(df.iloc[-1]["value"] - df.iloc[-2]["value"], "pp")
                d2 = _fmt(df.iloc[-1]["value"] - df.iloc[-13]["value"], "pp")
            elif sid == "CPIAUCSL":                  # YoY headline inflation %
                latest_idx  = df.iloc[-1]["value"]
                year_ago    = df.iloc[-13]["value"]
                yoy         = _pct(latest_idx, year_ago)
                d1          = "—"
                d2          = _fmt(yoy)
            elif sid == "FEDFUNDS":
                d1 = d2 = "0 bp"
            elif sid == "DGS10":
                d1 = _fmt(df.iloc[-1]["value"] - df.iloc[-6]["value"], "bp")
                d2 = _fmt(df.iloc[-1]["value"] - df.iloc[-66]["value"], "bp")
            elif sid == "GDPC1":                     # quarterly series
                lvl_tril  = last_value / 1_000      # billions → trillions
                qoq_sa    = _pct(last_value, df.iloc[-2]["value"])
                yoy_sa    = _pct(last_value, df.iloc[-5]["value"])
                last_value_disp = f"{lvl_tril:,.1f} T"
                d1 = _fmt(qoq_sa)
                d2 = _fmt(yoy_sa)
            else:
                d1 = d2 = "—"

            latest_disp = last_value_disp if sid=="GDPC1" else _fmt(last_value)
            rows_out.append({
                "group"       : meta["group"],
                "name"        : meta["name"],
                "latest"      : latest_disp,
                "d1"          : d1,
                "d2"          : d2,
                "next_release": next_rel
            })

            # ---------- charts ----------
            fig = plt.figure()
            ser.plot(title=meta["name"])
            fig.tight_layout()
            fig.savefig(CHART_DIR / f"{sid}_history.png", dpi=110)
            plt.close(fig)

    _render_html(rows_out)
    print("✓ Economic data updated, HTML + charts generated")
    return HTML_OUT

# ------------------------------------------------------------------
if __name__ == "__main__":
    p = generate_economic_data()
    print(f"[ECON] wrote → {p}  exists? {p.exists()}")

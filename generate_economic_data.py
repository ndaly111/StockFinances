#!/usr/bin/env python3
"""
generate_economic_data.py
Fetch headline U.S. economic indicators, store them in Stock Data.db,
render economic_data.html and history charts.

Mini-main: generate_economic_data()
"""

import os, re, sqlite3, datetime as dt
from pathlib import Path
import requests, pandas as pd, matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from fredapi import Fred

# ───── config ───────────────────────────────────────────────
DB_FILE   = Path("Stock Data.db")
HTML_OUT  = Path("economic_data.html")
CHART_DIR = Path("charts")

FRED_KEY  = os.getenv("FRED_API_KEY", "").strip()
fred      = Fred(api_key=FRED_KEY) if FRED_KEY else None
# ────────────────────────────────────────────────────────────

# ---------- next-release helpers (↑ moved up) ---------------
_BLS_ROOT = "https://www.bls.gov/schedule/news_release"

def _next_bls(slug: str) -> str:
    url  = f"{_BLS_ROOT}/{slug}.htm"
    soup = BeautifulSoup(requests.get(url, timeout=20).text, "html.parser")
    div  = soup.find("div", string=re.compile(r"Next Release", re.I))
    if div:
        return div.find_next("div").get_text(strip=True)
    m = re.search(r"Next release:? ?(\w+ \d{1,2}, \d{4})", soup.text, re.I)
    return m.group(1) if m else "—"

def _next_bea_gdp() -> str:
    url  = "https://www.bea.gov/data/gdp/gross-domestic-product"
    soup = BeautifulSoup(requests.get(url, timeout=20).text, "html.parser")
    m = re.search(r"Next release:\s+([A-Z][a-z]+ \d{1,2}, \d{4})", soup.text)
    return m.group(1) if m else "—"

# ---------- indicator map (now below helpers) ---------------
INDICATORS = {
    "UNRATE":  {"name": "Unemployment Rate",      "units": "%",     "source": "BLS (Employment Situation)",
                "schedule_func": lambda: _next_bls("empsit")},
    "CPIAUCSL":{"name": "CPI (All-Items, SA)",    "units": "Index", "source": "BLS (CPI)",
                "schedule_func": lambda: _next_bls("cpi")},
    "FEDFUNDS":{"name": "Fed Funds Target Rate",  "units": "%",     "source": "Federal Reserve",
                "schedule_func": None},
    "DGS10":   {"name": "10-Year Treasury Yield", "units": "%",     "source": "Federal Reserve",
                "schedule_func": None},
    "GDPC1":   {"name": "Real GDP (annual-rate)", "units": "Bn 2017$", "source": "BEA (GDP)",
                "schedule_func": _next_bea_gdp},
}

# ---------- DB helpers, HTML writer, main driver ------------
def _ensure_tables(conn: sqlite3.Connection):
    conn.execute("""CREATE TABLE IF NOT EXISTS economic_data(
                      indicator TEXT, date TEXT, value REAL,
                      PRIMARY KEY(indicator,date))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS economic_meta(
                      indicator TEXT PRIMARY KEY,
                      name TEXT, units TEXT, source TEXT,
                      last_release TEXT, next_release TEXT)""")

def _fetch_series(sid: str) -> pd.Series:
    start = (dt.date.today() - dt.timedelta(days=15*365)).strftime("%Y-%m-%d")
    return fred.get_series(sid, observation_start=start)

def _render_html(meta_df: pd.DataFrame):
    rows = meta_df.to_dict(orient="records")
    html = ['<h2>Key U.S. Economic Indicators</h2>',
            '<table class="econ-table"><thead>',
            '<tr><th>Indicator</th><th>Latest Value</th><th>Release Date</th><th>Next Release</th></tr>',
            '</thead><tbody>']
    for r in rows:
        html.append(f"<tr><td>{r['name']}</td>"
                    f"<td>{r['latest_value']}</td>"
                    f"<td>{r['last_release']}</td>"
                    f"<td>{r['next_release']}</td></tr>")
    html.append("</tbody></table>")
    HTML_OUT.write_text("\n".join(html), encoding="utf-8")

def generate_economic_data():
    if not fred:
        print("⚠️  FRED_API_KEY missing – skipping economic-data update.")
        HTML_OUT.write_text("Economic data not available", encoding="utf-8")
        return

    CHART_DIR.mkdir(exist_ok=True)
    with sqlite3.connect(DB_FILE) as conn:
        _ensure_tables(conn)
        summary = []

        for sid, meta in INDICATORS.items():
            ser = _fetch_series(sid)
            if ser.empty:
                continue

            df = ser.to_frame("value").reset_index().rename(columns={"index":"date"})
            df["indicator"] = sid
            df.to_sql("economic_data", conn, if_exists="append", index=False)

            last_date, last_val = df.iloc[-1][["date","value"]]
            next_rel = meta["schedule_func"]() if meta["schedule_func"] else "Daily"

            conn.execute("""INSERT INTO economic_meta
                            (indicator,name,units,source,last_release,next_release)
                            VALUES(?,?,?,?,?,?)
                            ON CONFLICT(indicator) DO UPDATE SET
                                last_release=excluded.last_release,
                                next_release=excluded.next_release""",
                         (sid, meta["name"], meta["units"], meta["source"],
                          last_date, next_rel))

            summary.append({"name": meta["name"],
                            "latest_value": f"{last_val:,.2f} {meta['units']}",
                            "last_release": last_date,
                            "next_release": next_rel})

            # chart
            fig = plt.figure()
            ser.plot(title=meta["name"])
            fig.tight_layout()
            fig.savefig(CHART_DIR / f"{sid}_history.png", dpi=110)
            plt.close(fig)

        _render_html(pd.DataFrame(summary))
    print("✓ Economic data updated and charts generated")

if __name__ == "__main__":
    generate_economic_data()

"""
---------------------------------------------------------------------------
generate_economic_data.py
Fetches headline U.S. economic indicators, stores them in Stock Data.db,
renders a summary <table> and one time-series chart per indicator.

Mini-main wrapper:  generate_economic_data()
---------------------------------------------------------------------------"""

import os, re, sqlite3, datetime as dt
from pathlib import Path
import requests, pandas as pd, matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from fredapi import Fred         # pip install fredapi beautifulsoup4 html5lib

# --------------------------- CONFIG ------------------------------------- #
DB_FILE          = Path("Stock Data.db")        # same DB the rest of the site uses
HTML_OUT         = Path("economic_data.html")   # picked up later by html_generator2
CHART_DIR        = Path("charts")               # already exists in your repo

FRED_KEY         = os.getenv("FRED_API_KEY")    # add in GH Secrets
fred             = Fred(api_key=FRED_KEY)

# FRED series → metadata that will appear in the table
INDICATORS = {
    "UNRATE"  : {"name": "Unemployment Rate",    "units": "%",   "source": "BLS (Employment Situation)",
                 "schedule_func": lambda: _next_bls("empsit")},      # monthly
    "CPIAUCSL": {"name": "CPI (U-C All Items)", "units": "Index", "source": "BLS (CPI)",
                 "schedule_func": lambda: _next_bls("cpi")},         # monthly
    "FEDFUNDS": {"name": "Fed Funds Rate",      "units": "%",   "source": "Federal Reserve", "schedule_func": None},
    "DGS10"   : {"name": "10-Yr Treasury Yield","units": "%",   "source": "Federal Reserve", "schedule_func": None},
    "GDPC1"   : {"name": "Real GDP (annual rate)","units": "Bn 2017 $","source": "BEA (GDP)",
                 "schedule_func": _next_bea_gdp},                     # quarterly
}

# ------------------------------------------------------------------------ #
def _ensure_tables(conn: sqlite3.Connection) -> None:
    conn.execute("""CREATE TABLE IF NOT EXISTS economic_data(
                      indicator TEXT, date TEXT, value REAL,
                      PRIMARY KEY(indicator, date))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS economic_meta(
                      indicator TEXT PRIMARY KEY,
                      name TEXT, units TEXT, source TEXT,
                      last_release TEXT, next_release TEXT)""")

def _fetch_series(series_id: str) -> pd.Series:
    """Pull latest 15 years (enough for charts)"""
    start = (dt.date.today() - dt.timedelta(days=15*365)).strftime("%Y-%m-%d")
    return fred.get_series(series_id, observation_start=start)

# ---------- “next release” helpers -------------------------------------- #
_BLS_ROOT = "https://www.bls.gov/schedule/news_release"
def _next_bls(slug: str) -> str:
    """
    Parse BLS schedule page for the next release date of a specific report.
    slug='empsit' → Employment Situation / Unemployment
    slug='cpi'    → CPI
    """
    url  = f"{_BLS_ROOT}/{slug}.htm"
    soup = BeautifulSoup(requests.get(url, timeout=20).text, "html.parser")
    upcoming = soup.find("div", string=re.compile(r"Next Release", re.I))
    if not upcoming:                      # fallback: look for explicit phrase in body
        m = re.search(r"Next release:? ?(\w+ \d{1,2}, \d{4})", soup.text, re.I)
        return m.group(1) if m else "—"
    return upcoming.find_next("div").get_text(strip=True)

def _next_bea_gdp() -> str:
    """Scrape BEA GDP page for the next scheduled date."""
    url  = "https://www.bea.gov/data/gdp/gross-domestic-product"
    soup = BeautifulSoup(requests.get(url, timeout=20).text, "html.parser")
    m = re.search(r"Next release:\s+([A-Z][a-z]+ \d{1,2}, \d{4})", soup.text)
    return m.group(1) if m else "—"

# ---------- HTML writer -------------------------------------------------- #
def _render_html(meta_df: pd.DataFrame) -> None:
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
    html.extend(["</tbody></table>"])
    HTML_OUT.write_text("\n".join(html), encoding="utf-8")

# ---------- Main driver -------------------------------------------------- #
def generate_economic_data() -> None:
    CHART_DIR.mkdir(exist_ok=True)
    with sqlite3.connect(DB_FILE) as conn:
        _ensure_tables(conn)
        summaries = []
        for sid, meta in INDICATORS.items():
            ser  = _fetch_series(sid)
            if ser.empty:
                continue
            # Store new obs to DB
            df   = ser.to_frame("value").reset_index().rename(columns={"index":"date"})
            df["indicator"] = sid
            df.to_sql("economic_data", conn, if_exists="append", index=False)
            # Latest obs
            last_date, last_val = df.iloc[-1][["date","value"]]
            # Write/update meta
            next_rel = meta["schedule_func"]() if meta.get("schedule_func") else "Daily"
            conn.execute("""INSERT INTO economic_meta(indicator,name,units,source,last_release,next_release)
                            VALUES(?,?,?,?,?,?)
                            ON CONFLICT(indicator) DO UPDATE SET
                                last_release=excluded.last_release,
                                next_release=excluded.next_release""",
                         (sid, meta["name"], meta["units"], meta["source"],
                          last_date, next_rel) )
            # Save for summary table
            summaries.append({"name": meta["name"],
                              "latest_value": f"{last_val:,.2f} {meta['units']}",
                              "last_release": last_date,
                              "next_release": next_rel})
            # Chart
            fig = plt.figure()
            ser.plot(title=meta["name"])
            fig.tight_layout()
            fig.savefig(CHART_DIR / f"{sid}_history.png", dpi=110)
            plt.close(fig)
        meta_df = pd.DataFrame(summaries)
        _render_html(meta_df)
    print("✓ Economic data updated and HTML + charts generated")

# Allow one-liner call from main.py / CI
if __name__ == "__main__":
    generate_economic_data()

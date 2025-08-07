#!/usr/bin/env python3
# generate_economic_data.py  – rev 08-Aug-2025
# -------------------------------------------------------------------
"""
• Pull indicator series from FRED
• Upsert into Stock Data.db
• Render economic_data.html (dashboard snippet)
• Render history PNGs
• Finally call economic_data_page.build_full_page()
"""

import os, re, sqlite3, datetime as dt
from pathlib import Path
import requests, pandas as pd, matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from fredapi import Fred

# ───────── config ─────────
DB_FILE   = Path("Stock Data.db")
CHART_DIR = Path("charts")
HTML_OUT  = CHART_DIR / "economic_data.html"

FRED_KEY  = os.getenv("FRED_API_KEY", "").strip()
fred      = Fred(api_key=FRED_KEY) if FRED_KEY else None
STAMP     = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
# ──────────────────────────

# ───────── helper to fetch next release dates ─────────
_BLS_ROOT = "https://www.bls.gov/schedule/news_release"
def _next_bls(slug):    # 'cpi', 'empsit'
    soup = BeautifulSoup(requests.get(f"{_BLS_ROOT}/{slug}.htm",timeout=20).text,"html.parser")
    m    = soup.find(string=re.compile(r"Next Release",re.I))
    if m:
        return m.find_next("div").get_text(strip=True)
    m = re.search(r"Next release:? ?(\w+ \d{1,2}, \d{4})", soup.text)
    return m.group(1) if m else "—"

def _next_bea_gdp():
    soup = BeautifulSoup(requests.get("https://www.bea.gov/data/gdp/gross-domestic-product",timeout=20).text,"html.parser")
    m = re.search(r"Next release:\s+([A-Z][a-z]+ \d{1,2}, \d{4})", soup.text)
    return m.group(1) if m else "—"

# ───────── indicator spec ─────────
INDICATORS = {
    "UNRATE":  {"name":"Unemployment Rate","units":"%","group":"labor",
                "schedule":lambda:_next_bls("empsit")},
    "CPIAUCSL":{"name":"CPI (All-Items YoY)","units":"%","group":"labor",
                "schedule":lambda:_next_bls("cpi")},
    "FEDFUNDS":{"name":"Fed Funds Target","units":"%","group":"rates"},
    "DGS10":   {"name":"10-Year Treasury","units":"%","group":"rates"},
    "GDPC1":   {"name":"Real GDP (2017$ SAAR)","units":"T","group":"rates",
                "schedule":_next_bea_gdp},
}

# ───────── DB helpers ─────────
def _ensure_tables(c):
    c.execute("""CREATE TABLE IF NOT EXISTS economic_data(
                   indicator TEXT,
                   date      TEXT,
                   value     REAL,
                   PRIMARY KEY(indicator,date))""")
    c.execute("""CREATE TABLE IF NOT EXISTS economic_meta(
                   indicator TEXT PRIMARY KEY,
                   name         TEXT,
                   units        TEXT,
                   source       TEXT,
                   last_release TEXT,
                   next_release TEXT)""")

def _upsert(c, df):
    rows = df[['indicator','date','value']].itertuples(False,None)
    c.executemany("INSERT OR REPLACE INTO economic_data VALUES (?,?,?)", rows)

# ───────── utilities ─────────
def _pct(a,b):
    return (a/b-1)*100 if b else None
def _fmt(x,unit="%"): return f"{x:,.1f} {unit}" if x is not None else "—"

# ───────── HTML snippet writer ─────────
def _render_dashboard(rows):
    lab  = [r for r in rows if r["group"]=="labor"]
    rat  = [r for r in rows if r["group"]=="rates"]

    def block(title,data,d1,d2):
        head = (f'<h3>{title}</h3><table class="econ-table"><thead>'
                f'<tr><th>Indicator</th><th>Latest</th>'
                f'<th>{d1}</th><th>{d2}</th><th>Next</th></tr>'
                '</thead><tbody>')
        body = "".join(
            f"<tr><td><a href=\"economic_charts.html#{r['sid']}\">{r['name']}</a></td>"
            f"<td>{r['latest']}</td><td>{r['d1']}</td><td>{r['d2']}</td>"
            f"<td>{r['next']}</td></tr>" for r in data)
        tail = "</tbody></table>"
        return head+body+tail

    html = [f'<p class="stamp">Updated: {STAMP} | Sources: BLS · FRED · BEA · U.S. Treasury</p>',
            block("Labor & Prices",lab,"1-mo Δ","YoY Δ"),
            block("Rates & Growth",rat,"1-wk Δ","3-mo / QoQ Δ")]
    HTML_OUT.write_text("\n".join(html),encoding="utf-8")

# ───────── main ─────────
def generate_economic_data():
    if not fred:
        print("⚠️ FRED_API_KEY missing – skipping update")
        HTML_OUT.write_text("Economic data not available",encoding="utf-8")
        return
    CHART_DIR.mkdir(exist_ok=True)
    rows=[]
    with sqlite3.connect(DB_FILE) as conn:
        _ensure_tables(conn)
        for sid,meta in INDICATORS.items():
            start=(dt.date.today()-dt.timedelta(days=15*365)).strftime("%Y-%m-%d")
            ser = fred.get_series(sid,observation_start=start)
            if ser.empty: continue
            df=(ser.to_frame("value").reset_index()
                  .rename(columns={"index":"date"}))
            df["indicator"]=sid
            df["date"]=pd.to_datetime(df["date"]).dt.date.astype(str)
            _upsert(conn,df)

            last=ser.iloc[-1]; last_date=str(ser.index[-1].date())
            next_rel=meta.get("schedule",lambda:"Daily")()
            # deltas
            d1=d2="—"
            if sid=="UNRATE":
                d1=_fmt(last-ser.iloc[-2],"pp"); d2=_fmt(last-ser.iloc[-13],"pp")
            elif sid=="CPIAUCSL":
                d2=_fmt(_pct(last,ser.iloc[-13]))
            elif sid=="DGS10":
                d1=_fmt(last-ser.iloc[-6],"bp"); d2=_fmt(last-ser.iloc[-66],"bp")
            elif sid=="GDPC1":
                trill=last/1_000
                qoq=_pct(last,ser.iloc[-2]); yoy=_pct(last,ser.iloc[-5])
                last_disp=f"{trill:,.1f} T"; d1=_fmt(qoq); d2=_fmt(yoy)
            else:
                last_disp=_fmt(last,meta["units"])
            if sid!="GDPC1": last_disp=_fmt(last,meta["units"])

            rows.append(dict(
                sid=sid,group=meta["group"],name=meta["name"],
                latest=last_disp,d1=d1,d2=d2,next=next_rel))

            # PNG chart
            plt.figure(); ser.plot(title=meta["name"]); plt.tight_layout()
            plt.savefig(CHART_DIR/f"{sid}_history.png",dpi=110); plt.close()

    _render_dashboard(rows)

    # ─── build single-page history site ───
    from economic_data_page import render_single_page
    render_single_page(STAMP, INDICATORS)   # <─ helper builds economic_charts.html

    print("✓ Economic data & charts updated")
# -------------------------------------------------------------------
if __name__=="__main__":
    generate_economic_data()

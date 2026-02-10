#!/usr/bin/env python3
# generate_economic_data.py  – rev 09-Feb-2026
# 10 indicators: +PCEPI, DGS2, T10Y2Y, ICSA, UMCSENT
# -------------------------------------------------------------------
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
    soup = BeautifulSoup(requests.get(f"{_BLS_ROOT}/{slug}.htm", timeout=20).text, "html.parser")
    m    = soup.find(string=re.compile(r"Next Release", re.I))
    if m:
        return m.find_next("div").get_text(strip=True)
    m = re.search(r"Next release:? ?(\w+ \d{1,2}, \d{4})", soup.text)
    return m.group(1) if m else "—"

def _next_bea_gdp():
    soup = BeautifulSoup(requests.get("https://www.bea.gov/data/gdp/gross-domestic-product", timeout=20).text, "html.parser")
    m = re.search(r"Next release:\s+([A-Z][a-z]+ \d{1,2}, \d{4})", soup.text)
    return m.group(1) if m else "—"

# ───────── indicator spec ─────────
# NOTE: CPIAUCSL is stored as YoY % (not the raw index).
INDICATORS = {
    "UNRATE":   {"name":"Unemployment Rate","units":"%","group":"labor",
                 "schedule":lambda:_next_bls("empsit")},
    "CPIAUCSL": {"name":"CPI (All-Items YoY)","units":"%","group":"labor",
                 "schedule":lambda:_next_bls("cpi")},
    "PCEPI":    {"name":"PCE Price Index (YoY)","units":"%","group":"labor"},
    "ICSA":     {"name":"Initial Jobless Claims","units":"K","group":"labor"},
    "UMCSENT":  {"name":"Consumer Sentiment","units":"idx","group":"labor"},
    "DGS10":    {"name":"10-Year Treasury","units":"%","group":"rates"},
    "DGS2":     {"name":"2-Year Treasury","units":"%","group":"rates"},
    "T10Y2Y":   {"name":"10Y-2Y Yield Spread","units":"%","group":"rates"},
    "GDPC1":    {"name":"Real GDP (2017$ SAAR)","units":"T","group":"rates",
                 "schedule":_next_bea_gdp},
    # pseudo-row for display; data actually from DFEDTARL/U
    "FEDFUNDS": {"name":"Fed Funds Target","units":"%","group":"rates"},
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
    rows = df[['indicator','date','value']].itertuples(False, None)
    c.executemany("INSERT OR REPLACE INTO economic_data VALUES (?,?,?)", rows)

def _normalize_dates(conn):
    """Safely normalize dates to 'YYYY-MM-DD' without breaking the PK.
       1) Delete long-date rows that already have a matching short-date row.
       2) If multiple long-date rows collapse to the same day, keep the latest.
       3) Truncate remaining long dates.
    """
    cur = conn.cursor()

    # 1) If a short 'YYYY-MM-DD' already exists for that indicator/day, drop the long one
    cur.execute("""
        DELETE FROM economic_data
        WHERE length(date) > 10
          AND EXISTS (
                SELECT 1
                FROM economic_data e2
                WHERE e2.indicator = economic_data.indicator
                  AND length(e2.date) = 10
                  AND e2.date = substr(economic_data.date,1,10)
          )
    """)
    conn.commit()

    # 2) Among remaining long dates that collapse to the same day, keep only the latest timestamp
    cur.execute("""
        DELETE FROM economic_data
        WHERE length(date) > 10
          AND date NOT IN (
                SELECT MAX(e2.date)
                FROM economic_data e2
                WHERE e2.indicator = economic_data.indicator
                  AND substr(e2.date,1,10) = substr(economic_data.date,1,10)
          )
    """)
    conn.commit()

    # 3) Now it's safe to truncate
    cur.execute("""
        UPDATE economic_data
        SET date = substr(date,1,10)
        WHERE length(date) > 10
    """)
    conn.commit()

# ───────── utilities ─────────
def _pct(a, b):
    """percent change (e.g., 2.7 for +2.7%)."""
    return (a / b - 1) * 100 if (b not in (None, 0)) else None

def _fmt(x, unit="%"):
    return f"{x:,.1f} {unit}" if x is not None else "—"

# ───────── HTML snippet writer ─────────
def _render_dashboard(rows):
    lab = [r for r in rows if r["group"] == "labor"]
    rat = [r for r in rows if r["group"] == "rates"]

    def block(title, data, d1, d2):
        head = (f'<h3>{title}</h3><table class="econ-table"><thead>'
                f'<tr><th>Indicator</th><th>Latest</th>'
                f'<th>{d1}</th><th>{d2}</th><th>Next</th></tr>'
                '</thead><tbody>')
        body = "".join(
            f"<tr><td><a href=\"economic_charts.html#{r['sid']}\">{r['name']}</a></td>"
            f"<td>{r['latest']}</td><td>{r['d1']}</td><td>{r['d2']}</td>"
            f"<td>{r['next']}</td></tr>"
            for r in data
        )
        tail = "</tbody></table>"
        return head + body + tail

    html = [
        f'<p class="stamp">Updated: {STAMP} | Sources: BLS · FRED · BEA · U.S. Treasury</p>',
        # Labor block shows deltas in percentage points (pp)
        block("Labor & Prices", lab, "1-mo Δ (pp)", "YoY Δ (pp)"),
        # Rates block keeps original mixed units (bp / % / QoQ)
        block("Rates & Growth", rat, "1-wk Δ", "3-mo / QoQ Δ"),
    ]
    HTML_OUT.write_text("\n".join(html), encoding="utf-8")

# ───────── main ─────────
def generate_economic_data():
    if not fred:
        print("⚠️ FRED_API_KEY missing – skipping update")
        HTML_OUT.write_text("Economic data not available", encoding="utf-8")
        return

    CHART_DIR.mkdir(exist_ok=True)
    rows = []

    with sqlite3.connect(DB_FILE) as conn:
        _ensure_tables(conn)

        # ---- fetch core series ----
        start = (dt.date.today() - dt.timedelta(days=15 * 365)).strftime("%Y-%m-%d")

        def _fred_series(sid: str) -> pd.Series:
            """Fetch a FRED series, returning an empty Series on error."""
            try:
                return fred.get_series(sid, observation_start=start)
            except Exception as e:
                print(f"\u26a0\ufe0f FRED: failed to fetch {sid}: {e}")
                return pd.Series(dtype=float)

        unrate = _fred_series("UNRATE")
        cpi_ix = _fred_series("CPIAUCSL")  # raw index; convert to YoY %
        pce_ix = _fred_series("PCEPI")     # raw index; convert to YoY %
        icsa   = _fred_series("ICSA")
        umcsent= _fred_series("UMCSENT")
        dgs10  = _fred_series("DGS10")
        dgs2   = _fred_series("DGS2")
        t10y2y = _fred_series("T10Y2Y")
        gdp    = _fred_series("GDPC1")
        tarL   = _fred_series("DFEDTARL")
        tarU   = _fred_series("DFEDTARU")

        # ---- upsert raw series EXCEPT CPI/PCE (we store those as YoY %) ----
        for sid, ser in {
            "UNRATE": unrate, "DGS10": dgs10, "DGS2": dgs2, "T10Y2Y": t10y2y,
            "GDPC1": gdp, "DFEDTARL": tarL, "DFEDTARU": tarU,
            "ICSA": icsa, "UMCSENT": umcsent,
        }.items():
            if ser is None or ser.empty:
                continue
            df = (ser.to_frame("value").reset_index().rename(columns={"index": "date"}))
            df["indicator"] = sid
            df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
            _upsert(conn, df)

        # ───── UNRATE row (pp deltas) ─────
        if not unrate.empty:
            last = float(unrate.iloc[-1])
            last_disp = _fmt(last, "%")
            d1 = f"{(last - float(unrate.iloc[-2])):+.2f} pp" if len(unrate) >= 2 else "—"
            d2 = f"{(last - float(unrate.iloc[-13])):+.2f} pp" if len(unrate) >= 13 else "—"
            rows.append(dict(sid="UNRATE", group="labor", name=INDICATORS["UNRATE"]["name"],
                             latest=last_disp, d1=d1, d2=d2, next=_next_bls("empsit")))
            plt.figure(); unrate.plot(title=INDICATORS["UNRATE"]["name"]); plt.tight_layout()
            plt.savefig(CHART_DIR / "UNRATE_history.png", dpi=110); plt.close()

        # ───── CPI row (purge old + store YoY %) ─────
        if not cpi_ix.empty:
            # 1) Remove any existing CPI rows (index or prior attempts)
            conn.execute("DELETE FROM economic_data WHERE indicator='CPIAUCSL'")
            conn.commit()

            # 2) Build YoY % and upsert with normalized dates
            cpi_yoy = (cpi_ix.pct_change(12) * 100).dropna()
            df = cpi_yoy.to_frame("value").reset_index().rename(columns={"index": "date"})
            df["indicator"] = "CPIAUCSL"
            df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
            _upsert(conn, df)

            # 3) Normalize any leftover time-stamped dates across the table
            _normalize_dates(conn)

            # Latest stats (pp deltas)
            last_yoy = float(cpi_yoy.iloc[-1])
            last_disp = _fmt(last_yoy, "%")
            mchg = f"{last_yoy - float(cpi_yoy.iloc[-2]):+.2f} pp" if len(cpi_yoy) >= 2 else "—"
            ychg = f"{last_yoy - float(cpi_yoy.iloc[-13]):+.2f} pp" if len(cpi_yoy) >= 13 else "—"

            rows.append(dict(sid="CPIAUCSL", group="labor", name=INDICATORS["CPIAUCSL"]["name"],
                             latest=last_disp, d1=mchg, d2=ychg, next=_next_bls("cpi")))

            # Chart YoY %
            plt.figure()
            cpi_yoy.plot(title="CPI (All Items, YoY %)")
            plt.tight_layout()
            plt.savefig(CHART_DIR / "CPIAUCSL_history.png", dpi=110)
            plt.close()

        # ───── PCE row (purge old + store YoY %) ─────
        if not pce_ix.empty:
            conn.execute("DELETE FROM economic_data WHERE indicator='PCEPI'")
            conn.commit()
            pce_yoy = (pce_ix.pct_change(12) * 100).dropna()
            df = pce_yoy.to_frame("value").reset_index().rename(columns={"index": "date"})
            df["indicator"] = "PCEPI"
            df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
            _upsert(conn, df)
            _normalize_dates(conn)
            last_pce = float(pce_yoy.iloc[-1])
            mchg = f"{last_pce - float(pce_yoy.iloc[-2]):+.2f} pp" if len(pce_yoy) >= 2 else "—"
            ychg = f"{last_pce - float(pce_yoy.iloc[-13]):+.2f} pp" if len(pce_yoy) >= 13 else "—"
            rows.append(dict(sid="PCEPI", group="labor", name=INDICATORS["PCEPI"]["name"],
                             latest=_fmt(last_pce, "%"), d1=mchg, d2=ychg, next="Monthly"))

        # ───── ICSA row (K, weekly) ─────
        if not icsa.empty:
            v = float(icsa.iloc[-1])
            d1 = f"{(v - float(icsa.iloc[-2])) / 1000:+.0f} K" if len(icsa) >= 2 else "—"
            d2 = f"{(v - float(icsa.iloc[-53])) / 1000:+.0f} K" if len(icsa) >= 53 else "—"
            rows.append(dict(sid="ICSA", group="labor", name=INDICATORS["ICSA"]["name"],
                             latest=f"{v / 1000:,.0f} K", d1=d1, d2=d2, next="Weekly"))

        # ───── UMCSENT row (index) ─────
        if not umcsent.empty:
            v = float(umcsent.iloc[-1])
            d1 = f"{v - float(umcsent.iloc[-2]):+.1f}" if len(umcsent) >= 2 else "—"
            d2 = f"{v - float(umcsent.iloc[-13]):+.1f}" if len(umcsent) >= 13 else "—"
            rows.append(dict(sid="UMCSENT", group="labor", name=INDICATORS["UMCSENT"]["name"],
                             latest=f"{v:.1f}", d1=d1, d2=d2, next="Monthly"))

        # ───── 10-Year row (bp deltas) ─────
        if not dgs10.empty:
            v = float(dgs10.iloc[-1])
            last_disp = _fmt(v, "%")
            d1 = f"{(v - float(dgs10.iloc[-6])) * 100:+.0f} bp" if len(dgs10) >= 6 else "—"
            d2 = f"{(v - float(dgs10.iloc[-66])) * 100:+.0f} bp" if len(dgs10) >= 66 else "—"
            rows.append(dict(sid="DGS10", group="rates", name=INDICATORS["DGS10"]["name"],
                             latest=last_disp, d1=d1, d2=d2, next="Daily"))
            plt.figure(); dgs10.plot(title=INDICATORS["DGS10"]["name"]); plt.tight_layout()
            plt.savefig(CHART_DIR / "DGS10_history.png", dpi=110); plt.close()

        # ───── 2-Year row (bp deltas) ─────
        if not dgs2.empty:
            v = float(dgs2.iloc[-1])
            d1 = f"{(v - float(dgs2.iloc[-6])) * 100:+.0f} bp" if len(dgs2) >= 6 else "—"
            d2 = f"{(v - float(dgs2.iloc[-66])) * 100:+.0f} bp" if len(dgs2) >= 66 else "—"
            rows.append(dict(sid="DGS2", group="rates", name=INDICATORS["DGS2"]["name"],
                             latest=_fmt(v, "%"), d1=d1, d2=d2, next="Daily"))

        # ───── 10Y-2Y Yield Spread row (bp deltas) ─────
        if not t10y2y.empty:
            v = float(t10y2y.iloc[-1])
            d1 = f"{(v - float(t10y2y.iloc[-6])) * 100:+.0f} bp" if len(t10y2y) >= 6 else "—"
            d2 = f"{(v - float(t10y2y.iloc[-66])) * 100:+.0f} bp" if len(t10y2y) >= 66 else "—"
            rows.append(dict(sid="T10Y2Y", group="rates", name=INDICATORS["T10Y2Y"]["name"],
                             latest=f"{v:+.2f} %", d1=d1, d2=d2, next="Daily"))

        # ───── GDP row ─────
        if not gdp.empty:
            last = float(gdp.iloc[-1])
            trill = last / 1_000
            qoq = _pct(last, float(gdp.iloc[-2])) if len(gdp) >= 2 else None
            yoy = _pct(last, float(gdp.iloc[-5])) if len(gdp) >= 5 else None
            rows.append(dict(sid="GDPC1", group="rates", name=INDICATORS["GDPC1"]["name"],
                             latest=f"{trill:,.1f} T", d1=_fmt(qoq, "%"), d2=_fmt(yoy, "%"),
                             next=_next_bea_gdp()))
            plt.figure(); gdp.plot(title=INDICATORS["GDPC1"]["name"]); plt.tight_layout()
            plt.savefig(CHART_DIR / "GDPC1_history.png", dpi=110); plt.close()

        # ───── Fed Funds TARGET RANGE row (uses DFEDTARL/U) ─────
        if not tarL.empty and not tarU.empty:
            comb = pd.concat([tarL.rename("L"), tarU.rename("U")], axis=1).dropna()
            low  = float(comb["L"].iloc[-1]); up = float(comb["U"].iloc[-1])
            last_disp = f"{low:.2f} – {up:.2f} %"
            rows.append(dict(sid="FEDFUNDS", group="rates", name=INDICATORS["FEDFUNDS"]["name"],
                             latest=last_disp, d1="—", d2="—", next="Daily"))
            comb["MID"] = (comb["L"] + comb["U"]) / 2.0
            plt.figure(); comb["MID"].plot(title="Fed Funds Target (Midpoint)"); plt.tight_layout()
            plt.savefig(CHART_DIR / "FEDFUNDS_history.png", dpi=110); plt.close()

    # write dashboard snippet
    _render_dashboard(rows)

    # build single-page history site
    from economic_data_page import render_single_page
    render_single_page(STAMP, INDICATORS)

    print("✓ Economic data & charts updated")

# -------------------------------------------------------------------
if __name__ == "__main__":
    generate_economic_data()

#!/usr/bin/env python3
# generate_economic_data.py  – rev 09-Feb-2026
# 10 indicators: +PCEPI, DGS2, T10Y2Y, ICSA, UMCSENT
# -------------------------------------------------------------------
import os, re, sqlite3, datetime as dt
from pathlib import Path
import requests, pandas as pd, matplotlib.pyplot as plt
from fredapi import Fred

# ───────── config ─────────
DB_FILE   = Path("Stock Data.db")
CHART_DIR = Path("charts")
HTML_OUT  = CHART_DIR / "economic_data.html"

FRED_KEY  = os.getenv("FRED_API_KEY", "").strip()
fred      = Fred(api_key=FRED_KEY) if FRED_KEY else None
STAMP     = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None).strftime("%Y-%m-%d %H:%M UTC")
# ──────────────────────────

# ───────── next release dates ─────────
# BLS/BEA hard-403 scrapers (bot-blocked), so the old scrape always fell through
# to an em-dash that rendered as mojibake on the dashboard. Primary source is
# now the FRED release-calendar API (includes FUTURE scheduled dates and tracks
# BLS/BEA schedule revisions automatically; FRED_API_KEY already in CI).
# Fallback: the verified static dates below. Final fallback: "TBD".
# All output is pure ASCII — no em-dashes, so the mojibake cannot recur.
_FRED_RELEASE_IDS = {"empsit": 50, "cpi": 10, "gdp": 53}

# Only dates verified against official schedules (do NOT guess — BLS revised
# its 2026 calendar after the appropriations lapse). FOMC decision days
# (2nd day of each meeting) are published years ahead; 2026 + 2027 verified
# against federalreserve.gov 2026-07-10 (2027 tentative per the Fed).
_STATIC_RELEASES = {
    "empsit": ["2026-08-07"],
    "cpi":    ["2026-07-14", "2026-08-12"],
    "gdp":    ["2026-07-30", "2026-08-26", "2026-09-30"],
    "fomc":   ["2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
               "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
               "2027-01-27", "2027-03-17", "2027-04-28", "2027-06-09",
               "2027-07-28", "2027-09-15", "2027-10-27", "2027-12-08"],
}

_MONTHS = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"], start=1)}


def _parse_fomc_calendar(html: str) -> list:
    """Decision days (last day of each meeting) from the Fed's calendar page.

    The page is organized as '<year> FOMC Meetings' sections with month +
    day-range entries ('27-28', '17-18*', or cross-month '31-November 1')."""
    out = []
    # Split into year sections; chunk i pairs with year i.
    parts = re.split(r"(20\d\d)\s+FOMC\s+Meetings", html)
    for j in range(1, len(parts), 2):
        year, chunk = int(parts[j]), parts[j + 1]
        # (?!\s*,\s*20\d\d): meeting entries are bare day ranges ('27-28');
        # narrative dates ('Released January 5, 2021') carry a trailing
        # ', <year>' — reject those.
        for month, days in re.findall(
                r"(January|February|March|April|May|June|July|August|"
                r"September|October|November|December)(?:\s|<[^>]*>|&nbsp;)*"
                r"([\d]{1,2}(?:\s*-\s*(?:[A-Z][a-z]+\s+)?[\d]{1,2})?)\*?"
                r"(?!\s*,\s*20\d\d)(?!\d)", chunk):
            end = days.split("-")[-1].strip().rstrip("*")
            m2 = re.match(r"([A-Z][a-z]+)\s+(\d{1,2})", end)
            if m2:                              # cross-month: 'October 31-November 1'
                month, day = m2.group(1), int(m2.group(2))
            else:
                day = int(end)
            try:
                out.append(dt.date(year, _MONTHS[month], day).isoformat())
            except (KeyError, ValueError):
                continue
    # Real meetings are 6+ weeks apart; parsed dates within 3 days of each other
    # are one meeting plus page noise — keep the latest (the decision day).
    dedup = []
    for d in sorted(set(out)):
        if dedup and (dt.date.fromisoformat(d) - dt.date.fromisoformat(dedup[-1])).days <= 3:
            dedup[-1] = d
        else:
            dedup.append(d)
    return dedup


def _fomc_dates_from_fed() -> list:
    """FOMC decision days scraped from the Fed's published meeting calendar
    (federalreserve.gov is not bot-blocked, unlike BLS/BEA)."""
    r = requests.get("https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
                     timeout=20)
    r.raise_for_status()
    dates = _parse_fomc_calendar(r.text)
    if not dates:
        raise RuntimeError("no FOMC dates parsed (page layout changed?)")
    return dates


def _fred_release_dates_raw(release_id):
    """Scheduled release dates (past + future) from the FRED release-dates API."""
    if not FRED_KEY:
        raise RuntimeError("FRED_API_KEY not set")
    r = requests.get(
        "https://api.stlouisfed.org/fred/release/dates",
        params={"release_id": release_id, "api_key": FRED_KEY, "file_type": "json",
                "include_release_dates_with_no_data": "true",
                "sort_order": "asc", "limit": 1000},
        timeout=20)
    r.raise_for_status()
    return [d["date"] for d in r.json().get("release_dates", [])]


def _next_release(kind, today=None):
    """Next scheduled release for 'empsit'/'cpi'/'gdp'/'fomc' as ASCII text.

    Fully automated: empsit/cpi/gdp from the FRED release-calendar API, fomc
    scraped from the Fed's published calendar — both refresh every build.
    Falling back to the static calendar or 'TBD' emits a GitHub Actions
    ::warning:: annotation so staleness is LOUD, never silent (the old
    scrapers rotted unnoticed for a year)."""
    today = today or dt.date.today()
    dates = []
    try:
        if kind == "fomc":
            dates = _fomc_dates_from_fed()
        elif kind in _FRED_RELEASE_IDS:
            dates = _fred_release_dates_raw(_FRED_RELEASE_IDS[kind])
    except Exception as e:
        print(f"::warning title=Econ release calendar::{kind}: live source failed "
              f"({e}); using static fallback (goes stale without maintenance)")
    if not dates:
        dates = _STATIC_RELEASES.get(kind, [])
    for d in dates:
        try:
            d_date = dt.date.fromisoformat(d[:10])
        except ValueError:
            continue
        if d_date >= today:
            label = f"{d_date.strftime('%b')} {d_date.day}, {d_date.year}"
            return f"{label} (FOMC)" if kind == "fomc" else label
    print(f"::warning title=Econ release calendar::{kind}: no future date available "
          f"- dashboard shows TBD; refresh _STATIC_RELEASES or fix the live source")
    return "TBD"

# ───────── indicator spec ─────────
# NOTE: CPIAUCSL is stored as YoY % (not the raw index).
INDICATORS = {
    "UNRATE":   {"name":"Unemployment Rate","units":"%","group":"labor",
                 "schedule":lambda:_next_release("empsit")},
    "CPIAUCSL": {"name":"CPI (All-Items YoY)","units":"%","group":"labor",
                 "schedule":lambda:_next_release("cpi")},
    "PCEPI":    {"name":"PCE Price Index (YoY)","units":"%","group":"labor"},
    "ICSA":     {"name":"Initial Jobless Claims","units":"K","group":"labor"},
    "UMCSENT":  {"name":"Consumer Sentiment","units":"idx","group":"labor"},
    "DGS10":    {"name":"10-Year Treasury","units":"%","group":"rates"},
    "DGS2":     {"name":"2-Year Treasury","units":"%","group":"rates"},
    "T10Y2Y":   {"name":"10Y-2Y Yield Spread","units":"%","group":"rates"},
    "GDPC1":    {"name":"Real GDP (2017$ SAAR)","units":"T","group":"rates",
                 "schedule":lambda:_next_release("gdp")},
    # pseudo-row for display; data actually from DFEDTARL/U
    "FEDFUNDS": {"name":"Fed Funds Target","units":"%","group":"rates"},
}

# Investor-focused indicators added for the stock-investor dashboard.
# Fetched as simple raw series; the page generator builds composites
# (recession risk, real rate, risk pricing) on top.
EQUITY_INVESTOR_INDICATORS = {
    # ── Cycle / recession risk ───────────────────────────────────────────
    "T10Y3M":         {"name":"10Y-3M Yield Spread","units":"%","group":"cycle"},
    "SAHMREALTIME":   {"name":"Sahm Rule Recession Indicator","units":"pp","group":"cycle"},
    "RECPROUSM156N":  {"name":"NY Fed 12-Mo Recession Probability","units":"%","group":"cycle"},
    "USSLIND":        {"name":"Leading Index (US)","units":"%","group":"cycle"},
    # ── Fed policy stance ────────────────────────────────────────────────
    "DFF":            {"name":"Effective Federal Funds Rate","units":"%","group":"fed"},
    "PCEPILFE":       {"name":"Core PCE Price Index","units":"idx","group":"fed"},
    "T10YIE":         {"name":"10Y Breakeven Inflation","units":"%","group":"fed"},
    # ── Risk pricing ─────────────────────────────────────────────────────
    "BAMLH0A0HYM2":   {"name":"High Yield OAS (Credit Spread)","units":"%","group":"risk"},
    "VIXCLS":         {"name":"VIX (Equity Volatility)","units":"idx","group":"risk"},
    "STLFSI4":        {"name":"St. Louis Fed Financial Stress","units":"idx","group":"risk"},
    "DTWEXBGS":       {"name":"Trade-Weighted Dollar Index","units":"idx","group":"risk"},
    "DCOILWTICO":     {"name":"WTI Crude Oil","units":"$/bbl","group":"risk"},
    # ── Earnings drivers ─────────────────────────────────────────────────
    # NAPM (ISM Manufacturing PMI) was discontinued by FRED; use PCU3331-
    # equivalent or live ISM. INDPRO is a reasonable industrial-output proxy
    # and is free + reliable.
    "INDPRO":         {"name":"Industrial Production","units":"idx","group":"earnings"},
    "PAYEMS":         {"name":"Nonfarm Payrolls","units":"K","group":"earnings"},
    "PCEC96":         {"name":"Real Consumer Spending","units":"$B","group":"earnings"},
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
        # Longer history for the investor-dashboard composites \u2014 recession
        # risk / real-rate / risk-pricing percentiles need ~50 years for context.
        long_start = "1970-01-01"

        def _fred_series(sid: str, observation_start: str | None = None) -> pd.Series:
            """Fetch a FRED series, returning an empty Series on error."""
            try:
                return fred.get_series(sid, observation_start=observation_start or start)
            except Exception as e:
                print(f"\u26a0\ufe0f FRED: failed to fetch {sid}: {e}")
                return pd.Series(dtype=float)

        # ---- fetch investor-dashboard series (long history) ----
        for sid in EQUITY_INVESTOR_INDICATORS:
            ser = _fred_series(sid, observation_start=long_start)
            if ser is None or ser.empty:
                continue
            df = ser.to_frame("value").reset_index().rename(columns={"index": "date"})
            df["indicator"] = sid
            df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
            _upsert(conn, df)

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
                             latest=last_disp, d1=d1, d2=d2, next=_next_release("empsit")))
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
                             latest=last_disp, d1=mchg, d2=ychg, next=_next_release("cpi")))

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
                             next=_next_release("gdp")))
            plt.figure(); gdp.plot(title=INDICATORS["GDPC1"]["name"]); plt.tight_layout()
            plt.savefig(CHART_DIR / "GDPC1_history.png", dpi=110); plt.close()

        # ───── Fed Funds TARGET RANGE row (uses DFEDTARL/U) ─────
        if not tarL.empty and not tarU.empty:
            comb = pd.concat([tarL.rename("L"), tarU.rename("U")], axis=1).dropna()
            low  = float(comb["L"].iloc[-1]); up = float(comb["U"].iloc[-1])
            # ASCII hyphen + "n/a": the en-/em-dashes rendered as mojibake ("?")
            # when this snippet was embedded into the homepage.
            last_disp = f"{low:.2f} - {up:.2f} %"
            rows.append(dict(sid="FEDFUNDS", group="rates", name=INDICATORS["FEDFUNDS"]["name"],
                             latest=last_disp, d1="n/a", d2="n/a",
                             next=_next_release("fomc")))
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

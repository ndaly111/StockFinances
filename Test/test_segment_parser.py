"""
Download each ticker’s most-recent 10-K via SEC’s JSON feed,
extract business-segment Revenue & Operating-Income tables
(or revenue categories if no GAAP segment op-income), and
save PNG charts in /test.

Works in public GitHub Actions (no 403/404) — no API key needed.
"""

import os
import re
import time
import requests
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

# ──────────────── USER CONFIG ────────────────
TICKERS = ["V", "MSFT", "TSLA"]  
CIK      = {"V":"0001403161","MSFT":"0000789019","TSLA":"0001318605"}
EMAIL    = "ndaly111@gmail.com"
OUT_DIR  = "test"
os.makedirs(OUT_DIR, exist_ok=True)

UA      = f"StockFinancesBot/1.0 (+https://github.com/your-repo; {EMAIL})"
HEADERS = {"User-Agent": UA, "Accept-Encoding": "gzip, deflate"}
SLEEP   = 0.3  # ~3 requests/sec

# ──────────────── NETWORK HELPER ────────────────
def get(url):
    time.sleep(SLEEP)
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r

# ──────────────── FETCH LATEST 10-K via JSON ────────────────
def latest_accession_and_doc(cik_str):
    """Return (accessionNumber, primaryDocument) for latest non-amended 10-K."""
    url = f"https://data.sec.gov/submissions/CIK{cik_str}.json"
    data = get(url).json()
    forms = data["filings"]["recent"]["form"]
    accs  = data["filings"]["recent"]["accessionNumber"]
    docs  = data["filings"]["recent"]["primaryDocument"]
    for form, acc, doc in zip(forms, accs, docs):
        if form == "10-K":  # skip amendments (10-K/A)
            return acc, doc
    raise RuntimeError(f"No 10-K found for CIK {cik_str}")

def filing_html(cik_str):
    acc, doc = latest_accession_and_doc(cik_str)
    base      = str(int(cik_str))             # drop leading zeros
    acc_dash  = acc.replace("-", "")
    url       = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{base}/{acc_dash}/{doc}"
    )
    return get(url).text

# ──────────────── PARSERS ─────────────────────
def tidy(block, metric):
    df = pd.DataFrame(block).iloc[:, :3]
    df.columns = ["Segment", "2024", "2023"]
    df = df.melt(id_vars="Segment", var_name="Year", value_name="Amount")
    df["Metric"] = metric
    df["Amount"] = (
        df.Amount.astype(str)
          .str.replace(r"[^\d\-.]", "", regex=True)
          .astype(float)
          * 1_000_000
    )
    return df

def parse_gaap_segment(soup):
    h = soup.find(string=re.compile(r"Segment\s.*(Results|Information|Operations)", re.I))
    if not h:
        return None
    tbl = h.find_parent().find_next("table")
    raw = pd.read_html(str(tbl), flavor="lxml")[0]
    raw.columns = [str(c).strip() for c in raw.columns]
    rev, opi, mode = [], [], "Revenue"
    for _, row in raw.iterrows():
        first = str(row.iloc[0])
        if re.search(r"Operating\s+Income", first, re.I):
            mode = "Operating Income"
            continue
        if pd.isna(row.iloc[0]) or "Total" in first:
            continue
        (rev if mode=="Revenue" else opi).append(row)
    if rev and opi:
        return pd.concat([tidy(rev, "Revenue"), tidy(opi, "Operating Income")])
    return None

def parse_revenue_categories(soup):
    h = soup.find(string=re.compile(r"revenue.*category", re.I)) \
        or soup.find(string=re.compile(r"revenues.*category", re.I))
    if not h:
        return None
    tbl = h.find_parent().find_next("table")
    raw = pd.read_html(str(tbl), flavor="lxml")[0]
    raw.columns = [str(c).strip() for c in raw.columns]
    rows = [row for _, row in raw.iterrows()
            if pd.notna(row.iloc[0]) and "Total" not in str(row.iloc[0])]
    return tidy(rows, "Revenue") if rows else None

# ──────────────── CHARTING ────────────────────
def chart(df, ticker):
    for metric in df.Metric.unique():
        p = df[df.Metric==metric].pivot(index="Year", columns="Segment", values="Amount")
        ax = p.div(1e9).plot(kind="bar", stacked=True, figsize=(6,4))
        ax.set_title(f"{ticker} – {metric} by segment/category")
        ax.set_ylabel("USD (billions)")
        plt.xticks(rotation=0)
        plt.tight_layout()
        path = f"{OUT_DIR}/{ticker}_{metric.lower().replace(' ','_')}.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print("✔ saved", path)

# ──────────────── MAIN LOOP ───────────────────
for ticker in TICKERS:
    try:
        print("⏬", ticker)
        html = filing_html(CIK[ticker])
        soup = BeautifulSoup(html, "html.parser")

        df = parse_gaap_segment(soup)
        if df is None:
            print("ℹ no GAAP segment—trying revenue categories")
            df = parse_revenue_categories(soup)

        if df is None:
            raise RuntimeError("No segment or category table found")

        print(df.pivot(index=["Year","Segment"], columns="Metric", values="Amount"))
        chart(df, ticker)

    except Exception as e:
        print("❌", ticker, e)

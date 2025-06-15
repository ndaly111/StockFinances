"""
Download each ticker’s most-recent 10-K from the SEC Open-Data S3 mirror,
extract business-segment Revenue and Operating-Income tables (or revenue
categories if no GAAP segment op-income), and save PNG charts in /test.

Works in public GitHub Actions (no 403/404) — no API key needed.
"""

import os
import re
import time
import gzip
from datetime import date

import requests
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

# ──────────────── USER CONFIG ────────────────
TICKERS = ["V", "MSFT", "TSLA"]                 # put your 90 tickers here
CIK     = {"V":"0001403161", "MSFT":"0000789019", "TSLA":"0001318605"}
EMAIL   = "ndaly111@gmail.com"
OUT_DIR = "test"
os.makedirs(OUT_DIR, exist_ok=True)

UA      = f"StockFinancesBot/1.0 (+https://github.com/your-repo; {EMAIL})"
HEADERS = {"User-Agent": UA, "Accept-Encoding": "gzip, deflate"}
SLEEP   = 0.3                                   # ≈3 requests/sec

# ──────────────── NETWORK HELPERS ─────────────
def get(url):
    time.sleep(SLEEP)
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r

# ──────────────── find latest accession via master.idx.gz ────────────────
def latest_accession(cik_str):
    year, qtr = date.today().year, (date.today().month - 1)//3 + 1
    pattern   = f"|{int(cik_str)}|10-K|"
    # search this quarter and up to 7 prior quarters
    for _ in range(8):
        idx_url = (
            f"https://sec-edgar-us.s3.amazonaws.com/Archives/edgar/"
            f"full-index/{year}/QTR{qtr}/master.idx.gz"
        )
        gz = get(idx_url).content
        text = gzip.decompress(gz).decode("latin1")
        # skip header (first 11 lines), read newest first
        for line in reversed(text.splitlines()[11:]):
            if pattern in line:
                parts = line.split("|")
                return parts[4]  # accessionNumber
        # move to previous quarter
        qtr -= 1
        if qtr == 0:
            year -= 1
            qtr = 4
    raise RuntimeError("No 10-K found in last 8 quarters")

# ──────────────── download the actual 10-K HTML ────────────────
def filing_html(cik_str):
    acc       = latest_accession(cik_str)
    base      = cik_str.lstrip("0")
    acc_nodash= acc.replace("-", "")
    # first grab the index page
    idx_url = (
        f"https://sec-edgar-us.s3.amazonaws.com/Archives/edgar/data/"
        f"{base}/{acc_nodash}/{acc}-index.html"
    )
    idx_html = get(idx_url).text
    match    = re.search(r'href="([^"]+\.htm)"', idx_html, re.I)
    if not match:
        raise RuntimeError("Main HTML filing not found in index page")
    main_doc = match.group(1)
    # now grab the main document
    doc_url  = (
        f"https://sec-edgar-us.s3.amazonaws.com/Archives/edgar/data/"
        f"{base}/{acc_nodash}/{main_doc}"
    )
    return get(doc_url).text

# ──────────────── PARSERS ─────────────────────
def tidy(block, metric):
    df = pd.DataFrame(block).iloc[:, :3]       # Segment | 2024 | 2023
    df.columns = ["Segment", "2024", "2023"]
    df = df.melt(id_vars="Segment", var_name="Year", value_name="Amount")
    df["Metric"] = metric
    df["Amount"] = (
        df.Amount.astype(str)
          .str.replace(r"[^\d\-.]", "", regex=True)
          .astype(float)
          * 1_000_000    # values are “in millions”
    )
    return df

def parse_gaap_segment(soup):
    h = soup.find(
        string=re.compile(r"Segment\s.*(Results|Information|Operations)", re.I)
    )
    if not h:
        return None
    tbl = h.find_parent().find_next("table")
    raw = pd.read_html(str(tbl))[0]
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
        return pd.concat([tidy(rev, "Revenue"),
                          tidy(opi, "Operating Income")])
    return None

def parse_revenue_categories(soup):
    h = soup.find(
        string=re.compile(r"revenue.*category", re.I)
    ) or soup.find(string=re.compile(r"revenues.*category", re.I))
    if not h:
        return None
    tbl = h.find_parent().find_next("table")
    raw = pd.read_html(str(tbl))[0]
    raw.columns = [str(c).strip() for c in raw.columns]
    rows = [
        row for _, row in raw.iterrows()
        if not pd.isna(row.iloc[0]) and "Total" not in str(row.iloc[0])
    ]
    return tidy(rows, "Revenue") if rows else None

# ──────────────── CHARTING ────────────────────
def chart(df, ticker):
    for metric in df.Metric.unique():
        p = df[df.Metric==metric].pivot(
            index="Year", columns="Segment", values="Amount"
        )
        ax = p.div(1e9).plot(
            kind="bar", stacked=True, figsize=(6,4)
        )
        ax.set_title(f"{ticker} – {metric} by segment/category")
        ax.set_ylabel("USD (billions)")
        plt.xticks(rotation=0)
        plt.tight_layout()
        file = f"{OUT_DIR}/{ticker}_{metric.lower().replace(' ','_')}.png"
        plt.savefig(file, dpi=150)
        plt.close()
        print("✔ saved", file)

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

        print(df.pivot(
            index=["Year","Segment"], columns="Metric", values="Amount"
        ))
        chart(df, ticker)

    except Exception as e:
        print("❌", ticker, e)

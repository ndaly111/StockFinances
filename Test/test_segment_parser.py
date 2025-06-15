"""
Download each ticker’s most-recent 10-K from the SEC Open-Data S3 mirror,
extract business-segment Revenue and Operating-Income tables (or revenue
categories if no GAAP segment op-income), and save PNG charts in /test.

Works in public GitHub Actions (no 403) — no API key needed.
"""

import os, re, time, requests, pandas as pd, matplotlib.pyplot as plt
from bs4 import BeautifulSoup

# ──────────────── USER CONFIG ────────────────
TICKERS = ["V", "MSFT", "TSLA"]                 # put your 90 tickers here
CIK      = {"V":"0001403161", "MSFT":"0000789019", "TSLA":"0001318605"}  # extend same list
EMAIL    = "ndaly111@gmail.com"
OUT_DIR  = "test"
os.makedirs(OUT_DIR, exist_ok=True)

UA = f"StockFinancesBot/1.0 (+https://github.com/your-repo; {EMAIL})"
HEADERS = {"User-Agent": UA, "Accept-Encoding": "gzip, deflate"}
SLEEP   = 0.3                                   # ≈3 requests/sec

# ──────────────── NETWORK HELPERS ─────────────
def get(url):
    time.sleep(SLEEP)
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status(); return r

def latest_accession(cik):
    url = f"https://sec-edgar-us.s3.amazonaws.com/submissions/CIK{cik}.json"
    js  = get(url).json()
    for form, acc in zip(js["filings"]["recent"]["form"],
                         js["filings"]["recent"]["accessionNumber"]):
        if form == "10-K": return acc
    raise RuntimeError("No recent 10-K found")

def filing_html(cik):
    acc = latest_accession(cik)
    base = cik.lstrip("0"); acc_nodash = acc.replace("-","")
    idx  = get(f"https://sec-edgar-us.s3.amazonaws.com/Archives/edgar/data/"
               f"{base}/{acc_nodash}/{acc}-index.html").text
    main = re.search(r'href="([^"]+\.htm)"', idx, re.I).group(1)
    return get(f"https://sec-edgar-us.s3.amazonaws.com/Archives/edgar/data/"
               f"{base}/{acc_nodash}/{main}").text

# ──────────────── PARSERS ─────────────────────
def tidy(block, metric):
    df = pd.DataFrame(block).iloc[:, :3]                # Segment | 2024 | 2023
    df.columns = ["Segment", "2024", "2023"]
    df = df.melt(id_vars="Segment", var_name="Year", value_name="Amount")
    df["Metric"] = metric
    df["Amount"] = (df.Amount.astype(str)
                    .str.replace(r"[^\d\-.]", "", regex=True)
                    .astype(float) * 1_000_000)         # “in millions”
    return df

def parse_gaap_segment(soup):
    h = soup.find(string=re.compile(r"Segment\s.*(Results|Information|Operations)", re.I))
    if not h: return None
    table = h.find_parent().find_next("table")
    raw   = pd.read_html(str(table))[0]
    raw.columns = [str(c).strip() for c in raw.columns]
    rev, opi, mode = [], [], "Revenue"
    for _, row in raw.iterrows():
        first = str(row.iloc[0])
        if re.search(r"Operating\s+Income", first, re.I): mode = "Operating Income"; continue
        if pd.isna(row.iloc[0]) or "Total" in first: continue
        (rev if mode=="Revenue" else opi).append(row)
    return pd.concat([tidy(rev,"Revenue"), tidy(opi,"Operating Income")]) if rev and opi else None

def parse_revenue_categories(soup):
    h = soup.find(string=re.compile(r"revenue.*category", re.I))
    h = h or soup.find(string=re.compile(r"revenues.*category", re.I))
    if not h: return None
    table = h.find_parent().find_next("table")
    raw   = pd.read_html(str(table))[0]
    raw.columns = [str(c).strip() for c in raw.columns]
    rows  = [row for _, row in raw.iterrows()
             if not pd.isna(row.iloc[0]) and "Total" not in str(row.iloc[0])]
    return tidy(rows,"Revenue") if rows else None

# ──────────────── CHARTING ────────────────────
def chart(df, ticker):
    for m in df.Metric.unique():
        p = df[df.Metric==m].pivot(index="Year", columns="Segment", values="Amount")
        ax = p.div(1e9).plot(kind="bar", stacked=True, figsize=(6,4))
        ax.set_title(f"{ticker} – {m} by segment/category")
        ax.set_ylabel("USD (billions)")
        plt.xticks(rotation=0); plt.tight_layout()
        file = f"{OUT_DIR}/{ticker}_{m.lower().replace(' ','_')}.png"
        plt.savefig(file, dpi=150); plt.close(); print("✔ saved", file)

# ──────────────── MAIN LOOP ───────────────────
for t in TICKERS:
    try:
        print("⏬", t)
        html = filing_html(CIK[t])
        soup = BeautifulSoup(html, "html.parser")

        df = parse_gaap_segment(soup)
        if df is None:
            df = parse_revenue_categories(soup)
            if df is not None: print("ℹ using revenue-category table")

        if df is None:
            raise RuntimeError("No segment or category table found")

        print(df.pivot(index=["Year","Segment"], columns="Metric", values="Amount"))
        chart(df, t)
    except Exception as e:
        print("❌", t, e)

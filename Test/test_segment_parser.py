"""
Test script: pull latest 10-K, extract “Segment Results of Operations” table
(revenue & operating income), print it, and save two PNG charts into /test.

This version uses:
  • primary JSON feed (data.sec.gov)
  • HTML fallback of browse-edgar → detail page
  • polite retry on 403
"""

import os
import re
import time
import requests
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

# ——— Configuration ———
TEST_TICKER = "MSFT"
TEST_DIR    = "test"
os.makedirs(TEST_DIR, exist_ok=True)

MY_EMAIL = "you@example.com"  # SEC requires contact info in UA

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; StockFinancesBot/1.0; "
        f"+https://github.com/your-repo; {MY_EMAIL})"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Encoding": "gzip, deflate",
    "Host": "www.sec.gov"
}

# ——— Hard-coded CIK Map ———
CIK_MAP = {
    "MSFT": "0000789019",
    "AAPL": "0000320193",
    "GOOGL": "0001652044",
    "V":     "0001403161",
    "TSLA":  "0001318605",
}

# ——— Safe GET with single retry on 403 ———
def safe_get(url, headers=HEADERS, timeout=30):
    resp = requests.get(url, headers=headers, timeout=timeout)
    if resp.status_code == 403:
        time.sleep(1)
        resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp

def get_cik(ticker: str) -> str:
    t = ticker.upper()
    if t not in CIK_MAP:
        raise ValueError(f"CIK missing for {t} – please add to CIK_MAP")
    return CIK_MAP[t]

def latest_10k_url(cik: str) -> str:
    """
    Return the primary HTML URL of the most recent 10-K.
    1) Try data.sec.gov JSON feed.
    2) If that fails, scrape browse-edgar HTML + detail page.
    """
    trim = cik.lstrip("0")
    # — Primary: JSON feed —
    feed = f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"
    try:
        d = safe_get(feed).json()
        recent = d["filings"]["recent"]
        for form, acc, doc in zip(recent["form"],
                                  recent["accessionNumber"],
                                  recent["primaryDocument"]):
            if form == "10-K":
                path = acc.replace("-", "")
                return f"https://www.sec.gov/Archives/edgar/data/{trim}/{path}/{doc}"
    except Exception as e:
        print("⚠️  JSON feed failed:", e)

    # — Fallback: browse-edgar HTML —
    list_url = (
        f"https://www.sec.gov/cgi-bin/browse-edgar?"
        f"CIK={cik}&type=10-K&owner=exclude&count=1"
    )
    resp = safe_get(list_url)
    soup = BeautifulSoup(resp.text, "html.parser")

    tbl = soup.find("table", {"class": "tableFile2"})
    if not tbl:
        raise RuntimeError("browse-edgar list table not found")
    rows = [r for r in tbl.find_all("tr") if r.find("td")]
    if not rows:
        raise RuntimeError("no data rows in browse-edgar list")
    link = rows[0].find("a", href=True)
    detail_url = "https://www.sec.gov" + link["href"]

    # — Detail page —
    resp2 = safe_get(detail_url)
    soup2 = BeautifulSoup(resp2.text, "html.parser")
    tbl2 = soup2.find("table", {"class": "tableFile"})
    if not tbl2:
        raise RuntimeError("detail page tableFile not found")
    rows2 = [r for r in tbl2.find_all("tr") if r.find("td")]
    if not rows2:
        raise RuntimeError("no data rows in detail page table")
    doc_link = rows2[0].find("a", href=True)["href"]
    return "https://www.sec.gov" + doc_link

def extract_segment_table(html: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    # Look for any heading with “Segment” + (“Results”|“Information”|“Operations”)
    heading = soup.find(string=re.compile(
        r"Segment\s.*(Results|Information|Operations)", re.I))
    if not heading:
        raise RuntimeError("Segment table heading not found")
    table = heading.find_parent().find_next("table")
    raw = pd.read_html(str(table), flavor="lxml")[0]
    raw.columns = [str(c).strip() for c in raw.columns]
    raw = raw.dropna(axis=1, how="all")

    revenue, op_inc = [], []
    mode = "Revenue"
    for _, row in raw.iterrows():
        first = str(row.iloc[0])
        if re.search(r"Operating\s+Income", first, re.I):
            mode = "Operating Income"
            continue
        if pd.isna(row.iloc[0]) or "Total" in first:
            continue
        (revenue if mode=="Revenue" else op_inc).append(row)

    def tidy(block, label):
        df = pd.DataFrame(block).iloc[:, :3]
        df.columns = ["Segment", "2024", "2023"]
        df = df.melt(id_vars="Segment", var_name="Year", value_name="Amount")
        df["Metric"] = label
        return df

    df = pd.concat([tidy(revenue, "Revenue"),
                    tidy(op_inc,  "Operating Income")],
                   ignore_index=True)
    df["Amount"] = (df["Amount"].astype(str)
                    .str.replace(r"[^\d\-.]", "", regex=True)
                    .astype(float) * 1_000_000)
    return df

def plot_segment_chart(df: pd.DataFrame, ticker: str):
    for metric in ["Revenue", "Operating Income"]:
        sub   = df[df["Metric"]==metric]
        pivot= sub.pivot(index="Year", columns="Segment", values="Amount")
        ax    = pivot.div(1e9).plot(kind="bar", stacked=True, figsize=(6,4))
        ax.set_title(f"{ticker} – {metric} by Segment")
        ax.set_ylabel("USD (Billions)")
        plt.xticks(rotation=0)
        plt.tight_layout()
        path = os.path.join(
            TEST_DIR, f"{ticker}_{metric.lower().replace(' ', '_')}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print("✔ chart saved →", path)

def run_test(ticker: str):
    print(f"🔍 extracting segment data for {ticker}")
    cik = get_cik(ticker)
    url = latest_10k_url(cik)
    print("   filing URL:", url)
    html = safe_get(url).text
    df   = extract_segment_table(html)

    print("\n📊 tidy segment table:\n")
    print(df.pivot(index=["Year","Segment"], columns="Metric", values="Amount"))

    plot_segment_chart(df, ticker)

if __name__ == "__main__":
    run_test(TEST_TICKER)

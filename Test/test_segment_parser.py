"""
Pull latest 10-K, extract the “Segment Results of Operations” table,
print a tidy DataFrame, and drop two PNG charts into /test.

Runs cleanly inside GitHub Actions (headers, retry, fallback).

Author: ChatGPT – 9 Jun 2025
"""

import os, re, time, json, requests, pandas as pd, matplotlib.pyplot as plt
from bs4 import BeautifulSoup

# ——— Configuration ———
TEST_TICKER = "MSFT"          # change to test another ticker
TEST_DIR    = "test"
os.makedirs(TEST_DIR, exist_ok=True)

MY_EMAIL = "you@example.com"  # put your email here – SEC requires contact info

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; StockFinancesBot/1.0; "
        f"+https://github.com/your-repo; {MY_EMAIL})"
    ),
    "Accept-Encoding": "gzip, deflate",
    "Accept": "application/json, text/html",
    "Host": "www.sec.gov"
}

# ——— Hard-coded CIKs (extend as needed) ———
CIK_MAP = {
    "MSFT": "0000789019",
    "AAPL": "0000320193",
    "GOOGL": "0001652044",
    "V":     "0001403161",
    "TSLA":  "0001318605",
}

# ——— Helper: polite GET with single retry on 403 ———
def safe_get(url, headers, timeout=30):
    resp = requests.get(url, headers=headers, timeout=timeout)
    if resp.status_code == 403:          # wait & try once more
        time.sleep(1.0)
        resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp

# ——— Step 1: map ticker → CIK ———
def get_cik(ticker: str) -> str:
    ticker = ticker.upper()
    if ticker not in CIK_MAP:
        raise ValueError(f"CIK missing for {ticker}. Add it to CIK_MAP.")
    return CIK_MAP[ticker]

# ——— Step 2: find latest 10-K HTML, with two fallbacks ———
def latest_10k_url(cik: str) -> str:
    """Return full URL of the primary 10-K HTML file."""
    cik_trim = cik.lstrip("0")
    # 2-a  primary – data.sec.gov JSON feed
    feed_url = f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"
    try:
        data = safe_get(feed_url, HEADERS).json()
        recent = data["filings"]["recent"]
        for frm, acc, doc in zip(recent["form"],
                                 recent["accessionNumber"],
                                 recent["primaryDocument"]):
            if frm == "10-K":
                return (
                    f"https://www.sec.gov/Archives/edgar/data/"
                    f"{cik_trim}/{acc.replace('-', '')}/{doc}"
                )
    except Exception as e:
        print("⚠️  primary feed failed:", e)

    # 2-b  backup – browse-edgar JSON
    be_url = (
        "https://www.sec.gov/cgi-bin/browse-edgar"
        f"?CIK={cik}&type=10-K&owner=exclude&count=1&output=json"
    )
    be_json = safe_get(be_url, HEADERS).json()
    filings = be_json["filings"]
    if not filings:
        raise RuntimeError("No 10-K found via browse-edgar backup.")
    acc = filings[0]["accessionNumber"]
    doc = filings[0]["filingHREF"].split("/")[-1] + ".txt"
    return (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{cik_trim}/{acc.replace('-', '')}/{doc}"
    )

# ——— Step 3: scrape the segment table ———
def extract_segment_table(html: str) -> pd.DataFrame:
    soup    = BeautifulSoup(html, "html.parser")
    heading = soup.find(string=re.compile(
        r"Segment\s.*(Results|Information|Operations)", re.I))
    if not heading:
        raise RuntimeError("Segment table heading not found.")
    table = heading.find_parent().find_next("table")
    raw   = pd.read_html(str(table), flavor="lxml")[0]
    raw.columns = [str(c).strip() for c in raw.columns]
    raw = raw.dropna(axis=1, how="all")

    # split rows into Revenue vs Operating Income blocks
    revenue, op_inc, current = [], [], "Revenue"
    for _, row in raw.iterrows():
        txt0 = str(row.iloc[0])
        if re.search(r"Operating\s+Income", txt0, re.I):
            current = "Operating Income"; continue
        if pd.isna(row.iloc[0]) or "Total" in txt0: continue
        (revenue if current == "Revenue" else op_inc).append(row)

    def tidy(block, label):
        df = pd.DataFrame(block).iloc[:, :3]     # Segment | 2024 | 2023
        df.columns = ["Segment", "2024", "2023"]
        df = df.melt(id_vars="Segment",
                     var_name="Year",
                     value_name="Amount")
        df["Metric"] = label
        return df

    df = pd.concat([tidy(revenue, "Revenue"),
                    tidy(op_inc, "Operating Income")],
                   ignore_index=True)
    df["Amount"] = (df["Amount"].astype(str)
                    .str.replace(r"[^\d\-.]", "", regex=True)
                    .astype(float) * 1_000_000)     # values “in millions”
    return df

# ——— Step 4: basic charts ———
def plot_segment_chart(df: pd.DataFrame, ticker: str):
    for metric in ["Revenue", "Operating Income"]:
        sub   = df[df["Metric"] == metric]
        pivot = sub.pivot(index="Year", columns="Segment", values="Amount")
        ax    = pivot.div(1e9).plot(kind="bar", stacked=True, figsize=(6, 4))
        ax.set_title(f"{ticker} – {metric} by Segment")
        ax.set_ylabel("USD (Billions)")
        plt.xticks(rotation=0)
        plt.tight_layout()
        path = os.path.join(
            TEST_DIR, f"{ticker}_{metric.lower().replace(' ', '_')}.png")
        plt.savefig(path, dpi=150); plt.close()
        print("✔  chart saved →", path)

# ——— Orchestrator ———
def run_test(ticker: str):
    print(f"🔍  extracting segment data for {ticker}")
    cik  = get_cik(ticker)
    url  = latest_10k_url(cik)
    print("    filing URL:", url)
    html = safe_get(url, HEADERS).text
    df   = extract_segment_table(html)

    print("\n📊  tidy segment table:\n")
    print(df.pivot(index=["Year", "Segment"],
                   columns="Metric",
                   values="Amount"))

    plot_segment_chart(df, ticker)

# ——— Entry point ———
if __name__ == "__main__":
    run_test(TEST_TICKER)

"""
Test script: pull the latest 10-K, extract the “Segment Results of Operations”
table (revenue & operating income), print it, and save two charts in /test.

Author: ChatGPT, 9 Jun 2025
"""

import os, re, json, time, requests, pandas as pd, matplotlib.pyplot as plt
from bs4 import BeautifulSoup

# ——— Config ———
TEST_TICKER = "MSFT"
TEST_DIR    = "test"
os.makedirs(TEST_DIR, exist_ok=True)

HEADERS = {
    # SEC says: identify yourself & give contact email
    "User-Agent": "SegmentTestBot/1.0 (github.com/your-repo; contact you@example.com)",
    "Accept-Encoding": "gzip, deflate",
    "Host": "www.sec.gov"
}

# ——— Hard-coded CIKs ———
CIK_MAP = {
    "MSFT": "0000789019",
    "AAPL": "0000320193",
    "GOOGL": "0001652044",
    "V":     "0001403161",
    "TSLA":  "0001318605",
}

# ——— Helpers ———
def get_cik(ticker: str) -> str:
    ticker = ticker.upper()
    if ticker not in CIK_MAP:
        raise ValueError(f"CIK missing for {ticker}. Add it to CIK_MAP.")
    return CIK_MAP[ticker]

def latest_10k_url(cik: str) -> str:
    """
    Use data.sec.gov/submissions/CIK####.json to get the most recent 10-K HTML.
    Returns full URL of the primary document.
    """
    cik_nolead = cik.lstrip("0")
    sub_url = f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"
    resp = requests.get(sub_url, headers=HEADERS, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"SEC submissions feed failed ({resp.status_code})")

    filings = resp.json()["filings"]["recent"]
    for form, acc, doc in zip(filings["form"],
                              filings["accessionNumber"],
                              filings["primaryDocument"]):
        if form == "10-K":
            acc_no_dashes = acc.replace("-", "")
            html_url = (f"https://www.sec.gov/Archives/edgar/data/"
                        f"{cik_nolead}/{acc_no_dashes}/{doc}")
            return html_url
    raise RuntimeError("No 10-K found in recent filings.")

def extract_segment_table(html: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")

    # Look for a heading containing “Segment” and “Income” or “Results”
    heading = soup.find(string=re.compile(
        r"Segment\s.+(Results|Information|Operations)", re.I))
    if not heading:
        raise RuntimeError("Segment table heading not found.")

    table = heading.find_parent().find_next("table")
    raw = pd.read_html(str(table), flavor="lxml")[0]
    raw.columns = [str(c).strip() for c in raw.columns]
    raw = raw.dropna(axis=1, how="all")

    # Split into revenue vs operating-income blocks
    revenue, op_inc, current = [], [], "Revenue"
    for _, row in raw.iterrows():
        txt_row0 = str(row.iloc[0])
        if re.search(r"Operating\s+Income", txt_row0, re.I):
            current = "Operating Income"
            continue
        if pd.isna(row.iloc[0]) or "Total" in txt_row0:
            continue
        (revenue if current == "Revenue" else op_inc).append(row)

    def tidy(block, label):
        df = pd.DataFrame(block).iloc[:, :3]       # Segment, 2024, 2023
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
                    .astype(float) * 1_000_000)  # filings are “in millions”
    return df

def plot_segment_chart(df: pd.DataFrame, ticker: str):
    for metric in ["Revenue", "Operating Income"]:
        sub = df[df["Metric"] == metric]
        pivot = sub.pivot(index="Year", columns="Segment", values="Amount")
        ax = pivot.div(1e9).plot(kind="bar", stacked=True, figsize=(6, 4))
        ax.set_title(f"{ticker} – {metric} by Segment")
        ax.set_ylabel("USD (Billions)")
        plt.xticks(rotation=0)
        plt.tight_layout()
        fname = os.path.join(
            TEST_DIR, f"{ticker}_{metric.lower().replace(' ', '_')}.png")
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"✔ Chart saved: {fname}")

# ——— Main ———
def run_test(ticker: str):
    print(f"🔍 Extracting segment data for {ticker}")
    cik  = get_cik(ticker)
    url  = latest_10k_url(cik)
    print("   Filing URL:", url)
    html = requests.get(url, headers=HEADERS, timeout=30).text
    df   = extract_segment_table(html)

    print("\n📊 Tidy Segment Table:\n")
    print(df.pivot(index=["Year", "Segment"],
                   columns="Metric",
                   values="Amount"))

    plot_segment_chart(df, ticker)

if __name__ == "__main__":
    run_test(TEST_TICKER)

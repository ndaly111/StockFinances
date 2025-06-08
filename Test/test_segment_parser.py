import os
import re
import requests
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

# ——— Configuration ———
TEST_TICKER = "MSFT"
TEST_DIR = "test"
os.makedirs(TEST_DIR, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (SegmentTest/1.0)"}

# ——— Hardcoded CIK lookup ———
CIK_MAP = {
    "MSFT": "0000789019",  # Microsoft
    "AAPL": "0000320193",  # Apple
    "GOOGL": "0001652044", # Alphabet
    "V":     "0001403161", # Visa
    "TSLA":  "0001318605", # Tesla
}

def get_cik(ticker):
    ticker = ticker.upper()
    if ticker not in CIK_MAP:
        raise ValueError(f"CIK not defined for ticker: {ticker}")
    return CIK_MAP[ticker]

def get_latest_10k_html(cik):
    index_url = f"https://www.sec.gov/cgi-bin/browse-edgar?CIK={cik}&type=10-K&count=1"
    soup = BeautifulSoup(requests.get(index_url, headers=HEADERS).text, "html.parser")
    doc_link = soup.find("a", string="Documents")["href"]
    detail_url = f"https://www.sec.gov{doc_link}"
    detail_page = requests.get(detail_url, headers=HEADERS).text
    detail_soup = BeautifulSoup(detail_page, "html.parser")
    filing_href = detail_soup.find("a", id="file0")["href"]
    return f"https://www.sec.gov{filing_href}"

def extract_segment_table(html):
    soup = BeautifulSoup(html, "html.parser")
    heading = soup.find(string=re.compile("Segment Results of Operations", re.I))
    if not heading:
        raise ValueError("Segment Results section not found")
    table = heading.find_parent().find_next("table")
    raw = pd.read_html(str(table))[0]

    raw.columns = [str(c).strip() for c in raw.columns]
    raw = raw.dropna(axis=1, how="all")

    # Clean & detect revenue vs. operating income sections
    revenue, op_inc = [], []
    current = "Revenue"
    for _, row in raw.iterrows():
        if "Operating Income" in row.astype(str).str.contains("Operating Income", case=False).any():
            current = "Operating Income"
            continue
        if pd.isna(row[0]) or "Total" in str(row[0]):
            continue
        target = revenue if current == "Revenue" else op_inc
        target.append(row)

    def tidy(block, label):
        df = pd.DataFrame(block).iloc[:, :3]
        df.columns = ["Segment", "2024", "2023"]
        df = df.melt(id_vars="Segment", var_name="Year", value_name="Amount")
        df["Metric"] = label
        return df

    rev_df = tidy(revenue, "Revenue")
    opi_df = tidy(op_inc, "Operating Income")
    combined = pd.concat([rev_df, opi_df], ignore_index=True)
    combined["Amount"] = (combined["Amount"]
                          .astype(str)
                          .str.replace(r"[^\d.-]", "", regex=True)
                          .astype(float) * 1e6)
    return combined

def plot_segment_chart(df, ticker):
    for metric in ["Revenue", "Operating Income"]:
        subset = df[df["Metric"] == metric]
        pivot = subset.pivot(index="Year", columns="Segment", values="Amount")
        ax = pivot.div(1e9).plot(kind="bar", stacked=True, figsize=(6, 4))
        ax.set_title(f"{ticker} – {metric} by Segment")
        ax.set_ylabel("USD (Billions)")
        plt.tight_layout()
        fname = os.path.join(TEST_DIR, f"{ticker}_{metric.lower().replace(' ', '_')}.png")
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"[✔] Chart saved: {fname}")

# ——— Main test run ———

def run_test(ticker):
    print(f"🔍 Testing segment data extraction for {ticker}")
    cik = get_cik(ticker)
    url = get_latest_10k_html(cik)
    print(f"📄 Fetching filing: {url}")
    html = requests.get(url, headers=HEADERS).text
    df = extract_segment_table(html)
    print("\n📊 Extracted Segment Table:\n")
    print(df.pivot(index=["Year", "Segment"], columns="Metric", values="Amount"))
    plot_segment_chart(df, ticker)

if __name__ == "__main__":
    run_test(TEST_TICKER)

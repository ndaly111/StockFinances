"""
Polite SEC test script
---------------------
1. Download the most-recent quarterly master.idx
2. Locate the latest 10-K for the given CIK
3. Fetch that filing's primary HTML
4. Parse the “Segment Results of Operations” table
5. Print a tidy DataFrame & save two charts in /test

Complies with SEC fair-access policy (≤10 req/s, proper UA, caching).
"""

import os, re, time, calendar, requests, pandas as pd, matplotlib.pyplot as plt
from datetime import date
from bs4 import BeautifulSoup

# ——— SETTINGS ———
TEST_TICKER = "MSFT"               # change as needed
TEST_DIR    = "test"
os.makedirs(TEST_DIR, exist_ok=True)

MY_EMAIL = "ndaly111@gmail.com"       # REQUIRED by SEC fair-access policy

UA = (
    "StockFinancesBot/1.0 "
    f"(https://github.com/your-repo; {MY_EMAIL})"
)
HEADERS = {
    "User-Agent": UA,
    "Accept-Encoding": "gzip, deflate",
    "Accept": "text/html, application/xhtml+xml, */*"
}
REQUEST_DELAY = 0.12               # 1/10  sec  →  ≤10 req/s

# ——— STATIC CIK MAP ———
CIK_MAP = {
    "MSFT": "0000789019",
    "AAPL": "0000320193",
    "GOOGL": "0001652044",
    "TSLA": "0001318605",
    "V":    "0001403161",
}

# ——— polite GET wrapper ———
def get(url, **kw):
    """GET with mandatory sleep to respect SEC 10 r/s limit."""
    time.sleep(REQUEST_DELAY)
    resp = requests.get(url, headers=HEADERS, timeout=30, **kw)
    resp.raise_for_status()
    return resp

# ——— 1. Which quarter are we in? ———
def current_qtr_yyyy():
    today = date.today()
    qtr   = (today.month - 1) // 3 + 1
    return today.year, qtr

# ——— 2. Download master.idx (cached) ———
def fetch_master_idx(year: int, qtr: int) -> list[str]:
    """
    Return list of lines (str) from master.idx.
    Uses a local cache in /test/master-YYYY-Q#.idx to save SEC bandwidth.
    """
    cache_path = os.path.join(TEST_DIR, f"master-{year}-Q{qtr}.idx")
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="latin1") as f:
            return f.readlines()

    url = (
        f"https://www.sec.gov/Archives/edgar/full-index/"
        f"{year}/QTR{qtr}/master.idx"
    )
    print(f"📥  downloading master.idx {year}-Q{qtr}")
    txt = get(url).text
    with open(cache_path, "w", encoding="latin1") as f:
        f.write(txt)
    return txt.splitlines()

# ——— 3. Find latest 10-K path for this CIK ———
def latest_10k_path(cik: str) -> str:
    year, qtr = current_qtr_yyyy()

    # Search current quarter then roll back until we find a 10-K
    for _ in range(8):                        # up to 2 years back
        lines = fetch_master_idx(year, qtr)
        # master.idx:  header lines … then `YYYY-MM-DD|COMPANY|CIK|FORM|PATH`
        matches = [l for l in lines if f"|{cik.lstrip('0')}|" in l and "|10-K|" in l]
        if matches:
            latest = matches[-1]              # last occurrence is latest
            path   = latest.strip().split("|")[-1]
            return path

        # move to previous quarter
        qtr -= 1
        if qtr == 0:
            qtr, year = 4, year - 1

    raise RuntimeError("No 10-K found in last 2 years via master.idx")

# ——— 4. Build full filing URL ———
def filing_html_url(path: str) -> str:
    # Some 10-Ks are .htm, some .txt; we’ll always fetch the .htm first
    base = "https://www.sec.gov/Archives/"
    if path.endswith(".txt"):                 # 1-file submission
        return base + path
    # multi-file: replace -index.htm with .htm if needed
    return base + path

# ——— 5. Extract segment table ———
def extract_segment_table(html: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    heading = soup.find(string=re.compile(
        r"Segment\s.*(Results|Information|Operations)", re.I))
    if not heading:
        raise RuntimeError("Segment heading not found.")
    table = heading.find_parent().find_next("table")
    raw   = pd.read_html(str(table), flavor="lxml")[0]
    raw.columns = [str(c).strip() for c in raw.columns]
    raw = raw.dropna(axis=1, how="all")

    revenue, op_inc, mode = [], [], "Revenue"
    for _, row in raw.iterrows():
        first = str(row.iloc[0])
        if re.search(r"Operating\s+Income", first, re.I):
            mode = "Operating Income"; continue
        if pd.isna(row.iloc[0]) or "Total" in first: continue
        (revenue if mode=="Revenue" else op_inc).append(row)

    def tidy(block, metric):
        df = pd.DataFrame(block).iloc[:, :3]          # Segment | 2024 | 2023
        df.columns = ["Segment", "2024", "2023"]
        df = df.melt(id_vars="Segment", var_name="Year",
                     value_name="Amount")
        df["Metric"] = metric
        return df

    df = pd.concat([tidy(revenue, "Revenue"),
                    tidy(op_inc,  "Operating Income")],
                   ignore_index=True)
    df["Amount"] = (df["Amount"].astype(str)
                    .str.replace(r"[^\d\-.]", "", regex=True)
                    .astype(float) * 1_000_000)   # “in millions”
    return df

# ——— 6. Charts ———
def plot_charts(df: pd.DataFrame, ticker: str):
    for metric in ["Revenue", "Operating Income"]:
        sub   = df[df["Metric"] == metric]
        pivot = sub.pivot(index="Year", columns="Segment", values="Amount")
        ax    = pivot.div(1e9).plot(kind="bar", stacked=True, figsize=(6,4))
        ax.set_title(f"{ticker} – {metric} by Segment")
        ax.set_ylabel("USD (Billions)")
        plt.xticks(rotation=0)
        plt.tight_layout()
        p = os.path.join(TEST_DIR,
                         f"{ticker}_{metric.lower().replace(' ', '_')}.png")
        plt.savefig(p, dpi=150); plt.close()
        print("✔ chart saved →", p)

# ——— MAIN orchestrator ———
def run_test(ticker: str):
    print(f"🔍  extracting segment data for {ticker}")
    cik   = CIK_MAP[ticker.upper()]
    path  = latest_10k_path(cik)
    url   = filing_html_url(path)
    print("    filing URL:", url)

    html  = get(url).text
    df    = extract_segment_table(html)

    print("\n📊  tidy segment table:\n")
    print(df.pivot(index=["Year","Segment"],
                   columns="Metric",
                   values="Amount"))

    plot_charts(df, ticker)

if __name__ == "__main__":
    run_test(TEST_TICKER)

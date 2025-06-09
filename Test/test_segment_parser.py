import os, re, time, requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
from bs4 import BeautifulSoup

# —————————————————————————————————— CONFIG —————————————————————————————————— #
TICKERS = ["MSFT", "AAPL", "GOOGL"]  # test tickers
OUT_DIR = "test"
os.makedirs(OUT_DIR, exist_ok=True)

MY_EMAIL = "you@example.com"  # SEC requires contact info
HEADERS = {
    "User-Agent": f"StockFinancesBot/1.0 (+https://github.com/your-repo; {MY_EMAIL})",
    "Accept-Encoding": "gzip, deflate"
}
DELAY_S = 0.25  # polite delay (~4 requests/sec)

CIK = {
    "MSFT": "0000789019",
    "AAPL": "0000320193",
    "GOOGL": "0001652044"
}

# ———————————————————————————— NETWORK HELPERS ———————————————————————————— #
def get(url, **kw):
    time.sleep(DELAY_S)
    r = requests.get(url, headers=HEADERS, timeout=30, **kw)
    if r.status_code == 403:
        time.sleep(1.0)
        r = requests.get(url, headers=HEADERS, timeout=30, **kw)
    r.raise_for_status()
    return r

# ———————————————————————————— FETCH 10-K URL ———————————————————————————— #
def latest_10k_from_json(cik_padded):
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    j = get(url).json()
    for form, acc, doc in zip(j["filings"]["recent"]["form"],
                              j["filings"]["recent"]["accessionNumber"],
                              j["filings"]["recent"]["primaryDocument"]):
        if form == "10-K":
            acc_no_dash = acc.replace("-", "")
            base = cik_padded.lstrip("0")
            return f"https://www.sec.gov/Archives/edgar/data/{base}/{acc_no_dash}/{doc}"
    raise RuntimeError("No 10-K found in recent filings")

# ———————————————————————————— FALLBACK TO master.idx ———————————————————————————— #
def current_qtr():
    m = date.today().month
    return (date.today().year, (m - 1) // 3 + 1)

def master_idx_lines(year, qtr):
    name = f"{OUT_DIR}/master-{year}-Q{qtr}.idx"
    if os.path.exists(name):
        with open(name, "r", encoding="latin1") as f:
            return f.readlines()
    url = f"https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{qtr}/master.idx"
    txt = get(url).text
    with open(name, "w", encoding="latin1") as f:
        f.write(txt)
    return txt.splitlines()

def latest_10k_from_master(cik_padded):
    year, qtr = current_qtr()
    for _ in range(8):
        for line in master_idx_lines(year, qtr)[11:]:
            if f"|{int(cik_padded)}|" in line and "|10-K|" in line:
                path = line.rsplit("|", 1)[-1].strip()
                return "https://www.sec.gov/Archives/" + path
        qtr -= 1
        if qtr == 0:
            year -= 1
            qtr = 4
    raise RuntimeError("10-K not found via master.idx")

# ———————————————————————————— TABLE PARSING + CHARTS ———————————————————————————— #
def parse_segment_table(html):
    soup = BeautifulSoup(html, "html.parser")
    head = soup.find(string=re.compile(r"Segment\s.*(Results|Operations|Information)", re.I))
    if not head:
        raise RuntimeError("Segment heading not found")
    table = head.find_parent().find_next("table")
    df = pd.read_html(str(table))[0]
    df.columns = [str(c).strip() for c in df.columns]
    rev, opi, mode = [], [], "Revenue"
    for _, row in df.iterrows():
        cell = str(row.iloc[0])
        if re.search(r"Operating\s+Income", cell, re.I):
            mode = "Operating Income"
            continue
        if pd.isna(row.iloc[0]) or "Total" in cell:
            continue
        (rev if mode == "Revenue" else opi).append(row)
    def tidy(block, metric):
        d = pd.DataFrame(block).iloc[:, :3]
        d.columns = ["Segment", "2024", "2023"]
        d = d.melt(id_vars="Segment", var_name="Year", value_name="Amount")
        d["Metric"] = metric
        return d
    out = pd.concat([tidy(rev, "Revenue"), tidy(opi, "Operating Income")])
    out["Amount"] = (out["Amount"].astype(str)
                     .str.replace(r"[^\d\-.]", "", regex=True)
                     .astype(float) * 1_000_000)
    return out

def save_charts(df, ticker):
    for metric in ["Revenue", "Operating Income"]:
        pvt = df[df["Metric"] == metric].pivot(index="Year", columns="Segment", values="Amount")
        ax = pvt.div(1e9).plot(kind="bar", stacked=True, figsize=(6, 4))
        ax.set_title(f"{ticker} – {metric} by Segment")
        ax.set_ylabel("USD (Billions)")
        plt.xticks(rotation=0)
        plt.tight_layout()
        path = f"{OUT_DIR}/{ticker}_{metric.lower().replace(' ', '_')}.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print("✔ Chart saved:", path)

# ———————————————————————————— MAIN TEST RUN ———————————————————————————— #
def run_one(ticker):
    cik = CIK[ticker]
    try:
        url = latest_10k_from_json(cik)
    except Exception as e:
        print(f"⚠ JSON path failed for {ticker}: {e}")
        url = latest_10k_from_master(cik)
    print(f"🔗 {ticker} 10-K URL: {url}")
    html = get(url).text
    df = parse_segment_table(html)
    print(df.pivot(index=["Year", "Segment"], columns="Metric", values="Amount"))
    save_charts(df, ticker)

if __name__ == "__main__":
    for t in TICKERS:
        run_one(t)

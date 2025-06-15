import os, re, time
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from sec_edgar_downloader import Downloader

# ——— CONFIG ———
TICKERS = ["MSFT", "AAPL", "GOOGL"]
OUT_DIR = "test"
os.makedirs(OUT_DIR, exist_ok=True)

EMAIL = "ndaly111@gmail.com"  # SEC contact
DOWNLOADER = Downloader("StockFinancesBot", EMAIL)

# ——— PARSING SEGMENT TABLE ———
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

# ——— SAVE CHART ———
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

# ——— MAIN ———
def run_one(ticker):
    print(f"📥 Downloading {ticker} 10-K...")
    DOWNLOADER.get("10-K", ticker, amount=1)
    folder = os.path.join(DOWNLOADER._save_directory, ticker, "10-K")
    files = sorted(os.listdir(folder), reverse=True)
    if not files:
        raise RuntimeError(f"No 10-K found for {ticker}")
    path = os.path.join(folder, files[0])
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()
    df = parse_segment_table(html)
    print(df.pivot(index=["Year", "Segment"], columns="Metric", values="Amount"))
    save_charts(df, ticker)

if __name__ == "__main__":
    for t in TICKERS:
        try:
            run_one(t)
        except Exception as e:
            print(f"❌ {t} failed: {e}")

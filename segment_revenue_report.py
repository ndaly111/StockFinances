#!/usr/bin/env python
"""
segment_revenue_report.py
=========================
Pull segment-level Revenue & Operating Income for any US ticker
from the SEC “companyconcept” API and save CSV + chart.

Changelog 2025-07-31:
  • Added broader GAAP tag fallbacks (…BySegment).
  • Added safe-exit guard when no segment data is returned.
  • Optional one-line switch for verbose diagnostics.
"""
import json, time, re, os
from datetime import date
from dateutil import parser as dtp

import requests
import pandas as pd
import matplotlib.pyplot as plt

# ── USER SETTINGS ──────────────────────────────────────────────────────────────
TICKERS     = ["AAPL", "MSFT", "TSLA"]        # add any tickers here
USER_AGENT  = "SegmentScraper/1.2 (you@real-email.com)"  # SEC requires contact
PAUSE_SEC   = 0.25
YEARS_BACK  = 3
CATALOG_DIR = "segment_catalogs"              # folder to store JSON catalogs

# Broadened GAAP tag lists – most common first
REV_TAGS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "SalesRevenueNet",
    "Revenues",
    "NetSalesBySegment",
    "NetRevenueByReportableSegment",
]
OPINC_TAGS = [
    "SegmentReportingInformationOperatingIncomeLoss",
    "OperatingIncomeLoss",
    "OperatingIncomeLossBySegment",
    "OperatingIncomeByReportableSegment",
]

AXIS = "StatementBusinessSegmentsAxis"
VERBOSE = False        # ← flip to True if you want diagnostic printouts
# ───────────────────────────────────────────────────────────────────────────────

# ── helpers ────────────────────────────────────────────────────────────────────
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def load_ticker_cik_map() -> dict[str, str]:
    url = "https://www.sec.gov/files/company_tickers.json"
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    r.raise_for_status()
    data = r.json()
    return {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in data.values()}

def fetch_concept(cik: str, concept: str) -> dict | None:
    url = (f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/"
           f"{concept}.json?dimension={AXIS}")
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    if r.status_code == 200:
        return r.json()
    return None

def select_concept_json(cik: str, tag_list: list[str]) -> dict | None:
    for tag in tag_list:
        d = fetch_concept(cik, tag)
        if d and "units" in d:
            return d
        time.sleep(PAUSE_SEC)
    return None

def json_to_df(data: dict, value_label: str) -> pd.DataFrame:
    recs = []
    for unit, items in data.get("units", {}).items():
        if unit != "USD":
            continue
        for it in items:
            if it.get("segment") is None:
                continue          # skip consolidated totals
            if it.get("form") not in ("10-K", "10-Q"):
                continue
            end = dtp.parse(it["end"]).date()
            val = it["val"] / 1_000_000            # → millions
            member = it["segment"]["member"]
            seg = re.sub(r".*?:", "", member).replace("Member", "")
            recs.append({"segment": seg, "end": end, value_label: val})
    return pd.DataFrame(recs)

def fiscal_year(d: date) -> int:                # naive FY
    return d.year

def keep_last_n_fy(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    keep = sorted({fiscal_year(x) for x in df["end"]}, reverse=True)[:n]
    return df[df["end"].apply(lambda x: fiscal_year(x) in keep)].copy()

# ── segment-catalog persistence ────────────────────────────────────────────────
def load_catalog(ticker: str) -> dict:
    path = os.path.join(CATALOG_DIR, f"{ticker}_segment_catalog.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_catalog(ticker: str, cat: dict) -> None:
    ensure_dir(CATALOG_DIR)
    path = os.path.join(CATALOG_DIR, f"{ticker}_segment_catalog.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cat, f, indent=2)

def update_catalog(ticker: str, df: pd.DataFrame) -> dict:
    cat = load_catalog(ticker)
    for _, row in df.iterrows():
        seg, end = row["segment"], str(row["end"])
        if seg not in cat:
            cat[seg] = {"first_seen": end, "last_seen": end}
        else:
            cat[seg]["first_seen"] = min(cat[seg]["first_seen"], end)
            cat[seg]["last_seen"]  = max(cat[seg]["last_seen"],  end)
    save_catalog(ticker, cat)
    return cat
# ───────────────────────────────────────────────────────────────────────────────

def build_ttm(rev_df: pd.DataFrame, op_df: pd.DataFrame) -> pd.DataFrame:
    if rev_df.empty or op_df.empty:
        return pd.DataFrame()
    fy_end = rev_df["end"].max()
    q_end  = op_df["end"].max()
    if fy_end == q_end:
        return pd.DataFrame()
    segs = set(rev_df["segment"]) | set(op_df["segment"])
    recs = []
    for s in segs:
        r_fy = rev_df[(rev_df["segment"]==s)&(rev_df["end"]==fy_end)]["revenue"].sum()
        r_q  = rev_df[(rev_df["segment"]==s)&(rev_df["end"]==q_end)]["revenue"].sum()
        o_fy = op_df[(op_df["segment"]==s)&(op_df["end"]==fy_end)]["op_income"].sum()
        o_q  = op_df[(op_df["segment"]==s)&(op_df["end"]==q_end)]["op_income"].sum()
        recs.append({"segment": s, "end": q_end,
                     "revenue": r_fy + r_q, "op_income": o_fy + o_q})
    return pd.DataFrame(recs)

def plot_segments(df: pd.DataFrame, ticker: str) -> None:
    if df.empty:
        print(f"[{ticker}] nothing to chart.")
        return
    label = "TTM" if "TTM" in df["fiscal_label"].values else str(
        max(int(x) for x in df["fiscal_label"] if x.isdigit()))
    d = df[df["fiscal_label"] == label]

    metrics = [m for m in ("revenue", "op_income") if m in d.columns]
    d_long = d.melt(id_vars=["segment"], value_vars=metrics,
                    var_name="metric", value_name="val")
    segs = d_long["segment"].unique()
    x = range(len(segs)); width = 0.35

    plt.figure(figsize=(max(8, 0.9*len(segs)), 6))
    if "revenue" in metrics:
        revs = [d_long[(d_long["metric"]=="revenue")&(d_long["segment"]==s)]["val"].sum() for s in segs]
        plt.bar(x, revs, width, label="Revenue")
    if "op_income" in metrics:
        ops = [d_long[(d_long["metric"]=="op_income")&(d_long["segment"]==s)]["val"].sum() for s in segs]
        plt.bar([i+width for i in x], ops, width, label="Op Income")

    plt.xticks([i+width/2 for i in x], segs, rotation=45, ha="right")
    plt.ylabel("Millions USD")
    plt.title(f"{ticker} – Segment Revenue vs Op Income ({label})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{ticker}_segments.png", dpi=200)
    plt.close()

def process_ticker(ticker: str, cik_map: dict[str, str]) -> None:
    if ticker not in cik_map:
        print(f"[{ticker}] not found in CIK list."); return
    cik = cik_map[ticker]
    print(f"[{ticker}] CIK {cik}")

    rev_json = select_concept_json(cik, REV_TAGS)
    if not rev_json:
        print(f"[{ticker}] no revenue data."); return
    op_json = select_concept_json(cik, OPINC_TAGS)
    if not op_json:
        print(f"[{ticker}] no op-income data."); return

    rev_df = json_to_df(rev_json, "revenue")
    op_df  = json_to_df(op_json, "op_income")

    rev_fy = keep_last_n_fy(rev_df, YEARS_BACK)
    op_fy  = keep_last_n_fy(op_df, YEARS_BACK)
    ttm_df = build_ttm(rev_df, op_df)

    fy_merged = rev_fy.merge(op_fy, on=["segment","end"], how="outer")
    full_df = pd.concat([fy_merged, ttm_df], ignore_index=True).fillna(0)

    # ── SAFE EXIT if no segment column ------------------------------------------------
    if "segment" not in full_df.columns or full_df.empty:
        print(f"[{ticker}] segment column missing or no segment facts – skipping.")
        return
    # ----------------------------------------------------------------------------------

    latest = full_df["end"].max()
    full_df["fiscal_label"] = full_df["end"].apply(
        lambda d: "TTM" if ttm_df.size and d==latest else str(fiscal_year(d)))

    catalog = update_catalog(ticker, full_df[["segment","end"]])
    for seg in catalog:
        if seg not in full_df["segment"].values:
            full_df.loc[len(full_df)] = [
                seg, latest, 0, 0,
                "TTM" if ttm_df.size else str(fiscal_year(latest))
            ]

    cols = ["segment","end","revenue","op_income","fiscal_label"]
    full_df[cols].to_csv(f"{ticker}_segments.csv", index=False)
    plot_segments(full_df, ticker)
    print(f"[{ticker}] ✓ CSV + chart saved")
    time.sleep(PAUSE_SEC)

    if VERBOSE:      # optional diagnostic dump
        print(full_df.head())

def main() -> None:
    cik_map = load_ticker_cik_map()
    for tkr in TICKERS:
        try:
            process_ticker(tkr.upper(), cik_map)
        except Exception as e:
            print(f"[{tkr}] ERROR → {e}")

if __name__ == "__main__":
    main()

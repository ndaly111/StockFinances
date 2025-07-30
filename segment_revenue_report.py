#!/usr/bin/env python
"""
sec_segment_data.py
===================

Pulls revenue & operating-income by business segment for any US ticker
directly from the SEC’s XBRL “companyconcept” API, then produces a
grouped-bar chart (last 3 FYs + TTM) ready to embed on your per-ticker
website page.

This script will:
  • Map ticker → CIK using the SEC JSON list.
  • Call the companyconcept endpoint twice per ticker — once for revenue and
    once for operating income — asking only for facts tagged with
    StatementBusinessSegmentsAxis.
  • Clean the JSON, convert numbers to millions, keep the last three fiscal
    years and the most-recent quarter, then stitch those into a TTM line.
  • Save a tidy CSV and render a chart that puts every segment on the same
    $-scale (so you can see which segments really matter).
  • Respect the SEC’s rate limit by sleeping between calls.

USAGE
-----

1. pip install requests pandas matplotlib python-dateutil
2. Edit TICKERS below (['AAPL','MSFT','TSLA', …])
3. python sec_segment_data.py
   –> CSV:   <TICKER>_segments.csv
   –> Chart: <TICKER>_segments.png

"""

import time
import json
import requests
import re
from datetime import datetime, date
from dateutil import parser as dtp
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ────────────────────────────────────────────────────────────────────
TICKERS = ["AAPL", "MSFT", "TSLA"]          # ← add your tickers here
USER_AGENT = "SegmentScraper/1.0 (email@example.com)"  # provide your email
PAUSE_SEC   = 0.25                          # SEC polite pause (seconds)
YEARS_BACK  = 3                             # how many fiscal years to keep
REV_TAGS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "SalesRevenueNet",
    "Revenues",
]
OPINC_TAGS = [
    "SegmentReportingInformationOperatingIncomeLoss",
    "OperatingIncomeLoss",
]
AXIS = "StatementBusinessSegmentsAxis"
# ────────────────────────────────────────────────────────────────────


def load_ticker_cik_map() -> dict[str, str]:
    """Download SEC's JSON list and build a mapping from ticker to CIK.

    The SEC publishes a JSON list of all company tickers and their corresponding
    CIK codes. We fetch that list and produce a dictionary keyed by ticker
    symbol (in uppercase) with values being zero-padded, 10-digit CIK strings.
    """
    url = "https://www.sec.gov/files/company_tickers.json"
    response = requests.get(url, headers={"User-Agent": USER_AGENT})
    response.raise_for_status()
    data = response.json()
    # JSON is a dict with numeric keys "0","1",…
    return {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in data.values()}


def fetch_concept(cik: str, concept: str) -> dict | None:
    """Return full JSON for one concept (revenue or operating income) or None.

    We query the SEC’s companyconcept endpoint by CIK and concept name and
    return the parsed JSON if the request succeeds (HTTP 200). Otherwise,
    None is returned. We respect SEC’s rate limits via global PAUSE_SEC.
    """
    url = (
        f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}"
        f"/us-gaap/{concept}.json?dimension={AXIS}"
    )
    response = requests.get(url, headers={"User-Agent": USER_AGENT})
    if response.status_code == 200:
        return response.json()
    return None


def select_concept_json(cik: str, tag_list: list[str]) -> dict | None:
    """Try each tag in order until SEC returns JSON with 'units'.

    Given a list of concept tags (e.g., multiple possible GAAP tags for
    revenue), this function attempts to fetch the concept JSON for each tag
    until a result with a "units" field is found. If none of the tags
    provide data, None is returned.
    """
    for tag in tag_list:
        data = fetch_concept(cik, tag)
        if data and "units" in data:
            return data
        time.sleep(PAUSE_SEC)
    return None


def json_to_df(data: dict, value_label: str) -> pd.DataFrame:
    """
    Convert SEC concept JSON to a tidy DataFrame with columns = ['segment','end', value_label].

    Only USD units are considered. Consolidated totals (where segment
    information is absent) and data from forms other than 10-K or 10-Q are
    skipped. Segment names are cleaned by stripping namespaces and the
    trailing "Member" suffix.
    """
    records: list[dict] = []
    for unit, items in data.get("units", {}).items():
        if unit != "USD":
            continue
        for item in items:
            # Skip entries without segment info (consolidated totals)
            if item.get("segment") is None:
                continue
            # Skip filings that aren't 10-K or 10-Q (e.g., 8-K, etc.)
            if item.get("form") not in ("10-K", "10-Q"):
                continue
            end_date = dtp.parse(item["end"]).date()
            # Convert to millions
            value_millions = item["val"] / 1_000_000
            member = item["segment"]["member"]
            # Human-readable: strip namespace + "Member"
            seg_name = re.sub(r".*?:", "", member).replace("Member", "")
            records.append({
                "segment": seg_name,
                "end": end_date,
                value_label: value_millions,
            })
    # If no records were captured, return an empty DataFrame with expected columns.
    if not records:
        return pd.DataFrame(columns=["segment", "end", value_label])
    return pd.DataFrame(records)


def fiscal_year(date_obj: date) -> int:
    """Return the fiscal year for a given date.

    This implementation assumes that fiscal years end within the calendar year
    indicated by the provided date. For example, a record ended on
    2023-09-30 will be labeled FY2023.
    """
    return date_obj.year


def keep_last_n_fy(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Filter a DataFrame to only keep rows within the last `n` fiscal years.

    The distinct fiscal years present in the data are collected, sorted
    descending, and the most recent `n` are kept. Only rows matching
    those fiscal years remain.
    """
    # If the input DataFrame is empty or lacks the 'end' column, return it as-is.
    if df.empty or "end" not in df.columns:
        # Return a copy to avoid modifying the original upstream
        return df.copy()
    # Collect distinct fiscal years and keep the latest `n` of them
    fy_list = sorted({fiscal_year(d) for d in df["end"]}, reverse=True)[:n]
    # Filter rows whose fiscal year is in fy_list. Use .loc to preserve columns on empty result.
    mask = df["end"].apply(lambda d: fiscal_year(d) in fy_list)
    return df.loc[mask].copy()


def latest_quarter(df: pd.DataFrame) -> pd.DataFrame:
    """Return the most-recent quarter row(s) for each segment.

    This function is unused in the current script but could be used to
    retrieve the latest quarter’s data for each segment. It may be helpful
    if future modifications require quarter granularity outside of TTM.
    """
    latest = df["end"].max()
    return df[df["end"] == latest]


def build_ttm(rev_df: pd.DataFrame, op_df: pd.DataFrame) -> pd.DataFrame:
    """Combine the last FY with the latest quarter to approximate trailing-twelve-months.

    For each segment, TTM is constructed by adding the last full fiscal year
    revenue and operating income to the most-recent quarter’s values (if they
    are different quarters). If the most recent quarter corresponds to the
    fiscal year end, an empty DataFrame is returned, meaning no separate
    TTM row is needed.
    """
    if rev_df.empty or op_df.empty:
        return pd.DataFrame()
    fy_latest = rev_df["end"].max()  # FY end date
    q_latest = op_df["end"].max()   # could be same as FY; okay
    # If the latest quarter is the same as the FY end, TTM is redundant.
    if fy_latest == q_latest:
        return pd.DataFrame()
    # Build TTM for each segment
    ttm_records: list[dict] = []
    segments = set(rev_df["segment"]) | set(op_df["segment"])
    for seg in segments:
        rev_fy = rev_df.loc[(rev_df["segment"] == seg) & (rev_df["end"] == fy_latest), "revenue"].sum()
        rev_q  = rev_df.loc[(rev_df["segment"] == seg) & (rev_df["end"] == q_latest), "revenue"].sum()
        op_fy  = op_df.loc[(op_df["segment"] == seg) & (op_df["end"] == fy_latest), "op_income"].sum()
        op_q   = op_df.loc[(op_df["segment"] == seg) & (op_df["end"] == q_latest), "op_income"].sum()
        if rev_fy or rev_q or op_fy or op_q:
            ttm_records.append({
                "segment": seg,
                "end": q_latest,  # Label TTM by latest quarter end date
                "revenue": rev_fy + rev_q,
                "op_income": op_fy + op_q,
            })
    return pd.DataFrame(ttm_records)


def plot_segments(df: pd.DataFrame, ticker: str) -> None:
    """
    Create a grouped-bar chart (Revenue vs Operating Income) by segment for the TTM.

    The chart displays revenue and operating income side-by-side for each
    segment. Only the TTM data is plotted to simplify visualization.
    """
    # Nothing to plot if the DataFrame has no rows or lacks the required columns
    required_cols = {"segment", "fiscal_label", "revenue", "op_income"}
    if df.empty or not required_cols.issubset(set(df.columns)):
        print(f"[{ticker}] no data to chart.")
        return

    # Determine which fiscal label to plot: use TTM if available, otherwise the latest fiscal year
    if "TTM" in df["fiscal_label"].values:
        target_label = "TTM"
    else:
        # Find the maximum numeric fiscal year label
        # Convert year-like strings to integers safely
        fy_labels = [int(label) for label in df["fiscal_label"] if label.isdigit()]
        if not fy_labels:
            print(f"[{ticker}] no valid fiscal year labels to chart.")
            return
        target_label = str(max(fy_labels))

    # Filter data for the chosen fiscal label
    df_plot = df[df["fiscal_label"] == target_label]
    if df_plot.empty:
        print(f"[{ticker}] no data to chart for {target_label}.")
        return

    # Prepare long-form data for plotting
    df_long = df_plot.melt(id_vars=["segment", "fiscal_label"],
                           value_vars=["revenue", "op_income"],
                           var_name="metric", value_name="value")
    unique_segments = df_long["segment"].unique()
    x_indices = range(len(unique_segments))
    width = 0.35

    # Extract values for revenue and operating income for each segment, preserving order
    rev_vals = [df_long[(df_long["metric"] == "revenue") & (df_long["segment"] == seg)]["value"].sum() for seg in unique_segments]
    op_vals  = [df_long[(df_long["metric"] == "op_income") & (df_long["segment"] == seg)]["value"].sum() for seg in unique_segments]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(x_indices, rev_vals, width, label="Revenue", color="#1f77b4")
    plt.bar([i + width for i in x_indices], op_vals, width, label="Op Income", color="#ff7f0e")

    plt.xticks([i + width / 2 for i in x_indices], unique_segments, rotation=45, ha="right")
    plt.ylabel("Millions USD")
    title_label = "TTM" if target_label == "TTM" else f"FY{target_label}"
    plt.title(f"{ticker} – Segment Revenue vs Operating Income ({title_label})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{ticker}_segments.png", dpi=200)
    plt.close()


def process_ticker(ticker: str, cik_map: dict[str, str]) -> None:
    """
    Process a single ticker: fetch data, compute fiscal-year and TTM, save CSV and chart.

    This function orchestrates the full pipeline for a given ticker. It
    retrieves revenue and operating income data, keeps only the last few
    fiscal years, constructs TTM values, concatenates data, writes to CSV,
    and generates the chart.
    """
    if ticker not in cik_map:
        print(f"[{ticker}] not found in CIK list.")
        return

    cik = cik_map[ticker]
    print(f"[{ticker}] CIK {cik}")

    # 1) Revenue
    rev_json = select_concept_json(cik, REV_TAGS)
    if not rev_json:
        print(f"[{ticker}] no revenue concept with segment axis.")
        return
    rev_df = json_to_df(rev_json, "revenue")

    # 2) Operating income
    op_json = select_concept_json(cik, OPINC_TAGS)
    if not op_json:
        print(f"[{ticker}] no operating-income concept with segment axis.")
        return
    op_df = json_to_df(op_json, "op_income")

    # 3) Keep last N fiscal years
    rev_fy = keep_last_n_fy(rev_df, YEARS_BACK)
    op_fy  = keep_last_n_fy(op_df, YEARS_BACK)

    # 4) Build TTM and append to FY data
    ttm_df = build_ttm(rev_df, op_df)
    # Merge revenue and op income on segment & end date, outer join to keep all rows
    fy_merged = rev_fy.merge(op_fy, on=["segment", "end"], how="outer")
    full_df = pd.concat([fy_merged, ttm_df], ignore_index=True).fillna(0)

    # 5) Add fiscal year labels
    # Determine the latest date across all data to label TTM.
    latest_date = full_df["end"].max() if not full_df.empty else None

    def label_fiscal(d: date) -> str:
        # If this row's end date equals the latest date and a separate TTM
        # DataFrame exists (ttm_df is not empty), label as TTM; otherwise use FY.
        if latest_date and d == latest_date and not ttm_df.empty:
            return "TTM"
        return str(fiscal_year(d))

    full_df["fiscal_label"] = full_df["end"].apply(label_fiscal)

    # 6) Reorder columns for consistency and save CSV and chart
    column_order = ["segment", "end", "revenue", "op_income", "fiscal_label"]
    # Use intersection to avoid missing columns if data is empty
    ordered_cols = [c for c in column_order if c in full_df.columns]
    csv_filename = f"{ticker}_segments.csv"
    full_df[ordered_cols].to_csv(csv_filename, index=False)
    plot_segments(full_df, ticker)
    print(f"[{ticker}] ✓ CSV + chart saved")
    time.sleep(PAUSE_SEC)


def main() -> None:
    """Main entry: load CIK map, iterate tickers, process each one."""
    cik_map = load_ticker_cik_map()
    for tkr in TICKERS:
        try:
            process_ticker(tkr.upper(), cik_map)
        except Exception as exc:
            print(f"[{tkr}] ERROR → {exc}")


if __name__ == "__main__":
    main()

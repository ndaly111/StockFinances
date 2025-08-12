#!/usr/bin/env python3
"""
consolidate_segment_tables.py

Collects every *_segments_table.html in charts/{ticker}/ and appends them
into one wide table with a Ticker column.  Saves as CSV by default.
Requirements: Python 3, BeautifulSoup (bs4), pandas.

Run it after your segment charts have been generated.
"""

import os
import pandas as pd
from bs4 import BeautifulSoup


def parse_one(html_path: str, ticker: str):
    """Extract rows and header from a single segments table."""
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if table is None:
        return None, []
    headers = [th.get_text(strip=True) for th in table.find_all("th")]
    rows = []
    for tr in table.find_all("tr")[1:]:
        cells = ["".join(td.stripped_strings) for td in tr.find_all("td")]
        if cells:
            rows.append([ticker] + cells)
    return headers, rows


def consolidate(charts_dir: str = "charts"):
    """Walk the charts directory, merging all segment tables."""
    all_rows = []
    columns = None
    for ticker in sorted(os.listdir(charts_dir)):
        t_dir = os.path.join(charts_dir, ticker)
        if not os.path.isdir(t_dir):
            continue
        candidates = [
            os.path.join(t_dir, f"{ticker}_segments_table.html"),
            os.path.join(t_dir, "segments_table.html"),
        ]
        html_path = next((p for p in candidates if os.path.exists(p)), None)
        if not html_path:
            continue
        header, rows = parse_one(html_path, ticker)
        if header is None:
            continue
        if columns is None:
            columns = ["Ticker"] + header
        all_rows.extend(rows)
    if not all_rows or columns is None:
        return None
    return pd.DataFrame(all_rows, columns=columns)


if __name__ == "__main__":
    df = consolidate("charts")
    if df is None:
        print("No segment tables found under charts/")
    else:
        out_csv = "combined_segment_tables.csv"
        df.to_csv(out_csv, index=False)
        print(f"Combined data saved to {out_csv}")
